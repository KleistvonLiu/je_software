#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, threading, time, re, logging
import gc
from collections import deque
from typing import List, Tuple, Optional, Any, Dict
from datetime import datetime
from queue import SimpleQueue, Empty
import pynput

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
import cv2
import numpy as np

from .utils.image_writer import AsyncImageWriter
from common_utils.ros2_qos import reliable_qos

# ====================== 小工具 ======================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name.strip('/'))


def stamp_to_ns(stamp) -> int:
    # 假设 stamp 有 sec / nanosec，且可能为 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

def _pack_depth_u16_to_rgb8(depth_u16: np.ndarray, order: str = "HI_LO", b_fill: int = 0) -> np.ndarray:
    """
    depth_u16: (H, W) uint16
    返回: (H, W, 3) uint8
      - order="HI_LO": R=高8位, G=低8位
      - order="LO_HI": R=低8位, G=高8位
      - B 通道用 b_fill 填充（0/255/或其他标记）
    """
    if depth_u16.dtype != np.uint16:
        raise ValueError("depth_u16 must be uint16")
    hi = ((depth_u16 >> 8) & 0xFF).astype(np.uint8)
    lo = (depth_u16 & 0xFF).astype(np.uint8)
    if order.upper() == "HI_LO":
        r, g = hi, lo
    elif order.upper() == "LO_HI":
        r, g = lo, hi
    else:
        raise ValueError("order must be 'HI_LO' or 'LO_HI'")
    b = np.full_like(r, np.uint8(b_fill))
    rgb = np.stack([r, g, b], axis=-1)  # (H,W,3)
    return rgb

# ====================== 高效多流对齐器 ======================
class MultiStreamAligner:
    """
    无锁摄取 → 工作线程独占本地窗口：
      - 回调 put_nowait 到 ingest 队列
      - step():
        1) drain 各 ingest → 本地 times / msgs（保持单调；乱序帧丢弃）
        2) t_ref = min(latest_i)（仅参考 ref_indices）
        3) 对每路推进“只前进”游标到 <= target 的最大位置，并在它与后一位择近；误差<=tol_i
        4) 成功返回 (t_ref, picks)，并剪枝到选中位置
      - 支持 max_window_ns 以限制窗口大小
    新增：
      - optional_indices：这些流若空，不阻塞对齐，也不做容差检查；picks 中对应位置为 None
      - non_consuming_indices：成功后不剪枝（低频可复用）
    """

    def __init__(
            self,
            num_streams: int,
            tolerances_ns: List[int],
            offsets_ns: Optional[List[int]] = None,
            max_window_ns: Optional[int] = None,
            *,
            logger: Optional[Any] = None,
            name: str = "aligner",
            debug: bool = False,
            log_every: int = 200,
            log_period_s: float = 5.0,
            ref_indices: Optional[List[int]] = None,
            non_consuming_indices: Optional[List[int]] = None,
            optional_indices: Optional[List[int]] = None,   # <-- 新增
    ):
        assert num_streams == len(tolerances_ns), "tolerances_ns length must match num_streams"
        self.N = int(num_streams)
        self.tolerances_ns = list(map(int, tolerances_ns))
        self.offsets_ns = list(map(int, offsets_ns)) if offsets_ns else [0] * self.N
        self.max_window_ns = int(max_window_ns) if max_window_ns else None

        # 参考流
        if ref_indices is None:
            self.ref_indices = list(range(self.N))
        else:
            self.ref_indices = sorted({int(i) for i in ref_indices if 0 <= int(i) < self.N})
            assert self.ref_indices, "ref_indices cannot be empty"

        # 可选/非消耗流
        self.optional_set = set(int(i) for i in (optional_indices or []) if 0 <= int(i) < self.N)
        non_consuming_set = set(int(i) for i in (non_consuming_indices or []) if 0 <= int(i) < self.N)
        self.consume_mask = [False if i in non_consuming_set else True for i in range(self.N)]

        self.ingest: List[SimpleQueue] = [SimpleQueue() for _ in range(self.N)]
        self.times: List[List[int]] = [[] for _ in range(self.N)]
        self.msgs: List[List[Any]] = [[] for _ in range(self.N)]
        self.cursor: List[int] = [-1] * self.N

        # ---- 日志 & 统计 ----
        self.debug = bool(debug)
        self.log_every = int(max(1, log_every))
        self.log_period_s = float(max(0.5, log_period_s))
        self._last_log_t = time.monotonic()
        self.log = logger if logger is not None else logging.getLogger(f"{__name__}.{name}")

        self.stats: Dict[str, Any] = {
            "enq": [0] * self.N,
            "drain": [0] * self.N,
            "append": [0] * self.N,
            "drop_ooo": [0] * self.N,
            "prune": [0] * self.N,
            "consumed": [0] * self.N,
            "step_attempt": 0,
            "step_success": 0,
            "last_fail": "",
            "last_fail_detail": None,
        }

        self._dbg("initialized: N=%d, max_window_ns=%s, tolerances=%s, offsets=%s, optional=%s",
                  self.N, str(self.max_window_ns), self.tolerances_ns, self.offsets_ns, sorted(self.optional_set))

    # ---------- 日志工具 ----------
    def _dbg(self, msg: str, *args):
        if self.debug and hasattr(self.log, "debug"):
            try:
                self.log.debug(msg % args if args else msg)
            except Exception:
                pass

    def _info(self, msg: str, *args):
        if hasattr(self.log, "info"):
            try:
                self.log.info(msg % args if args else msg)
            except Exception:
                pass

    def _warn(self, msg: str, *args):
        if hasattr(self.log, "warn"):
            try:
                self.log.warn(msg % args if args else msg)
            except Exception:
                if hasattr(self.log, "warning"):
                    try:
                        self.log.warning(msg % args if args else msg)
                    except Exception:
                        pass

    def _err(self, msg: str, *args):
        if hasattr(self.log, "error"):
            try:
                self.log.error(msg % args if args else msg)
            except Exception:
                pass

    def _maybe_log_summary(self):
        t = time.monotonic()
        need = (self.stats["step_attempt"] % self.log_every == 0) or ((t - self._last_log_t) >= self.log_period_s)
        if not need:
            return
        self._last_log_t = t
        hit = (self.stats["step_success"] / self.stats["step_attempt"]) if self.stats["step_attempt"] else 0.0
        lens = [len(ti) for ti in self.times]
        self._info(
            "[align] attempts=%d, success=%d, hit=%.1f%%, window_len=%s, "
            "drop_ooo=%s, prune=%s, consumed=%s, last_fail=%s %s",
            self.stats["step_attempt"], self.stats["step_success"], 100.0 * hit,
            lens, self.stats["drop_ooo"], self.stats["prune"], self.stats["consumed"],
            self.stats["last_fail"],
            f"detail={self.stats['last_fail_detail']}" if self.stats["last_fail_detail"] else "",
        )

    # ---------- 接口 ----------
    def put_nowait(self, i: int, t_ns: int, msg: Any):
        try:
            self.ingest[i].put_nowait((t_ns, msg))
            self.stats["enq"][i] += 1
            if self.debug and (self.stats["enq"][i] % (self.log_every // 2 or 1) == 0):
                self._dbg("enqueued stream[%d] total=%d last_ts=%d", i, self.stats["enq"][i], t_ns)
        except Exception as e:
            self._warn("enqueue failed stream[%d]: %s", i, e)

    def _drain_one(self, i: int):
        qi = self.ingest[i]
        ti = self.times[i]
        mi = self.msgs[i]
        last = ti[-1] if ti else -1

        drained = 0
        appended = 0
        drop_ooo = 0

        while True:
            try:
                t_ns, msg = qi.get_nowait()
                drained += 1
            except Empty:
                break
            if t_ns >= last:
                ti.append(t_ns)
                mi.append(msg)
                last = t_ns
                appended += 1
            else:
                drop_ooo += 1

        self.stats["drain"][i] += drained
        self.stats["append"][i] += appended
        self.stats["drop_ooo"][i] += drop_ooo

        if self.debug and drained:
            self._dbg("drain stream[%d]: drained=%d, appended=%d, drop_ooo=%d, window_len=%d, last_ts=%d",
                      i, drained, appended, drop_ooo, len(ti), last)

        if self.max_window_ns and len(ti) >= 2:
            cutoff = last - self.max_window_ns
            drop_k, n = 0, len(ti)
            while drop_k < n and ti[drop_k] < cutoff:
                drop_k += 1
            if drop_k > 0:
                del ti[:drop_k]
                del mi[:drop_k]
                self.cursor[i] -= drop_k
                if self.cursor[i] < -1:
                    self.cursor[i] = -1
                self.stats["prune"][i] += drop_k
                self._dbg("prune stream[%d]: pruned=%d, new_window_len=%d, cursor=%d",
                          i, drop_k, len(ti), self.cursor[i])

    def _drain(self):
        for i in range(self.N):
            self._drain_one(i)

    @staticmethod
    def _advance_and_pick(times: List[int], cur: int, target: int) -> Tuple[int, int]:
        n = len(times)
        while cur + 1 < n and times[cur + 1] <= target:
            cur += 1
        pick = 0 if cur < 0 else cur
        if cur + 1 < n:
            left_ts, right_ts = times[pick], times[cur + 1]
            if abs(right_ts - target) < abs(left_ts - target):
                pick = cur + 1
        return pick, cur

    def step(self) -> Optional[Tuple[int, List[Optional[Tuple[int, Any]]]]]:
        """尝试对齐；成功返回 (t_ref, picks)。optional 流在无数据时对应 picks 元素为 None。"""
        self._drain()
        self.stats["step_attempt"] += 1

        # 1) 准备参考流的 latest，若参考流为空则直接失败
        ref_latest = []
        for i in self.ref_indices:
            if not self.times[i]:
                self.stats["last_fail"] = f"waiting_ref_stream_{i}"
                self.stats["last_fail_detail"] = {"stream": i, "reason": "empty_ref"}
                self._maybe_log_summary()
                return None
            ref_latest.append(self.times[i][-1])
        t_ref = min(ref_latest)

        picks_idx: List[int] = [-1] * self.N
        picks: List[Optional[Tuple[int, Any]]] = [None] * self.N

        for i in range(self.N):
            # 可选流且当前无数据：跳过，不做容差检查
            if (i in self.optional_set) and (not self.times[i]):
                continue

            # 必需流没有数据：失败
            if not self.times[i]:
                self.stats["last_fail"] = f"waiting_stream_{i}"
                self.stats["last_fail_detail"] = {"stream": i, "reason": "empty"}
                self._maybe_log_summary()
                return None

            target = t_ref + self.offsets_ns[i]
            pick_i, cur_after = self._advance_and_pick(self.times[i], self.cursor[i], target)
            err = abs(self.times[i][pick_i] - target)

            # 非可选流做容差检查；可选流有数据时也做容差检查（以保持质量）
            if err > self.tolerances_ns[i]:
                self.stats["last_fail"] = "tolerance_exceeded"
                self.stats["last_fail_detail"] = {
                    "stream": i,
                    "target_ns": int(target),
                    "picked_ts_ns": int(self.times[i][pick_i]),
                    "error_ns": int(err),
                    "tolerance_ns": int(self.tolerances_ns[i]),
                }
                self._maybe_log_summary()
                return None

            picks_idx[i] = pick_i
            picks[i] = (self.times[i][pick_i], self.msgs[i][pick_i])
            self.cursor[i] = cur_after

        # 成功：剪枝（仅剪枝有 pick 且 consume_mask=True 的流）
        for i in range(self.N):
            k = picks_idx[i]
            if k < 0:
                continue
            if self.consume_mask[i]:
                del self.times[i][:k + 1]
                del self.msgs[i][:k + 1]
                self.cursor[i] -= (k + 1)
                if self.cursor[i] < -1:
                    self.cursor[i] = -1
                self.stats["consumed"][i] += (k + 1)

        self.stats["step_success"] += 1
        self.stats["last_fail"] = ""
        self.stats["last_fail_detail"] = None

        self._maybe_log_summary()
        return t_ref, picks

    # ---------- 公共统计接口 ----------
    def metrics(self) -> Dict[str, Any]:
        hit = (self.stats["step_success"] / self.stats["step_attempt"]) if self.stats["step_attempt"] else 0.0
        return {
            "streams": self.N,
            "window_len": [len(ti) for ti in self.times],
            "enqueued": list(self.stats["enq"]),
            "drained": list(self.stats["drain"]),
            "appended": list(self.stats["append"]),
            "dropped_out_of_order": list(self.stats["drop_ooo"]),
            "pruned_by_window": list(self.stats["prune"]),
            "consumed": list(self.stats["consumed"]),
            "attempts": int(self.stats["step_attempt"]),
            "success": int(self.stats["step_success"]),
            "hit_ratio": hit,
            "last_fail": self.stats["last_fail"],
            "last_fail_detail": self.stats["last_fail_detail"],
        }

    def reset_metrics(self):
        for k in ("enq", "drain", "append", "drop_ooo", "prune"):
            self.stats[k] = [0] * self.N
        self.stats["step_attempt"] = 0
        self.stats["step_success"] = 0
        self.stats["last_fail"] = ""
        self.stats["last_fail_detail"] = None


# ====================== ROS2 节点 ======================

class Manager(Node):
    """
    订阅多路彩色/深度/关节/触觉；保存线程使用 min-latest + 单调游标 对齐；锁外转换/写盘。
    触觉(tactile)改为可选：无触觉消息时不会阻塞同步。
    """

    def __init__(self):
        super().__init__('manager')

        # ---------- 参数 ----------
        # 话题
        self.declare_parameter('color_topics', [])
        self.declare_parameter('depth_topics', [])
        self.declare_parameter('color_topics_csv', '/camera_01/color/image_raw,/camera_03/color/image_raw,/camera_04/color/image_raw')
        self.declare_parameter('depth_topics_csv', '/camera_01/depth/image_raw,/camera_03/depth/image_raw,/camera_04/depth/image_raw')

        # joint/tactile：多路 + 兼容单路
        self.declare_parameter('joint_state_topics', [])
        self.declare_parameter('joint_state_topics_csv', '')
        self.declare_parameter('tactile_topics', [])
        self.declare_parameter('tactile_topics_csv', '')
        self.declare_parameter('joint_state_topic', '/robot/joint_states')  # legacy
        self.declare_parameter('tactile_topic', '/tactile_data')            # legacy

        # 频率与容差（ms）
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('image_tolerance_ms', 15.0)  # ≤ 半帧
        self.declare_parameter('joint_tolerance_ms', 15.0)
        self.declare_parameter('tactile_tolerance_ms', 60.0)

        # 窗口与目录
        self.declare_parameter('queue_seconds', 2.0)
        self.declare_parameter('save_dir', os.path.expanduser('/home/test/jemotor/log/'))
        self.declare_parameter('session_name', '')
        self.declare_parameter('save_depth', True)

        # 其他
        self.declare_parameter('use_ros_time', True)
        self.declare_parameter('color_jpeg_quality', 95)
        self.declare_parameter('do_calculate_hz', True)
        self.declare_parameter('stats_window_s', 5.0)
        self.declare_parameter('stats_log_period_s', 2.0)
        self.declare_parameter('episode_idx', 0)
        self.declare_parameter('mode', 1)

        # 偏置（ms）
        self.declare_parameter('color_offsets_ms', [])
        self.declare_parameter('depth_offsets_ms', [])
        self.declare_parameter('joint_offset_ms', 0.0)
        self.declare_parameter('tactile_offset_ms', 0.0)

        p = self.get_parameter

        # 读取话题参数
        color_topics = list(p('color_topics').value or [])
        depth_topics = list(p('depth_topics').value or [])
        if not color_topics:
            csv = p('color_topics_csv').value or ''
            if csv.strip():
                color_topics = [s.strip() for s in csv.split(',') if s.strip()]
        if not depth_topics:
            csv = p('depth_topics_csv').value or ''
            if csv.strip():
                depth_topics = [s.strip() for s in csv.split(',') if s.strip()]

        # 过滤空字符串话题
        color_topics = [t.strip() for t in color_topics if t and t.strip()]
        depth_topics = [t.strip() for t in depth_topics if t and t.strip()]

        self.color_topics: List[str] = color_topics
        self.depth_topics: List[str] = depth_topics

        # 读取 joint/tactile 多路参数（向后兼容）
        joint_topics = list(p('joint_state_topics').value or [])
        if not joint_topics:
            csv = p('joint_state_topics_csv').value or ''
            if csv.strip():
                joint_topics = [s.strip() for s in csv.split(',') if s.strip()]
        if not joint_topics:
            legacy = p('joint_state_topic').value
            if legacy and str(legacy).strip():
                joint_topics = [str(legacy).strip()]
        self.joint_topics: List[str] = [t for t in joint_topics if t and t.strip()]

        tactile_topics = list(p('tactile_topics').value or [])
        if not tactile_topics:
            csv = p('tactile_topics_csv').value or ''
            if csv.strip():
                tactile_topics = [s.strip() for s in csv.split(',') if s.strip()]
        if not tactile_topics:
            legacy = p('tactile_topic').value
            if legacy and str(legacy).strip():
                tactile_topics = [str(legacy).strip()]
        self.tactile_topics: List[str] = [t for t in tactile_topics if t and t.strip()]

        # 频率/容差
        self.rate_hz: float = float(p('rate_hz').value)
        image_tol_ns = int(float(p('image_tolerance_ms').value) * 1e6)

        def _as_list_ns(param_name: str, n: int) -> List[int]:
            raw = p(param_name).value
            if isinstance(raw, (list, tuple)):
                seq = list(raw)
                if len(seq) == 0:
                    return [0] * n
                if len(seq) == 1 and n > 1:
                    return [int(float(seq[0]) * 1e6)] * n
                assert len(seq) == n, f"{param_name} length must == {n} (got {len(seq)})"
                return [int(float(x) * 1e6) for x in seq]
            else:
                return [int(float(raw) * 1e6)] * n

        # 窗口/目录
        self.queue_seconds: float = float(p('queue_seconds').value)
        self.save_dir: str = p('save_dir').value
        session_name: str = p('session_name').value
        self.save_depth: bool = bool(p('save_depth').value)
        self.use_ros_time: bool = bool(p('use_ros_time').value)
        self.jpeg_quality: int = int(p('color_jpeg_quality').value)

        # 偏置 -> ns 列表
        def _as_list_offset_ns(param_name: str, n: int) -> List[int]:
            raw = p(param_name).value
            if isinstance(raw, (list, tuple)):
                seq = list(raw)
                if len(seq) == 0:
                    return [0] * n
                if len(seq) == 1 and n > 1:
                    return [int(float(seq[0]) * 1e6)] * n
                assert len(seq) == n, f"{param_name} length must == {n} (got {len(seq)})"
                return [int(float(x) * 1e6) for x in seq]
            else:
                return [int(float(raw) * 1e6)] * n

        # 统计频率
        self.stats_window_s: float = float(p('stats_window_s').value)
        self.stats_log_period_s: float = float(p('stats_log_period_s').value)
        self.do_calculate_hz: bool = bool(p('do_calculate_hz').value)
        self._attempt_win = 0
        self._success_win = 0
        self._save_times = deque()
        self._last_rate_log_t = time.perf_counter()

        self.episode_idx: int = int(p('episode_idx').value)
        self.mode: int = int(p('mode').value)
        self.frame_idx = 0

        if not session_name:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.save_dir, sanitize(session_name))
        ensure_dir(self.session_dir)

        # 相机目录
        self.cam_count = len(self.color_topics)
        self.cam_dirs = []
        for i in range(self.cam_count):
            cam_name = sanitize(self.color_topics[i])
            base = os.path.join(self.session_dir, f'cam_{i:02d}_{cam_name}')
            ensure_dir(base)
            ensure_dir(os.path.join(base, 'color'))
            if self.save_depth and (i < len(self.depth_topics) and self.depth_topics[i]):
                ensure_dir(os.path.join(base, 'depth'))
            self.cam_dirs.append(base)

        self.meta_dir = os.path.join(self.session_dir, 'meta')
        ensure_dir(self.meta_dir)

        self.get_logger().info(f"Session: {self.session_dir}")
        self.get_logger().info(f"Color topics: {self.color_topics}")
        self.get_logger().info(f"Depth  topics: {self.depth_topics}")
        self.get_logger().info(f"Joint topics:  {self.joint_topics}")
        self.get_logger().info(f"Tactile topics:{self.tactile_topics}")

        # ---------- 对齐器 ----------
        C = len(self.color_topics)
        D = len(self.depth_topics) if self.save_depth else 0
        J = len(self.joint_topics) if self.joint_topics else 0
        T = len(self.tactile_topics) if self.tactile_topics else 0

        self._idx_color = list(range(C))
        self._idx_depth = list(range(C, C + D))
        self._idx_joint = list(range(C + D, C + D + J))
        self._idx_tact  = list(range(C + D + J, C + D + J + T))
        num_streams = C + D + J + T

        tolerances_ns: List[int] = []
        tolerances_ns += [image_tol_ns] * C
        tolerances_ns += [image_tol_ns] * D
        tolerances_ns += _as_list_ns('joint_tolerance_ms', J) if J else []
        tolerances_ns += _as_list_ns('tactile_tolerance_ms', T) if T else []

        color_offsets_ms = list(p('color_offsets_ms').value or [])
        depth_offsets_ms = list(p('depth_offsets_ms').value or [])
        offsets_ns: List[int] = []
        offsets_ns += ([int(ms * 1e6) for ms in color_offsets_ms] if color_offsets_ms and len(color_offsets_ms) == C else [0] * C)
        offsets_ns += ([int(ms * 1e6) for ms in depth_offsets_ms] if D and depth_offsets_ms and len(depth_offsets_ms) == D else [0] * D)
        offsets_ns += _as_list_offset_ns('joint_offset_ms', J) if J else []
        offsets_ns += _as_list_offset_ns('tactile_offset_ms', T) if T else []

        max_window_ns = int(self.queue_seconds * 1e9)
        camera_refs = self._idx_color + self._idx_depth
        non_consuming = list(self._idx_tact)          # tactile 非消耗
        optional_indices = list(self._idx_tact)       # tactile 可选（新增关键）

        self.aligner = MultiStreamAligner(
            num_streams=num_streams,
            tolerances_ns=tolerances_ns,
            offsets_ns=offsets_ns,
            max_window_ns=max_window_ns,
            logger=self.get_logger(),
            ref_indices=camera_refs,
            non_consuming_indices=non_consuming,
            optional_indices=optional_indices,  # 使无 tactile 数据也能同步
        )

        self.get_logger().info(
            f"rate={self.rate_hz}Hz, streams: C={C}, D={D}, J={J}, T={T}, total={num_streams}"
        )

        # ---------- 订阅（区分 QoS） ----------
        cg_color = ReentrantCallbackGroup()
        cg_depth = ReentrantCallbackGroup()

        for i, topic in enumerate(self.color_topics):
            self.create_subscription(Image, topic, self._mk_color_cb(i), reliable_qos, callback_group=cg_color)

        for j in range(D):
            topic = self.depth_topics[j]
            if topic and self.save_depth:
                self.create_subscription(Image, topic, self._mk_depth_cb(j), reliable_qos, callback_group=cg_depth)

        cg_joint = ReentrantCallbackGroup()
        cg_tact  = ReentrantCallbackGroup()
        for k, topic in enumerate(self.joint_topics):
            self.create_subscription(JointState, topic, self._mk_joint_cb(k), reliable_qos, callback_group=cg_joint)
        for k, topic in enumerate(self.tactile_topics):
            self.create_subscription(Float32MultiArray, topic, self._mk_tactile_cb(k), reliable_qos, callback_group=cg_tact)

        # 工具
        self.bridge = CvBridge()

        # 图像保存器
        self.image_writer = AsyncImageWriter(
            num_processes=0,
            num_threads=12,
        )

        # 元数据缓冲
        self.meta_buffer = []
        self.meta_jsonl_path = os.path.join(self.session_dir, 'meta.jsonl')

        # === 录制开关与键盘监听 ===
        self._record_enabled = False
        self._kb_listener = None
        self._last_pause_log = 0.0

        self.get_logger().info("Keyboard control: right Alt = start/resume, right Ctrl = stop")

        try:
            from pynput import keyboard as _kb
            self._pynput_kb = _kb
            self._kb_listener = _kb.Listener(on_press=self._on_key_press)
            self._kb_listener.daemon = True
            self._kb_listener.start()
        except Exception as e:
            self.get_logger().warn(
                f"Keyboard control unavailable (install pynput or ensure GUI session). "
                f"Recording remains paused until Ctrl event can be captured. detail={e}"
            )

        # 保存线程
        self.sample_idx = 0
        self._stop_evt = threading.Event()
        if self.mode == 0:
            self.get_logger().error("Select a mode!!!")
        elif self.mode == 1:
            self.worker = threading.Thread(target=self._save_loop, name='logger-aligner', daemon=False)
            self.worker.start()
        elif self.mode == 2:
            self.worker = threading.Thread(target=self._inference_loop, name='logger-aligner', daemon=False)
            self.worker.start()

    def _on_key_press(self, key):
        """Alt_R = start/resume，Ctrl_R = stop"""
        try:
            kb = getattr(self, "_pynput_kb", None)
            if kb is None:
                return

            if key == kb.Key.alt_r:
                if not self._record_enabled:
                    self._record_enabled = True
                    self.get_logger().info("Recording ENABLED by right Alt")

            elif key == kb.Key.ctrl_r:
                if self._record_enabled:
                    prev_episode_idx = self.episode_idx
                    try:
                        episode_dir = os.path.join(self.save_dir, f"episode_{prev_episode_idx:06d}")
                        self._flush_meta_buffer(episode_dir)
                    except Exception as e:
                        self.get_logger().warn(f"Flush meta_buffer on stop failed: {e}")
                    self._record_enabled = False
                    self.episode_idx += 1
                    self.frame_idx = 0
                    self.get_logger().info("Recording DISABLED by right Ctrl")
                    self._pending_safe_log = True

                    # Drain image writer queue and run a best-effort cleanup to reduce resident memory.
                    try:
                        iw = getattr(self, 'image_writer', None)
                        if iw is not None:
                            self.get_logger().info("Draining image writer queue (wait_until_done)...")
                            try:
                                iw.wait_until_done()
                            except Exception as e:
                                self.get_logger().warn(f"image_writer.wait_until_done() failed: {e}")
                            # Keep writer available for reuse; if you want to fully stop threads uncomment stop()
                            # try:
                            #     iw.stop()
                            #     self.image_writer = None
                            # except Exception:
                            #     pass
                            gc.collect()
                    except Exception as e:
                        self.get_logger().warn(f"Post-stop cleanup failed: {e}")

        except Exception as e:
            self.get_logger().warn(f"Keyboard handler error: {e}")

    # ---------- 时间戳回退：0/无时间戳 → 本地时钟 ----------
    def _ns_from_header_or_clock(self, header) -> int:
        try:
            s = int(header.stamp.sec)
            ns = int(header.stamp.nanosec)
            if s != 0 or ns != 0:
                return s * 1_000_000_000 + ns
        except Exception:
            pass
        return self.get_clock().now().nanoseconds

    # ---------- 回调：无锁摄取 ----------
    def _mk_color_cb(self, i: int):
        def _cb(msg: Image):
            if self.use_ros_time:
                t_ns = self._ns_from_header_or_clock(msg.header)
            else:
                t_ns = self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_color[i], t_ns, msg)
        return _cb

    def _mk_depth_cb(self, j: int):
        def _cb(msg: Image):
            if self.use_ros_time:
                t_ns = self._ns_from_header_or_clock(msg.header)
            else:
                t_ns = self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_depth[j], t_ns, msg)
        return _cb

    def _mk_joint_cb(self, k: int):
        def _cb(msg: JointState):
            t_ns = self._ns_from_header_or_clock(msg.header)
            self.aligner.put_nowait(self._idx_joint[k], t_ns, msg)
        return _cb

    def _mk_tactile_cb(self, k: int):
        def _cb(msg: Float32MultiArray):
            t_ns = self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_tact[k], t_ns, msg)
        return _cb

    def _save_loop(self):
        period = 1.0 / max(1e-6, self.rate_hz)
        next_t = time.perf_counter()
        while rclpy.ok() and not self._stop_evt.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += period
            try:
                if not self._record_enabled:
                    _ = self.aligner.step()
                    if (now - self._last_pause_log) > 2.0:
                        self.get_logger().debug("Paused: press Alt_R to start/resume, Ctrl_R to stop.")
                        self._last_pause_log = now
                    if hasattr(self, '_pending_safe_log') and self._pending_safe_log:
                        if self.image_writer.is_idle():
                            self.get_logger().info("[SAFE] 所有数据已安全保存，可以安全退出程序！")
                            self._pending_safe_log = False
                    continue

                if self.do_calculate_hz:
                    self._attempt_win += 1
                out = self.aligner.step()
                if out is None:
                    # self.get_logger().warn("[SAFE] failed to align!")
                    continue
                if self.do_calculate_hz:
                    self._success_win += 1
                t_ref, picks = out
                self._save_once(t_ref, picks)
                if self.do_calculate_hz:
                    self._on_saved_stats()
            except Exception as e:
                self.get_logger().error(f"save loop error: {e}")

    # === 新增：统计函数 ===
    def _on_saved_stats(self):
        now = time.perf_counter()
        self._save_times.append(now)
        cutoff = now - self.stats_window_s
        while self._save_times and self._save_times[0] < cutoff:
            self._save_times.popleft()

        win_fps = 0.0
        inst_fps = 0.0
        if len(self._save_times) >= 2:
            dt_win = self._save_times[-1] - self._save_times[0]
            if dt_win > 0:
                win_fps = (len(self._save_times) - 1) / dt_win
            dt_inst = self._save_times[-1] - self._save_times[-2]
            if dt_inst > 0:
                inst_fps = 1.0 / dt_inst

        if (now - self._last_rate_log_t) >= self.stats_log_period_s:
            hit_ratio = (self._success_win / self._attempt_win) if self._attempt_win > 0 else 0.0
            self.get_logger().info(
                f"[save stats] inst_fps={inst_fps:.2f}, "
                f"win({self.stats_window_s:.1f}s)_fps={win_fps:.2f}, "
                f"align_hit_ratio={hit_ratio:.1%} "
                f"(attempts={self._attempt_win}, success={self._success_win})"
            )
            self._attempt_win = 0
            self._success_win = 0
            self._last_rate_log_t = now

    def _save_once(self, t_ref: int, picks: List[Optional[Tuple[int, Any]]]):
        idx = self.frame_idx
        self.frame_idx += 1

        episode_idx = getattr(self, 'episode_idx', 0)
        episode_dir = os.path.join(self.save_dir, f"episode_{episode_idx:06d}")
        ensure_dir(episode_dir)

        C = len(self._idx_color)
        D = len(self._idx_depth)
        J = len(self._idx_joint)
        T = len(self._idx_tact)

        color_picks = picks[0:C]
        depth_picks = picks[C:C + D] if D > 0 else []
        joint_picks = picks[C + D:C + D + J] if J > 0 else []
        tact_picks  = picks[C + D + J:C + D + J + T] if T > 0 else []

        # === 1. 图片异步写 ===
        image_fields = {}
        for cam_i, item in enumerate(color_picks):
            if item is None:
                continue
            t_ns, msg = item
            cam_name = sanitize(self.color_topics[cam_i])
            img_dir = os.path.join(episode_dir, "images", cam_name)
            ensure_dir(img_dir)
            fn = f"frame_{idx:06d}.png"
            fp = os.path.join(img_dir, fn)
            try:
                if getattr(msg, "encoding", "") != 'rgb8':
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                else:
                    img = self.bridge.imgmsg_to_cv2(msg)
            except Exception as e:
                # Conversion failed; log and continue
                self.get_logger().error(f"Failed to convert color msg for camera {cam_name}: {e}")
                continue
            try:
                iw = getattr(self, 'image_writer', None)
                if iw is None:
                    self.get_logger().error("No image_writer available to save color image")
                else:
                    iw.save_image(img, fp)
                    image_fields[cam_name] = os.path.relpath(fp, episode_dir)
            except Exception as e:
                self.get_logger().error(f"Failed to enqueue color image for saving {fp}: {e}")

        if self.save_depth and depth_picks:
            for dep_i, item in enumerate(depth_picks):
                if item is None:
                    continue
                t_ns, msg = item
                cam_name = sanitize(self.depth_topics[dep_i])
                img_dir = os.path.join(episode_dir, "images", cam_name)
                ensure_dir(img_dir)
                fn = f"frame_{idx:06d}.png"
                fp = os.path.join(img_dir, fn)
                try:
                    try:
                        depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                    except Exception:
                        depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
                except Exception as e:
                    self.get_logger().error(f"Failed to convert depth msg for camera {cam_name}: {e}")
                    continue
                try:
                    depth_rgb8 = _pack_depth_u16_to_rgb8(depth_u16, order="HI_LO", b_fill=0)
                except Exception as e:
                    self.get_logger().error(f"Failed to pack depth for camera {cam_name}: {e}")
                    continue
                try:
                    iw = getattr(self, 'image_writer', None)
                    if iw is None:
                        self.get_logger().error("No image_writer available to save depth image")
                    else:
                        iw.save_image(depth_rgb8, fp)
                        image_fields[cam_name] = os.path.relpath(fp, episode_dir)
                except Exception as e:
                    self.get_logger().error(f"Failed to enqueue depth image for saving {fp}: {e}")

        # === 2. 组装多路 joint/tactile 元数据 ===
        joints_meta = []
        for k, item in enumerate(joint_picks):
            if item is None:
                continue
            t_ns, js_msg = item
            joints_meta.append(dict(
                topic=self.joint_topics[k],
                stamp_ns=int(t_ns),
                name=list(getattr(js_msg, "name", [])),
                position=[float(x) for x in getattr(js_msg, "position", [])],
                velocity=[float(x) for x in getattr(js_msg, "velocity", [])],
                effort=[float(x) for x in getattr(js_msg, "effort", [])],
            ))

        tactiles_meta = []
        for k, item in enumerate(tact_picks):
            if item is None:
                continue
            t_ns, tact_msg = item
            tactiles_meta.append(dict(
                topic=self.tactile_topics[k],
                stamp_ns=int(t_ns),
                data=[float(x) for x in getattr(tact_msg, "data", [])],
            ))

        meta = dict(
            episode_idx=episode_idx,
            frame_index=idx,
            timestamp=time.time(),
            **image_fields,
            joints=joints_meta,
            tactiles=tactiles_meta,  # 可能为空列表
        )

        self.meta_buffer.append(meta)
        if len(self.meta_buffer) >= 100:
            self._flush_meta_buffer(episode_dir)
        return

    def _flush_meta_buffer(self, episode_dir=None):
        if not self.meta_buffer:
            return
        if episode_dir is None:
            episode_idx = getattr(self, 'episode_idx', 0)
            episode_dir = os.path.join(self.save_dir, f"episode_{episode_idx:06d}")
        meta_jsonl_path = os.path.join(episode_dir, 'meta.jsonl')
        old_lines = []
        old_keys = set()
        if os.path.exists(meta_jsonl_path):
            with open(meta_jsonl_path, 'r') as f:
                for line in f:
                    try:
                        old = json.loads(line)
                        key = (old.get('episode_idx'), old.get('frame_index'))
                        old_keys.add(key)
                        old_lines.append((key, line.rstrip('\n')))
                    except Exception:
                        old_lines.append((None, line.rstrip('\n')))
        buffer_dict = {(m['episode_idx'], m['frame_index']): json.dumps(m, ensure_ascii=False) for m in self.meta_buffer}
        lines = [l for k, l in old_lines if k not in buffer_dict]
        lines.extend(buffer_dict.values())
        ensure_dir(episode_dir)
        with open(meta_jsonl_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        self.meta_buffer.clear()

    # ---------- 关闭 ----------
    def destroy_node(self):
        try:
            if self._kb_listener is not None:
                self._kb_listener.stop()
        except Exception:
            pass

        self._stop_evt.set()
        # Ensure image writer is stopped and threads/processes joined to release resources.
        try:
            iw = getattr(self, 'image_writer', None)
            if iw is not None:
                try:
                    iw.stop()
                except Exception as e:
                    self.get_logger().warn(f"image_writer.stop() failed during destroy: {e}")
                try:
                    # drop reference and collect
                    self.image_writer = None
                    gc.collect()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if getattr(self, "worker", None) and self.worker.is_alive():
                self.worker.join(timeout=3.0)
        except Exception:
            pass
        try:
            self._flush_meta_buffer()
        except Exception:
            pass
        return super().destroy_node()

    def _inference_loop(self):
        while rclpy.ok() and not self._stop_evt.is_set():
            time.sleep(0.1)


# ====================== main：多线程执行器 ======================

def main(args=None):
    rclpy.init(args=args)
    node = Manager()
    try:
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor(num_threads=os.cpu_count() or 4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception as e:
            # ignore duplicate shutdown / RCLError on context
            print(f"rclpy.shutdown() ignored error: {e}")


if __name__ == '__main__':
    main()
