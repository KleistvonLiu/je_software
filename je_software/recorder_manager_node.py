#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import threading
import time
from typing import List, Optional, Tuple, Any

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from .base_manager import BaseManager, ensure_dir, sanitize

from cv_bridge import CvBridge

from utils.image_writer import AsyncImageWriter


def _pack_depth_u16_to_rgb8(depth_u16: np.ndarray, order: str = "HI_LO", b_fill: int = 0) -> np.ndarray:
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
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


class RecorderManager(BaseManager):
    """
    录制管理器：只负责启动按键检测 + save线程。
    其它通用逻辑（参数、订阅、对齐、写盘等）在 BaseManager 内。
    """

    def __init__(self):
        super().__init__(node_name='manager')
        self._meta_lock = threading.Lock()

        self._pending_safe_log = False

        # 工具
        self.bridge = CvBridge()
        self.image_writer = AsyncImageWriter(num_processes=0, num_threads=12)

        # 元数据缓冲
        self.meta_buffer = []
        self.meta_jsonl_path = os.path.join(self.session_dir, 'meta.jsonl')

        # 录制开关 & 键盘监听（子类决定是否启动）
        self._record_enabled = False
        self._kb_listener = None
        self._last_pause_log = 0.0

        # 线程管理
        self._stop_evt = threading.Event()
        self._threads: List[threading.Thread] = []

        # 启动按键检测
        self.start_keyboard_control()
        # 启动保存线程
        self.start_save_thread()

    # ---------- 键盘控制（按需启动） ----------
    def start_keyboard_control(self):
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

    def _on_key_press(self, key):
        """Alt_R = start/resume，Ctrl_R = stop（保证已入队图片全部写完再切集）"""
        # —— 重入保护：避免多次按键导致并发切换 ——
        if not hasattr(self, "_kb_lock"):
            self._kb_lock = threading.Lock()

        try:
            kb = getattr(self, "_pynput_kb", None)
            if kb is None:
                return

            # ===== 启动 / 恢复 =====
            if key == kb.Key.alt_r:
                with self._kb_lock:
                    if not self._record_enabled:
                        self._record_enabled = True
                        self._pending_safe_log = False
                        self.get_logger().info("Recording ENABLED by right Alt")
                return

            # ===== 停止并确保不丢帧 =====
            if key == kb.Key.ctrl_r:
                with self._kb_lock:
                    if not self._record_enabled:
                        return  # 已经是停止状态，忽略重复按键

                    # 1) 立刻阻止新帧入队（_save_once 不会再执行）
                    self._record_enabled = False
                    prev_episode_idx = self.episode_idx
                    episode_dir = os.path.join(self.save_dir, f"episode_{prev_episode_idx:06d}")
                    self.get_logger().info("Recording DISABLING by right Ctrl: draining image writer...")

                # 2) **阻塞**直到所有已入队图片写完（严格不丢帧）
                try:
                    if hasattr(self, "image_writer") and self.image_writer is not None:
                        self.image_writer.wait_until_done()  # 关键：join 而非轮询 is_idle()
                except Exception as e:
                    self.get_logger().warn(f"[stop] wait_until_done error: {e}")

                # 3) 将 meta 缓冲落盘（原子替换）
                try:
                    self._flush_meta_buffer(episode_dir)
                    self.get_logger().info(f"[stop] meta flushed to {os.path.join(episode_dir, 'meta.jsonl')}")
                except Exception as e:
                    self.get_logger().warn(f"Flush meta_buffer on stop failed: {e}")

                # 4) 切到下一集并复位帧号（已确保前一集所有图片存在）
                with self._kb_lock:
                    self.episode_idx += 1
                    self.frame_idx = 0
                    self._pending_safe_log = False
                    self.get_logger().info("Recording DISABLED by right Ctrl (all frames saved)")
                return

        except Exception as e:
            self.get_logger().warn(f"Keyboard handler error: {e}")

    # ---------- 线程控制（按需启动） ----------
    def start_save_thread(self):
        th = threading.Thread(target=self._save_loop, name='save-loop', daemon=False)
        th.start()
        self._threads.append(th)
        self.get_logger().info("[threads] save-loop started")

    # 保存循环与写盘逻辑
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
                    # _ = self.aligner.step()
                    if (now - self._last_pause_log) > 2.0:
                        self.get_logger().debug("Paused: press Alt_R to start/resume, Ctrl_R to stop.")
                        self._last_pause_log = now
                    if hasattr(self, '_pending_safe_log') and self._pending_safe_log:
                        if self.image_writer.is_idle():
                            self.get_logger().info("\x1b[32m[SAFE] 所有数据已安全保存，可以安全退出程序！\x1b[0m")
                            self._pending_safe_log = False
                    continue

                if self.do_calculate_hz:
                    self._attempt_win += 1
                out = self.aligner.step()
                if out is None:
                    continue
                if self.do_calculate_hz:
                    self._success_win += 1
                t_ref, picks = out
                self._save_once(t_ref, picks)
                if self.do_calculate_hz:
                    self._on_saved_stats()
            except Exception as e:
                self.get_logger().error(f"save loop error: {e}")

    # === 统计 ===
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
        episode_dir = os.path.join(self.session_dir, f"episode_{episode_idx:06d}")
        ensure_dir(episode_dir)

        C = len(self._idx_color)
        D = len(self._idx_depth)
        J = len(self._idx_joint)
        T = len(self._idx_tact)

        color_picks = picks[0:C]
        depth_picks = picks[C:C + D] if D > 0 else []
        joint_picks = picks[C + D:C + D + J] if J > 0 else []
        tact_picks = picks[C + D + J:C + D + J + T] if T > 0 else []

        # 1) color
        image_fields = {}
        for cam_i, item in enumerate(color_picks):
            if item is None: continue
            t_ns, msg = item
            cam_name = sanitize(self.color_topics[cam_i])
            img_dir = os.path.join(episode_dir, "images", cam_name)
            ensure_dir(img_dir)
            fn = f"frame_{idx:06d}.png"
            fp = os.path.join(img_dir, fn)
            if getattr(msg, "encoding", "") != 'rgb8':
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            else:
                img = self.bridge.imgmsg_to_cv2(msg)
            self.image_writer.save_image(img, fp)
            image_fields[cam_name] = os.path.relpath(fp, episode_dir)

        # 2) depth
        if self.save_depth and depth_picks:
            for dep_i, item in enumerate(depth_picks):
                if item is None: continue
                t_ns, msg = item
                cam_name = sanitize(self.depth_topics[dep_i])
                img_dir = os.path.join(episode_dir, "images", cam_name)
                ensure_dir(img_dir)
                fn = f"frame_{idx:06d}.png"
                fp = os.path.join(img_dir, fn)
                try:
                    depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                except Exception:
                    depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
                depth_rgb8 = _pack_depth_u16_to_rgb8(depth_u16, order="HI_LO", b_fill=0)
                self.image_writer.save_image(depth_rgb8, fp)
                image_fields[cam_name] = os.path.relpath(fp, episode_dir)

        # 3) meta（可自行扩展 joint/tactile）
        joints_meta, tactiles_meta = [], []
        for k, item in enumerate(joint_picks):
            if item is None: continue
            t_ns, js_msg = item
            joints_meta.append(dict(
                topic=self.joint_topics[k],
                stamp_ns=int(t_ns),
                name=list(getattr(js_msg, "name", [])),
                position=[float(x) for x in getattr(js_msg, "position", [])],
                velocity=[float(x) for x in getattr(js_msg, "velocity", [])],
                effort=[float(x) for x in getattr(js_msg, "effort", [])],
            ))
        for k, item in enumerate(tact_picks):
            if item is None: continue
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
            tactiles=tactiles_meta,
        )
        # 仅缓存，不在录制过程中落盘；由 Ctrl_R 停止时统一落盘，保证一致性
        with self._meta_lock:
            self.meta_buffer.append(meta)

    def _flush_meta_buffer(self, episode_dir=None):
        # ----- 锁内：取走缓冲，快速返回 -----
        with self._meta_lock:
            if not self.meta_buffer:
                return
            buf = self.meta_buffer
            self.meta_buffer = []  # 直接换空
        # ----- 锁外：I/O 与合并 -----
        if episode_dir is None:
            episode_idx = getattr(self, 'episode_idx', 0)
            episode_dir = os.path.join(self.session_dir, f"episode_{episode_idx:06d}")
        ensure_dir(episode_dir)
        meta_jsonl_path = os.path.join(episode_dir, 'meta.jsonl')

        old = {}
        if os.path.exists(meta_jsonl_path):
            with open(meta_jsonl_path, 'r') as f:
                for line in f:
                    s = line.rstrip('\n')
                    try:
                        obj = json.loads(s)
                        k = (obj.get('episode_idx'), obj.get('frame_index'))
                        old[k] = s
                    except Exception:
                        old[(None, len(old))] = s  # 保底保留

        for m in buf:
            k = (m['episode_idx'], m['frame_index'])
            old[k] = json.dumps(m, ensure_ascii=False)

        def _key(item):
            k, _ = item
            return (k[0], k[1]) if isinstance(k, tuple) and isinstance(k[0], int) and isinstance(k[1], int) else (
                1 << 60, 1 << 60)

        lines = [v for _, v in sorted(old.items(), key=_key)]
        tmp = meta_jsonl_path + ".tmp"
        with open(tmp, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        os.replace(tmp, meta_jsonl_path)  # 原子替换

    def destroy_node(self):
        try:
            if self._kb_listener is not None:
                self._kb_listener.stop()
        except Exception:
            pass

        self._stop_evt.set()
        for th in self._threads:
            try:
                if th.is_alive():
                    th.join(timeout=3.0)
                    if th.is_alive():
                        th.join()  # 兜底
            except Exception:
                pass

        # 先停写图线程，避免随后 flush 期间仍有新入队
        try:
            if hasattr(self, "image_writer"):
                self.image_writer.stop()
        except Exception:
            pass

        try:
            self._flush_meta_buffer()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderManager()
    try:
        executor = MultiThreadedExecutor(num_threads=os.cpu_count() or 4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
