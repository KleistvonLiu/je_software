#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python ./je_software/interpolate_meta.py /media/kleist/Lenovo Y910/log/
"""
import os
import sys
import json

DT_THRESH = 0.07   # 判定需要插值的时间差阈值
DT_STEP = 0.033    # 插值步长（秒）


def interpolate_positions(joints0, joints1, t0, t1, t):
    """
    对两帧的 joints 按 topic 对齐，并对 position 做线性插值。
    只返回 topic/name/position，后续再拼接 velocity/effort/stamp_ns。
    """
    if t1 == t0:
        return [
            {
                "topic": j0["topic"],
                "name": j0.get("name", []),
                "position": list(j0["position"]),
            }
            for j0 in joints0
        ]

    alpha = float(t - t0) / float(t1 - t0)

    by_topic0 = {j["topic"]: j for j in joints0}
    by_topic1 = {j["topic"]: j for j in joints1}

    out = []
    for topic, j0 in by_topic0.items():
        j1 = by_topic1.get(topic)
        if j1 is None:
            # 如下一帧没有该 topic，可选复制 j0；目前保持跳过，避免造假数据
            continue

        pos0 = j0["position"]
        pos1 = j1["position"]
        names = j0.get("name", j1.get("name", []))  # 假设 name 顺序一致

        pos_interp = [
            int(p0 + (p1 - p0) * alpha)
            for p0, p1 in zip(pos0, pos1)
        ]

        out.append({
            "topic": topic,
            "name": names,
            "position": pos_interp,
        })

    return out


def process_episode(ep_dir: str):
    meta_path = os.path.join(ep_dir, "meta.jsonl")
    if not os.path.exists(meta_path):
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        frames = [json.loads(line) for line in f if line.strip()]

    if len(frames) <= 1:
        return

    ep_name = os.path.basename(ep_dir)

    new_frames = []
    new_frame_index = 0
    missing_indices = []  # 插值帧 index

    for i in range(len(frames) - 1):
        cur = frames[i]
        nxt = frames[i + 1]

        t0 = float(cur["timestamp"])
        t1 = float(nxt["timestamp"])

        cur_orig_idx = cur.get("frame_index", i)
        nxt_orig_idx = nxt.get("frame_index", i + 1)

        # 写入当前原始帧（重写 frame_index，标记 interpolated=0）
        cur_out = dict(cur)
        cur_out["frame_index"] = new_frame_index
        cur_out["interpolated"] = 0
        new_frames.append(cur_out)
        new_frame_index += 1

        gap = t1 - t0
        if gap > DT_THRESH:
            start_missing_idx = new_frame_index

            t = t0 + DT_STEP
            while t < t1 - 1e-9:
                # 上一帧：新序列中已写入的最后一帧
                prev_frame = new_frames[-1]

                # 1) 插值 position（始终在原始 cur/nxt 之间）
                pos_only = interpolate_positions(
                    cur["joints"],
                    nxt["joints"],
                    t0,
                    t1,
                    t,
                )

                # 2) 从上一帧 joints 继承 velocity / effort / stamp_ns
                prev_joints = prev_frame.get("joints", [])
                prev_by_topic = {j.get("topic"): j for j in prev_joints}

                joints_interp = []
                for j_pos in pos_only:
                    topic = j_pos["topic"]
                    prev_j = prev_by_topic.get(topic)

                    if prev_j is not None:
                        joints_interp.append({
                            "topic": topic,
                            "stamp_ns": prev_j.get("stamp_ns"),
                            "name": j_pos.get("name", prev_j.get("name", [])),
                            "position": j_pos["position"],
                            "velocity": prev_j.get("velocity", []),
                            "effort": prev_j.get("effort", []),
                        })
                    else:
                        # 没有上一帧的同 topic，只写 position/name（无 velocity/effort/stamp_ns）
                        joints_interp.append({
                            "topic": topic,
                            "name": j_pos.get("name", []),
                            "position": j_pos["position"],
                        })

                # 3) 构造插值帧，其他字段（图像路径、tactiles 等）继承上一帧
                interp_frame = dict(prev_frame)
                interp_frame["episode_idx"] = cur["episode_idx"]
                interp_frame["frame_index"] = new_frame_index
                interp_frame["timestamp"] = float(t)
                interp_frame["joints"] = joints_interp
                interp_frame["interpolated"] = 1

                new_frames.append(interp_frame)
                missing_indices.append(new_frame_index)

                new_frame_index += 1
                t += DT_STEP

            end_missing_idx = new_frame_index - 1
            num_inserted = max(0, end_missing_idx - start_missing_idx + 1)

            print(
                f"[GAP] {ep_name}: "
                f"orig_frame {cur_orig_idx} -> {nxt_orig_idx}, "
                f"dt={gap:.6f}s > {DT_THRESH}, "
                f"insert {num_inserted} frames at indices "
                f"[{start_missing_idx}..{end_missing_idx}]"
            )

    # 最后一帧（原始）
    last = frames[-1]
    last_out = dict(last)
    last_out["frame_index"] = new_frame_index
    last_out["interpolated"] = 0
    new_frames.append(last_out)

    out_path = os.path.join(ep_dir, "meta_interp.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in new_frames:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] {ep_name}: {len(frames)} -> {len(new_frames)}")

    if missing_indices:
        print(f"[MISSING] {ep_name} missing frame indices (interpolated): {missing_indices}")
    else:
        print(f"[MISSING] {ep_name}: no missing frames detected (no gap > {DT_THRESH})")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} ROOT_DIR", file=sys.stderr)
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    for name in sorted(os.listdir(root)):
        ep_dir = os.path.join(root, name)
        if os.path.isdir(ep_dir) and name.startswith("episode_"):
            process_episode(ep_dir)


if __name__ == "__main__":
    main()
