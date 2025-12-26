#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python je_software/record_valid_indices.py "/media/kleist/Lenovo Y910/log"
"""
import os
import sys
import json
import csv


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} ROOT_DIR", file=sys.stderr)
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    global_index = 0
    rows = []  # (global_index, episode_idx, frame_index)

    # 按 episode_000000, episode_000001, ... 顺序遍历
    for ep_name in sorted(os.listdir(root)):
        ep_dir = os.path.join(root, ep_name)
        if not (os.path.isdir(ep_dir) and ep_name.startswith("episode_")):
            continue

        meta_path = os.path.join(ep_dir, "meta_interp.jsonl")
        if not os.path.exists(meta_path):
            # 若没有插值文件则跳过
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                frame = json.loads(line)

                # 记录所有 interpolated == 0 的帧
                # 若缺少该字段，按 0 处理（视为原始帧）
                if frame.get("interpolated", 0) == 0:
                    episode_idx = frame.get("episode_idx")
                    frame_index = frame.get("frame_index")
                    rows.append((global_index, episode_idx, frame_index))

                # 全局 index 对所有帧递增（包括 interpolated=1）
                global_index += 1

    # 输出 CSV 到根目录
    out_path = os.path.join(root, "interpolated_0_frames.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_index", "episode_idx", "frame_index"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Total frames (including interpolated=1): {global_index}")


if __name__ == "__main__":
    main()
