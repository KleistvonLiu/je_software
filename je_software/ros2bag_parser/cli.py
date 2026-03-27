#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .parser import parse_bag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Parse a ROS2 sqlite3 bag and export common topic data.',
    )
    parser.add_argument('bag_path', help='Path to a rosbag2 directory.')
    parser.add_argument(
        '--output-dir',
        default='',
        help='Output directory. Defaults to <bag_path>/_parsed.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove the output directory first if it already exists.',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output while parsing.',
    )
    args = parser.parse_args(argv)

    bag_path = Path(args.bag_path).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else bag_path / '_parsed'
    )

    manifest = parse_bag(
        bag_path=str(bag_path),
        output_dir=str(output_dir),
        overwrite=bool(args.overwrite),
        progress_callback=(
            None
            if args.quiet
            else (lambda message: print(message, flush=True))
        ),
    )
    print(f'Parsed bag into: {manifest["output_dir"]}')
    print(f'Manifest: {manifest["manifest_path"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
