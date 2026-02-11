#!/usr/bin/env python3
import ast
import sys
from argparse import ArgumentParser, Namespace
from typing import Iterable


def _parse_ids(text: str) -> list[int]:
    value = text.strip()
    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        out: list[int] = []
        for item in parsed:
            out.append(int(item))
        return out

    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Read and print Dynamixel Operating_Mode for selected motors."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1000000, help="Bus baudrate")
    parser.add_argument(
        "--ids",
        default="[1,2,3,4,5,6,7,8]",
        help="Motor IDs, e.g. '[1,2,3]' or '1,2,3'",
    )
    parser.add_argument(
        "--model",
        default="xl330-m077",
        help="Motor model name used by this codebase, e.g. xl330-m077",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=1,
        help="Additional retry attempts for sync_read",
    )
    return parser


def _mode_name(raw_value: int) -> str:
    mapping = {
        0: "CURRENT",
        1: "VELOCITY",
        3: "POSITION",
        4: "EXTENDED_POSITION",
        5: "CURRENT_POSITION",
        16: "PWM",
    }
    return mapping.get(raw_value, "UNKNOWN")


def _print_results(values: dict[str, int], name_to_id: dict[str, int]) -> None:
    print("Operating_Mode results:")
    for motor_name in sorted(values, key=lambda n: name_to_id[n]):
        raw = int(values[motor_name])
        motor_id = name_to_id[motor_name]
        print(f"  id={motor_id:<3d} mode={raw:<3d} ({_mode_name(raw)})")


def _create_motors(
    ids: Iterable[int], model: str, motor_cls, motor_norm_mode
) -> tuple[dict[str, object], dict[str, int]]:
    motors: dict[str, object] = {}
    name_to_id: dict[str, int] = {}
    for motor_id in ids:
        name = f"id_{motor_id}"
        motors[name] = motor_cls(motor_id, model, motor_norm_mode.RANGE_M100_100)
        name_to_id[name] = motor_id
    return motors, name_to_id


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args: Namespace = parser.parse_args(argv)

    try:
        from je_software.motors import Motor, MotorNormMode
        from je_software.motors.dynamixel import DynamixelMotorsBus
    except ModuleNotFoundError as exc:
        print(
            f"Import error: {exc}. Please install required dependencies (e.g. deepdiff, dynamixel_sdk).",
            file=sys.stderr,
        )
        return 1

    try:
        ids = _parse_ids(args.ids)
    except ValueError as exc:
        print(f"Invalid --ids value '{args.ids}': {exc}", file=sys.stderr)
        return 2

    if not ids:
        print("--ids is empty; provide at least one motor id.", file=sys.stderr)
        return 2

    unique_ids = list(dict.fromkeys(ids))
    if len(unique_ids) != len(ids):
        print("Duplicated ids detected in --ids; using unique IDs in input order.")
        ids = unique_ids

    motors, name_to_id = _create_motors(ids, args.model, Motor, MotorNormMode)
    bus = DynamixelMotorsBus(port=args.port, motors=motors)

    try:
        bus.connect(handshake=False)
        bus.set_baudrate(args.baudrate)
        bus._handshake()
        values = bus.sync_read("Operating_Mode", normalize=False, num_retry=max(0, args.retry))
        _print_results(values, name_to_id)
    except Exception as exc:
        print(f"Failed to read Operating_Mode: {exc}", file=sys.stderr)
        return 1
    finally:
        if bus.is_connected:
            try:
                bus.disconnect()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
