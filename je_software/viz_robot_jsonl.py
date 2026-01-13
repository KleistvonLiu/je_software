#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Docstring for je_software.viz_robot_jsonl

Examples:
python3 ./je_software/viz_robot_jsonl.py your_log.jsonl
python3 ./je_software/viz_robot_jsonl.py your_log.jsonl --deg
python3 ./je_software/viz_robot_jsonl.py your_log.jsonl --stride 10
python3 ./je_software/viz_robot_jsonl.py your_log.jsonl --save-prefix out
"""
import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def _safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


def load_jsonl(
    path: str,
    robot_key: str = "Robot0",
    time_key_sec: str = "__ros_stamp_sec",
    time_key_ns: str = "__ros_stamp_ns",
    stride: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Load robot state jsonl into numpy arrays.
    Returns dict with:
      t, joint, target_joint, cart, target_cart, move_state, power_state,
      joint_velocity, joint_torque, joint_sensor_torque
    """
    ts: List[float] = []
    joint: List[List[float]] = []
    target_joint: List[List[float]] = []
    cart: List[List[float]] = []
    target_cart: List[List[float]] = []
    move_state: List[int] = []
    power_state: List[int] = []

    joint_velocity: List[List[float]] = []
    joint_torque: List[List[float]] = []
    joint_sensor_torque: List[List[float]] = []  # NEW

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if stride > 1 and (i % stride) != 0:
                continue
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # time
            t = None
            t_sec = _safe_get(obj, time_key_sec, None)
            if t_sec is not None:
                try:
                    t = float(t_sec)
                except Exception:
                    t = None
            if t is None:
                t_ns = _safe_get(obj, time_key_ns, None)
                if t_ns is not None:
                    try:
                        t = float(t_ns) * 1e-9
                    except Exception:
                        t = None
            if t is None:
                continue

            rob = _safe_get(obj, robot_key, None)
            if not isinstance(rob, dict):
                continue

            j = _safe_get(rob, "Joint", None)
            tj = _safe_get(rob, "TargetJoint", None)
            c = _safe_get(rob, "Cartesian", None)
            tc = _safe_get(rob, "TargetCartesian", None)

            jv = _safe_get(rob, "JointVelocity", None)
            jtq = _safe_get(rob, "JointTorque", None)
            jstq = _safe_get(rob, "JointSensorTorque", None)  # NEW

            # Basic validation
            if j is None or c is None:
                continue

            try:
                j = [float(x) for x in j]
                c = [float(x) for x in c]
            except Exception:
                continue

            dof = len(j)

            if tj is not None:
                try:
                    tj = [float(x) for x in tj]
                except Exception:
                    tj = None

            if tc is not None:
                try:
                    tc = [float(x) for x in tc]
                except Exception:
                    tc = None

            # JointVelocity / JointTorque / JointSensorTorque: allow missing; if present validate length
            if jv is not None:
                try:
                    jv = [float(x) for x in jv]
                    if len(jv) != dof:
                        jv = None
                except Exception:
                    jv = None

            if jtq is not None:
                try:
                    jtq = [float(x) for x in jtq]
                    if len(jtq) != dof:
                        jtq = None
                except Exception:
                    jtq = None

            if jstq is not None:
                try:
                    jstq = [float(x) for x in jstq]
                    if len(jstq) != dof:
                        jstq = None
                except Exception:
                    jstq = None

            ms = _safe_get(rob, "MoveState", None)
            ps = _safe_get(rob, "PowerState", None)
            try:
                ms = int(ms) if ms is not None else -1
            except Exception:
                ms = -1
            try:
                ps = int(ps) if ps is not None else -1
            except Exception:
                ps = -1

            ts.append(t)
            joint.append(j)
            cart.append(c)
            target_joint.append(tj if tj is not None else [np.nan] * dof)
            target_cart.append(tc if tc is not None else [np.nan] * len(c))
            move_state.append(ms)
            power_state.append(ps)

            joint_velocity.append(jv if jv is not None else [np.nan] * dof)
            joint_torque.append(jtq if jtq is not None else [np.nan] * dof)
            joint_sensor_torque.append(jstq if jstq is not None else [np.nan] * dof)  # NEW

    if not ts:
        raise RuntimeError(f"No valid records loaded from: {path}")

    # sort by time
    ts_np = np.array(ts, dtype=np.float64)
    order = np.argsort(ts_np)
    ts_np = ts_np[order]

    joint_np = np.array(joint, dtype=np.float64)[order]
    cart_np = np.array(cart, dtype=np.float64)[order]
    target_joint_np = np.array(target_joint, dtype=np.float64)[order]
    target_cart_np = np.array(target_cart, dtype=np.float64)[order]
    move_np = np.array(move_state, dtype=np.int64)[order]
    power_np = np.array(power_state, dtype=np.int64)[order]

    joint_vel_np = np.array(joint_velocity, dtype=np.float64)[order]
    joint_torque_np = np.array(joint_torque, dtype=np.float64)[order]
    joint_sensor_torque_np = np.array(joint_sensor_torque, dtype=np.float64)[order]  # NEW

    # relative time
    t0 = ts_np[0]
    t_rel = ts_np - t0

    return {
        "t": t_rel,
        "joint": joint_np,
        "target_joint": target_joint_np,
        "cart": cart_np,
        "target_cart": target_cart_np,
        "move_state": move_np,
        "power_state": power_np,
        "joint_velocity": joint_vel_np,
        "joint_torque": joint_torque_np,
        "joint_sensor_torque": joint_sensor_torque_np,  # NEW
    }


def plot_joints(t: np.ndarray, joint: np.ndarray, target_joint: np.ndarray, save: Optional[str] = None):
    n = joint.shape[1]
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(12, 2.0 * n))
    if n == 1:
        axs = [axs]

    for i in range(n):
        axs[i].plot(t, joint[:, i], label=f"Joint[{i}]")
        if not np.all(np.isnan(target_joint[:, i])):
            axs[i].plot(t, target_joint[:, i], label=f"TargetJoint[{i}]")
        axs[i].set_ylabel("rad")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="best")

    axs[-1].set_xlabel("t (s)")
    fig.suptitle("Joint vs TargetJoint")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=200)
    return fig


def plot_joint_velocity(t: np.ndarray, joint_velocity: np.ndarray, save: Optional[str] = None):
    """
    Plot JointVelocity (rad/s). If all-NaN, return None.
    """
    if joint_velocity.size == 0 or np.all(np.isnan(joint_velocity)):
        return None

    n = joint_velocity.shape[1]
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(12, 2.0 * n))
    if n == 1:
        axs = [axs]

    for i in range(n):
        if np.all(np.isnan(joint_velocity[:, i])):
            continue
        axs[i].plot(t, joint_velocity[:, i], label=f"JointVelocity[{i}]")
        axs[i].set_ylabel("rad/s")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="best")

    axs[-1].set_xlabel("t (s)")
    fig.suptitle("JointVelocity")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=200)
    return fig


def plot_joint_torque_compare(
    t: np.ndarray,
    joint_torque: np.ndarray,
    joint_sensor_torque: np.ndarray,
    save: Optional[str] = None,
):
    """
    Plot JointSensorTorque vs JointTorque (Nm or N; depends on hardware).
    For each joint dimension i: plot both in the same subplot.
    If both are all-NaN, return None.
    """
    factor = 1e-0
    if (
        (joint_torque.size == 0 or np.all(np.isnan(joint_torque)))
        and (joint_sensor_torque.size == 0 or np.all(np.isnan(joint_sensor_torque)))
    ):
        return None

    # Determine DOF robustly
    n = 0
    if joint_torque.ndim == 2 and joint_torque.shape[1] > 0:
        n = joint_torque.shape[1]
    if joint_sensor_torque.ndim == 2 and joint_sensor_torque.shape[1] > 0:
        n = max(n, joint_sensor_torque.shape[1])
    if n <= 0:
        return None

    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(12, 2.0 * n))
    if n == 1:
        axs = [axs]

    for i in range(n):
        # JointTorque
        if joint_torque.ndim == 2 and i < joint_torque.shape[1] and not np.all(np.isnan(joint_torque[:, i])):
            axs[i].plot(t, joint_torque[:, i], label=f"JointTorque[{i}]")

        # JointSensorTorque
        if joint_sensor_torque.ndim == 2 and i < joint_sensor_torque.shape[1] and not np.all(np.isnan(joint_sensor_torque[:, i])):
            axs[i].plot(t, joint_sensor_torque[:, i] * factor, label=f"JointSensorTorque[{i}]")

        axs[i].set_ylabel("Nm/N")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="best")

    axs[-1].set_xlabel("t (s)")
    fig.suptitle("JointSensorTorque vs JointTorque")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=200)
    return fig


def plot_cartesian(
    t: np.ndarray,
    cart: np.ndarray,
    target_cart: np.ndarray,
    angles_in_deg: bool = False,
    save: Optional[str] = None,
):
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    unit = ["m", "m", "m", "rad", "rad", "rad"]

    cart2 = cart.copy()
    tgt2 = target_cart.copy()
    if angles_in_deg:
        cart2[:, 3:6] = np.rad2deg(cart2[:, 3:6])
        tgt2[:, 3:6] = np.rad2deg(tgt2[:, 3:6])
        unit[3:6] = ["deg", "deg", "deg"]

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
    for i in range(6):
        axs[i].plot(t, cart2[:, i], label=f"Cartesian[{labels[i]}]")
        if not np.all(np.isnan(tgt2[:, i])):
            axs[i].plot(t, tgt2[:, i], label=f"TargetCartesian[{labels[i]}]")
        axs[i].set_ylabel(unit[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="best")

    axs[-1].set_xlabel("t (s)")
    fig.suptitle("Cartesian vs TargetCartesian")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=200)
    return fig


def plot_xyz_trajectory(cart: np.ndarray, save: Optional[str] = None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(cart[:, 0], cart[:, 1], cart[:, 2], label="XYZ trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.suptitle("End-effector XYZ trajectory")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=200)
    return fig


def plot_states(t: np.ndarray, move_state: np.ndarray, power_state: np.ndarray, save: Optional[str] = None):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    axs[0].step(t, move_state, where="post", label="MoveState")
    axs[0].set_ylabel("value")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].step(t, power_state, where="post", label="PowerState")
    axs[1].set_ylabel("value")
    axs[1].set_xlabel("t (s)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    fig.suptitle("MoveState / PowerState")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=200)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize robot jsonl logs.")
    parser.add_argument("jsonl", help="Path to .jsonl log file")
    parser.add_argument("--robot-key", default="Robot1", help="Robot key in JSON, e.g., Robot0")
    parser.add_argument("--stride", type=int, default=1, help="Downsample by keeping 1 out of N lines")
    parser.add_argument("--deg", action="store_true", help="Plot roll/pitch/yaw in degrees")
    parser.add_argument("--no-xyz", action="store_true", help="Disable 3D XYZ trajectory plot")
    parser.add_argument("--save-prefix", default="", help="If set, save figures as <prefix>_*.png")
    args = parser.parse_args()

    data = load_jsonl(
        path=args.jsonl,
        robot_key=args.robot_key,
        stride=max(1, args.stride),
    )

    save_prefix = args.save_prefix.strip()
    save_joint = f"{save_prefix}_joints.png" if save_prefix else None
    save_cart = f"{save_prefix}_cart.png" if save_prefix else None
    save_xyz = f"{save_prefix}_xyz.png" if save_prefix else None
    save_state = f"{save_prefix}_state.png" if save_prefix else None
    save_vel = f"{save_prefix}_joint_vel.png" if save_prefix else None
    save_torque_cmp = f"{save_prefix}_joint_torque_cmp.png" if save_prefix else None  # NEW

    plot_joints(data["t"], data["joint"], data["target_joint"], save=save_joint)
    plot_cartesian(data["t"], data["cart"], data["target_cart"], angles_in_deg=args.deg, save=save_cart)
    plot_states(data["t"], data["move_state"], data["power_state"], save=save_state)

    plot_joint_velocity(data["t"], data["joint_velocity"], save=save_vel)

    # NEW: compare JointSensorTorque vs JointTorque in the same figure
    plot_joint_torque_compare(
        data["t"],
        data["joint_torque"],
        data["joint_sensor_torque"],
        save=save_torque_cmp,
    )

    if not args.no_xyz:
        plot_xyz_trajectory(data["cart"], save=save_xyz)

    plt.show()


if __name__ == "__main__":
    main()
