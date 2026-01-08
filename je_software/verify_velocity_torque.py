#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python3 ./je_software/verify_velocity_torque.py /home/kleist/je_robot_logs/robot_state_20251229_173755.jsonl --robot-key Robot0 --save-prefix out --plot
"""
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def safe_get(d: Dict[str, Any], k: str, default=None):
    return d[k] if k in d else default


def load_jsonl(
    path: str,
    robot_key: str = "Robot0",
    time_key_ns: str = "__ros_stamp_ns",
    time_key_sec: str = "__ros_stamp_sec",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t: shape (N,) seconds (relative)
      q: shape (N, dof)
      v_log: shape (N, dof) with NaN if missing
      tau_log: shape (N, dof) with NaN if missing
    """
    ts: List[float] = []
    qs: List[List[float]] = []
    vs: List[List[float]] = []
    taus: List[List[float]] = []

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

            # time prefer ns
            t = None
            t_ns = safe_get(obj, time_key_ns, None)
            if t_ns is not None:
                try:
                    t = float(t_ns) * 1e-9
                except Exception:
                    t = None
            if t is None:
                t_sec = safe_get(obj, time_key_sec, None)
                if t_sec is not None:
                    try:
                        t = float(t_sec)
                    except Exception:
                        t = None
            if t is None:
                continue

            rob = safe_get(obj, robot_key, None)
            if not isinstance(rob, dict):
                continue

            j = safe_get(rob, "Joint", None)
            if j is None:
                continue
            try:
                j = [float(x) for x in j]
            except Exception:
                continue

            dof = len(j)

            v = safe_get(rob, "JointVelocity", None)
            if v is not None:
                try:
                    v = [float(x) for x in v]
                    if len(v) != dof:
                        v = None
                except Exception:
                    v = None

            tau = safe_get(rob, "JointTorque", None)
            if tau is not None:
                try:
                    tau = [float(x) for x in tau]
                    if len(tau) != dof:
                        tau = None
                except Exception:
                    tau = None

            ts.append(t)
            qs.append(j)
            vs.append(v if v is not None else [np.nan] * dof)
            taus.append(tau if tau is not None else [np.nan] * dof)

    if not ts:
        raise RuntimeError(f"No valid records loaded from: {path}")

    ts_np = np.array(ts, dtype=np.float64)
    order = np.argsort(ts_np)
    ts_np = ts_np[order]
    q = np.array(qs, dtype=np.float64)[order]
    v_log = np.array(vs, dtype=np.float64)[order]
    tau_log = np.array(taus, dtype=np.float64)[order]

    # relative time
    t0 = ts_np[0]
    t = ts_np - t0

    return t, q, v_log, tau_log


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    Simple moving average along axis=0. win must be odd >=1.
    """
    if win <= 1:
        return x
    if win % 2 == 0:
        raise ValueError("smooth window must be odd")
    pad = win // 2
    xpad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / float(win)
    out = np.empty_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(xpad[:, j], kernel, mode="valid")
    return out


def finite_diff_velocity(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Use np.gradient with uneven dt support.
    """
    v = np.zeros_like(q, dtype=np.float64)
    for j in range(q.shape[1]):
        v[:, j] = np.gradient(q[:, j], t)
    return v


def finite_diff_accel(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    a = np.zeros_like(v, dtype=np.float64)
    for j in range(v.shape[1]):
        a[:, j] = np.gradient(v[:, j], t)
    return a


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    x2, y2 = x[m], y[m]
    if np.std(x2) < 1e-12 or np.std(y2) < 1e-12:
        return np.nan
    return float(np.corrcoef(x2, y2)[0, 1])


def fit_scale_bias(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit y ≈ s*x + b on finite samples (least squares closed-form).
    Returns (s, b).
    """
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan
    x2, y2 = x[m], y[m]
    vx = np.var(x2)
    if vx < 1e-12:
        return np.nan, np.nan
    s = float(np.cov(x2, y2, bias=True)[0, 1] / vx)
    b = float(np.mean(y2) - s * np.mean(x2))
    return s, b


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    d = x[m] - y[m]
    return float(np.sqrt(np.mean(d * d)))


def best_lag_by_corr(v_fd: np.ndarray, v_log: np.ndarray, max_lag: int) -> int:
    """
    Find lag in samples that maximizes mean correlation across joints:
      compare v_log shifted vs v_fd.
    lag > 0 means v_log is delayed (need to shift v_log left to align).
    """
    dof = v_fd.shape[1]
    best_lag = 0
    best_score = -1e9

    for lag in range(-max_lag, max_lag + 1):
        cors = []
        for j in range(dof):
            x = v_fd[:, j]
            y = v_log[:, j]
            if not np.isfinite(y).any():
                continue

            if lag > 0:
                x2 = x[:-lag]
                y2 = y[lag:]
            elif lag < 0:
                x2 = x[-lag:]
                y2 = y[:lag]
            else:
                x2, y2 = x, y

            c = corrcoef_safe(x2, y2)
            if np.isfinite(c):
                cors.append(c)
        if cors:
            score = float(np.nanmean(cors))
            if score > best_score:
                best_score = score
                best_lag = lag

    return best_lag


def shift_by_lag(arr: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align arrays by cutting edges. Returns (a, b) with same length:
      a: original (reference)
      b: shifted version aligned
    """
    if lag > 0:
        return arr[:-lag], arr[lag:]
    if lag < 0:
        return arr[-lag:], arr[:lag]
    return arr, arr


def check_velocity(t: np.ndarray, q: np.ndarray, v_log: np.ndarray, max_lag: int) -> Dict[str, Any]:
    v_fd = finite_diff_velocity(t, q)

    # determine best lag globally
    lag = best_lag_by_corr(v_fd, v_log, max_lag=max_lag)

    # align for metrics
    v_fd_a, v_log_a = shift_by_lag(v_fd, lag)
    t_a, _ = shift_by_lag(t, lag)

    dof = q.shape[1]
    per_joint = []
    scales = []

    for j in range(dof):
        s, b = fit_scale_bias(v_fd_a[:, j], v_log_a[:, j])
        c = corrcoef_safe(v_fd_a[:, j], v_log_a[:, j])
        e = rmse(v_fd_a[:, j], v_log_a[:, j])
        per_joint.append({"j": j, "scale": s, "bias": b, "corr": c, "rmse": e})
        if np.isfinite(s):
            scales.append(s)

    # integration check: integrate v_log to reconstruct q
    # use aligned series
    q_a, _ = shift_by_lag(q, lag)
    q0 = q_a[0].copy()
    dt = np.diff(t_a)
    q_hat = np.zeros_like(q_a)
    q_hat[0] = q0
    for k in range(1, len(t_a)):
        q_hat[k] = q_hat[k - 1] + v_log_a[k] * (t_a[k] - t_a[k - 1])

    q_err = q_a - q_hat
    q_err_rms = np.sqrt(np.nanmean(q_err * q_err, axis=0))

    # unit / scale heuristics
    scale_med = float(np.nanmedian(scales)) if scales else np.nan
    diagnosis = []
    if np.isfinite(scale_med):
        if 50.0 < abs(scale_med) < 70.0:
            diagnosis.append(f"Velocity scale median ≈ {scale_med:.3f}: suspect deg/s vs rad/s mismatch (≈57.3x).")
        if 0.014 < abs(scale_med) < 0.020:
            diagnosis.append(f"Velocity scale median ≈ {scale_med:.5f}: suspect rad/s vs deg/s mismatch (≈0.01745x).")
        if abs(scale_med - 1.0) < 0.2:
            diagnosis.append(f"Velocity scale median ≈ {scale_med:.3f}: scale looks broadly consistent.")
        else:
            diagnosis.append(f"Velocity scale median ≈ {scale_med:.3f}: scale inconsistency possible (gear ratio / motor-side vs joint-side / mapping).")

    return {
        "v_fd": v_fd,
        "lag": lag,
        "per_joint": per_joint,
        "q_err_rms": q_err_rms,
        "diagnosis": diagnosis,
    }


def linfit_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit y ≈ kx + b and return (k, b, R^2) on finite samples.
    """
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan, np.nan, np.nan
    x2, y2 = x[m], y[m]
    vx = np.var(x2)
    if vx < 1e-12:
        return np.nan, np.nan, np.nan
    k = float(np.cov(x2, y2, bias=True)[0, 1] / vx)
    b = float(np.mean(y2) - k * np.mean(x2))
    yhat = k * x2 + b
    ss_res = float(np.sum((y2 - yhat) ** 2))
    ss_tot = float(np.sum((y2 - np.mean(y2)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return k, b, r2


def check_torque(
    t: np.ndarray,
    q: np.ndarray,
    v_log: np.ndarray,
    tau_log: np.ndarray,
    lag_vel: int,
    v_static_eps: float,
) -> Dict[str, Any]:
    # compute v_fd/a_fd from q
    v_fd = finite_diff_velocity(t, q)
    a_fd = finite_diff_accel(t, v_fd)

    # align tau with velocity lag (common situation: v has 1-cycle delay)
    v_fd_a, v_log_a = shift_by_lag(v_fd, lag_vel)
    a_fd_a, _ = shift_by_lag(a_fd, lag_vel)
    tau_a, _ = shift_by_lag(tau_log, lag_vel)
    t_a, _ = shift_by_lag(t, lag_vel)

    dof = q.shape[1]

    # static mask based on |v_fd|
    static_mask = np.nanmax(np.abs(v_fd_a), axis=1) < v_static_eps

    # power checks
    power_joint = tau_a * v_log_a
    power_total = np.nansum(power_joint, axis=1)

    static_power_abs_med = float(np.nanmedian(np.abs(power_total[static_mask]))) if np.any(static_mask) else np.nan
    moving_power_abs_med = float(np.nanmedian(np.abs(power_total[~static_mask]))) if np.any(~static_mask) else np.nan

    # torque stability on static segments
    tau_static_mean = np.full(dof, np.nan)
    tau_static_std = np.full(dof, np.nan)
    if np.any(static_mask):
        tau_static_mean = np.nanmean(tau_a[static_mask], axis=0)
        tau_static_std = np.nanstd(tau_a[static_mask], axis=0)

    # inertia-like consistency: tau vs ddq on moving segments
    # use a threshold to avoid fitting on near-zero accel noise
    a_abs = np.nanmax(np.abs(a_fd_a), axis=1)
    a_thr = float(np.nanpercentile(a_abs[np.isfinite(a_abs)], 60)) if np.isfinite(a_abs).any() else np.nan
    if not np.isfinite(a_thr) or a_thr < 1e-6:
        a_thr = 1e-3
    moving_mask = (~static_mask) & (a_abs > a_thr)

    per_joint_fit = []
    for j in range(dof):
        k, b, r2 = linfit_r2(a_fd_a[moving_mask, j], tau_a[moving_mask, j])
        per_joint_fit.append({"j": j, "tau_vs_ddq_k": k, "tau_vs_ddq_b": b, "r2": r2})

    diagnosis = []
    if np.isfinite(static_power_abs_med) and np.isfinite(moving_power_abs_med):
        diagnosis.append(
            f"Power |Στ·v| median: static≈{static_power_abs_med:.6g}, moving≈{moving_power_abs_med:.6g} "
            f"(static should be much smaller; if not, suspect τ or v issues / timestamp mismatch)."
        )

    return {
        "t_aligned": t_a,
        "v_fd_aligned": v_fd_a,
        "v_log_aligned": v_log_a,
        "a_fd_aligned": a_fd_a,
        "tau_aligned": tau_a,
        "static_mask": static_mask,
        "power_total": power_total,
        "tau_static_mean": tau_static_mean,
        "tau_static_std": tau_static_std,
        "per_joint_fit": per_joint_fit,
        "diagnosis": diagnosis,
    }


def plot_velocity_compare(t: np.ndarray, v_fd: np.ndarray, v_log: np.ndarray, save: Optional[str] = None):
    dof = v_fd.shape[1]
    fig, axs = plt.subplots(dof, 1, sharex=True, figsize=(12, 2.0 * dof))
    if dof == 1:
        axs = [axs]
    for j in range(dof):
        axs[j].plot(t, v_fd[:, j], label=f"v_fd[{j}] (d/dt Joint)")
        axs[j].plot(t, v_log[:, j], label=f"v_log[{j}] (JointVelocity)")
        axs[j].set_ylabel("rad/s")
        axs[j].grid(True, alpha=0.3)
        axs[j].legend(loc="best")
    axs[-1].set_xlabel("t (s)")
    fig.suptitle("Velocity validation: finite-diff vs logged")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200)
    return fig


def plot_torque_power(t: np.ndarray, tau: np.ndarray, power_total: np.ndarray, static_mask: np.ndarray, save: Optional[str] = None):
    dof = tau.shape[1]
    fig, axs = plt.subplots(dof + 1, 1, sharex=True, figsize=(12, 2.0 * (dof + 1)))
    for j in range(dof):
        axs[j].plot(t, tau[:, j], label=f"tau[{j}] (JointTorque)")
        axs[j].grid(True, alpha=0.3)
        axs[j].legend(loc="best")
        axs[j].set_ylabel("Nm/N")

    axs[-1].plot(t, power_total, label="total power Σ(τ·v)")
    if np.any(static_mask):
        axs[-1].plot(t[static_mask], power_total[static_mask], ".", label="static samples")
    axs[-1].grid(True, alpha=0.3)
    axs[-1].legend(loc="best")
    axs[-1].set_ylabel("W (relative)")
    axs[-1].set_xlabel("t (s)")
    fig.suptitle("Torque & power sanity checks")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200)
    return fig


def main():
    ap = argparse.ArgumentParser("Verify JointVelocity & JointTorque against Joint position in jsonl.")
    ap.add_argument("jsonl", help="Path to jsonl file")
    ap.add_argument("--robot-key", default="Robot0")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--smooth-win", type=int, default=1, help="Odd window for moving average on Joint before diff (e.g. 5, 11).")
    ap.add_argument("--max-lag", type=int, default=5, help="Search lag in samples for best velocity correlation.")
    ap.add_argument("--v-static-eps", type=float, default=2e-2, help="Velocity threshold for static detection (rad/s).")
    ap.add_argument("--plot", action="store_true", help="Show plots.")
    ap.add_argument("--save-prefix", default="", help="Save plots with this prefix (e.g. out).")
    args = ap.parse_args()

    t, q, v_log, tau_log = load_jsonl(args.jsonl, robot_key=args.robot_key, stride=max(1, args.stride))

    # dt sanity
    dt = np.diff(t)
    print(f"[INFO] Loaded N={len(t)} samples, dof={q.shape[1]}")
    print(f"[INFO] dt mean={np.mean(dt):.6f}s, std={np.std(dt):.6f}s, min={np.min(dt):.6f}s, max={np.max(dt):.6f}s")

    # smooth q if requested (helps noisy differentiation)
    if args.smooth_win > 1:
        if args.smooth_win % 2 == 0:
            raise ValueError("--smooth-win must be odd")
        q_s = moving_average(q, args.smooth_win)
    else:
        q_s = q

    # velocity checks
    vel_res = check_velocity(t, q_s, v_log, max_lag=args.max_lag)
    lag = vel_res["lag"]
    print(f"\n[VELOCITY] Best lag (samples) = {lag}  (lag>0 means JointVelocity is delayed)")
    for s in vel_res["diagnosis"]:
        print(f"[VELOCITY] {s}")

    print("[VELOCITY] Per-joint metrics: y=scale*x+b, x=v_fd, y=v_log")
    print("  joint   scale        bias        corr      rmse")
    for r in vel_res["per_joint"]:
        print(f"  {r['j']:>5d}  {r['scale']:>10.6g}  {r['bias']:>10.6g}  {r['corr']:>8.4f}  {r['rmse']:>10.6g}")

    print("[VELOCITY] q reconstruction RMS error per joint (integrate v_log):")
    print(" ", " ".join([f"{e:.6g}" for e in vel_res["q_err_rms"]]))

    # torque checks (consistency)
    tau_res = check_torque(
        t=t,
        q=q_s,
        v_log=v_log,
        tau_log=tau_log,
        lag_vel=lag,
        v_static_eps=args.v_static_eps,
    )
    print("\n[TORQUE] Static torque mean/std per joint (|v_fd|<eps):")
    for j in range(q.shape[1]):
        m = tau_res["tau_static_mean"][j]
        s = tau_res["tau_static_std"][j]
        print(f"  joint {j}: mean={m:.6g}, std={s:.6g}")

    for s in tau_res["diagnosis"]:
        print(f"[TORQUE] {s}")

    print("[TORQUE] Linear consistency on moving segments: tau ≈ k*ddq + b (R^2 indicates inertia-like consistency only)")
    print("  joint   k(tau/ddq)    b        R^2")
    for r in tau_res["per_joint_fit"]:
        print(f"  {r['j']:>5d}  {r['tau_vs_ddq_k']:>10.6g}  {r['tau_vs_ddq_b']:>10.6g}  {r['r2']:>8.4f}")

    # plots
    save_prefix = args.save_prefix.strip()
    save_v = f"{save_prefix}_vel_check.png" if save_prefix else None
    save_tau = f"{save_prefix}_tau_power.png" if save_prefix else None

    # align arrays for plotting
    v_fd = vel_res["v_fd"]
    v_fd_a, v_log_a = shift_by_lag(v_fd, lag)
    t_a, _ = shift_by_lag(t, lag)

    if args.plot or save_prefix:
        plot_velocity_compare(t_a, v_fd_a, v_log_a, save=save_v)
        plot_torque_power(
            tau_res["t_aligned"],
            tau_res["tau_aligned"],
            tau_res["power_total"],
            tau_res["static_mask"],
            save=save_tau,
        )

    if args.plot:
        plt.show()


if __name__ == "__main__":
    main()
