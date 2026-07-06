#!/usr/bin/env python3
"""Prepare a compact stationary-spoke dataset for PySR in Colab."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with rho_i_*.txt files.")
    parser.add_argument("--pattern", default="rho_i_*.txt", help="Input filename pattern.")
    parser.add_argument("--out-dir", default="colab_outputs/prepared", help="Output directory.")
    parser.add_argument("--dx", type=float, default=1.0, help="Physical x spacing per cell.")
    parser.add_argument("--dy", type=float, default=1.0, help="Physical y spacing per cell.")
    parser.add_argument("--dt", type=float, default=1.0, help="Physical time per iteration unit.")
    parser.add_argument("--center-x", type=float, default=None, help="Grid center x in cell-index units.")
    parser.add_argument("--center-y", type=float, default=None, help="Grid center y in cell-index units.")
    parser.add_argument("--steady-fraction", type=float, default=0.55, help="Use frames after this fraction of the sequence.")
    parser.add_argument("--t-start", type=float, default=None, help="Optional physical start time for stationary interval.")
    parser.add_argument("--t-end", type=float, default=None, help="Optional physical end time for stationary interval.")
    parser.add_argument("--r-min-frac", type=float, default=0.08, help="Ignore central disk below this fraction of max radius.")
    parser.add_argument("--r-max-frac", type=float, default=0.96, help="Ignore outer cells above this fraction of max radius.")
    parser.add_argument("--samples-per-frame", type=int, default=1200, help="Training samples drawn per stationary frame.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--smooth-sigma", type=float, default=0.8, help="Gaussian smoothing in cells before fitting.")
    parser.add_argument("--high-density-fraction", type=float, default=0.55, help="Fraction of samples biased toward high density.")
    parser.add_argument("--signed-target", action="store_true", help="Fit signed rho_i instead of abs(rho_i).")
    parser.add_argument("--random-state", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot extract iteration number from {path.name}")
    return int(match.group(1))


def load_paths(data_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(data_dir.glob(pattern), key=extract_step)
    if not paths:
        raise FileNotFoundError(f"No files matched {data_dir / pattern}")
    return paths


def make_grid(shape: tuple[int, int], dx: float, dy: float, cx: float | None, cy: float | None):
    ny, nx = shape
    if cx is None:
        cx = (nx - 1) / 2.0
    if cy is None:
        cy = (ny - 1) / 2.0
    j, i = np.meshgrid(np.arange(nx), np.arange(ny))
    x = (j - cx) * dx
    y = (i - cy) * dy
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return x, y, r, theta, float(cx), float(cy)


def stationary_indices(times: np.ndarray, steady_fraction: float, t_start: float | None, t_end: float | None) -> np.ndarray:
    mask = np.ones(times.shape, dtype=bool)
    if t_start is not None:
        mask &= times >= t_start
    if t_end is not None:
        mask &= times <= t_end
    if t_start is None and t_end is None:
        first = int(math.floor(len(times) * steady_fraction))
        mask[:first] = False
    idx = np.flatnonzero(mask)
    if idx.size < 5:
        raise ValueError("Stationary interval has fewer than 5 frames. Relax --steady-fraction/--t-start/--t-end.")
    return idx


def estimate_phase(frame: np.ndarray, theta: np.ndarray, mask: np.ndarray) -> float:
    values = frame[mask]
    baseline = np.quantile(values, 0.55)
    weights = np.clip(values - baseline, 0.0, None)
    if not np.any(weights > 0):
        weights = np.clip(values, 0.0, None)
    moment = np.sum(weights * np.exp(1j * theta[mask]))
    return float(np.angle(moment))


def fit_rotation(times: np.ndarray, phases: np.ndarray) -> tuple[float, float, np.ndarray]:
    unwrapped = np.unwrap(phases)
    omega, phase0 = np.polyfit(times, unwrapped, deg=1)
    fitted = omega * times + phase0
    return float(omega), float(phase0), fitted


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def sample_frame(
    rng: np.random.Generator,
    frame: np.ndarray,
    base_mask: np.ndarray,
    n_samples: int,
    high_density_fraction: float,
) -> np.ndarray:
    flat_mask = np.flatnonzero(base_mask.ravel())
    if flat_mask.size == 0:
        raise ValueError("Sampling mask is empty.")

    n_high = int(round(n_samples * high_density_fraction))
    n_random = n_samples - n_high
    values = frame.ravel()[flat_mask]
    scores = np.clip(values - np.quantile(values, 0.60), 0.0, None)
    if np.sum(scores) <= 0:
        scores = np.ones_like(scores)
    probs = scores / np.sum(scores)
    high = rng.choice(flat_mask, size=n_high, replace=True, p=probs)
    random = rng.choice(flat_mask, size=n_random, replace=True)
    return np.concatenate([high, random])


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_paths(Path(args.data_dir), args.pattern)
    steps = np.array([extract_step(path) for path in paths], dtype=float)
    times = steps * args.dt

    first = np.loadtxt(paths[0])
    x, y, r, theta, center_x, center_y = make_grid(first.shape, args.dx, args.dy, args.center_x, args.center_y)
    r_max = float(np.max(r))
    annulus = (r >= args.r_min_frac * r_max) & (r <= args.r_max_frac * r_max)
    steady_idx = stationary_indices(times, args.steady_fraction, args.t_start, args.t_end)

    phase_frames = []
    phases = []
    for idx in steady_idx:
        raw = np.loadtxt(paths[idx])
        frame = raw if args.signed_target else np.abs(raw)
        if args.smooth_sigma > 0:
            frame = gaussian_filter(frame, sigma=args.smooth_sigma)
        phases.append(estimate_phase(frame, theta, annulus))
        phase_frames.append(idx)

    phase_times = times[np.array(phase_frames)]
    omega, phase0, fitted_phase = fit_rotation(phase_times, np.array(phases))

    feature_rows = []
    target_rows = []
    meta_rows = []
    for idx in steady_idx:
        raw = np.loadtxt(paths[idx])
        frame = raw if args.signed_target else np.abs(raw)
        if args.smooth_sigma > 0:
            frame = gaussian_filter(frame, sigma=args.smooth_sigma)
        picked = sample_frame(rng, frame, annulus, args.samples_per_frame, args.high_density_fraction)
        iy, ix = np.unravel_index(picked, frame.shape)
        psi = wrap_angle(theta[iy, ix] - omega * times[idx] - phase0)
        r_n = r[iy, ix] / r_max
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        xrot_n = r_n * cpsi
        yrot_n = r_n * spsi
        features = np.column_stack([r_n, cpsi, spsi, xrot_n, yrot_n])
        feature_rows.append(features)
        target_rows.append(frame[iy, ix])
        meta_rows.append(np.column_stack([np.full_like(r_n, steps[idx]), np.full_like(r_n, times[idx]), x[iy, ix], y[iy, ix]]))

    X = np.vstack(feature_rows)
    y_target = np.concatenate(target_rows)
    sample_meta = np.vstack(meta_rows)
    target_scale = float(np.percentile(np.abs(y_target), 99.0))
    if target_scale <= 0:
        target_scale = 1.0
    y_scaled = y_target / target_scale

    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X,
        y_scaled,
        sample_meta,
        test_size=args.validation_size,
        random_state=args.random_state,
    )

    feature_names = np.array(["r_n", "cpsi", "spsi", "xrot_n", "yrot_n"])
    np.savez_compressed(
        out_dir / "spoke_dataset.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        meta_train=meta_train,
        meta_val=meta_val,
        feature_names=feature_names,
    )

    metadata = {
        "data_dir": str(Path(args.data_dir)),
        "pattern": args.pattern,
        "n_frames_total": len(paths),
        "stationary_frame_indices": steady_idx.tolist(),
        "stationary_steps": steps[steady_idx].tolist(),
        "stationary_times": times[steady_idx].tolist(),
        "shape": list(first.shape),
        "dx": args.dx,
        "dy": args.dy,
        "dt": args.dt,
        "center_x_cell": center_x,
        "center_y_cell": center_y,
        "r_scale": r_max,
        "omega": omega,
        "phase0": phase0,
        "target_scale": target_scale,
        "target_kind": "signed_rho_i" if args.signed_target else "abs_rho_i",
        "feature_names": feature_names.tolist(),
        "formula_mapping": "rho_i ~= target_scale * F(r/r_scale, cos(theta - omega*t - phase0), sin(theta - omega*t - phase0), (r/r_scale)*cos(...), (r/r_scale)*sin(...))",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    residual = wrap_angle(np.array(phases) - fitted_phase)
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(phase_times, np.unwrap(phases), "o", ms=3, label="tracked phase")
    ax[0].plot(phase_times, fitted_phase, "-", label=f"fit: omega={omega:.6g}")
    ax[0].set_ylabel("unwrapped phase")
    ax[0].legend()
    ax[1].plot(phase_times, residual, "o", ms=3)
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("wrapped residual")
    fig.tight_layout()
    fig.savefig(out_dir / "angle_fit.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    shown = min(6000, meta_train.shape[0])
    ax.scatter(meta_train[:shown, 2], meta_train[:shown, 3], c=y_train[:shown], s=2, cmap="magma")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sampled training points")
    fig.tight_layout()
    fig.savefig(out_dir / "sampled_points.png", dpi=180)
    plt.close(fig)

    print(f"Saved dataset to {out_dir / 'spoke_dataset.npz'}")
    print(f"Saved metadata to {out_dir / 'metadata.json'}")
    print(f"Stationary frames: {steady_idx.size} / {len(paths)}")
    print(f"Estimated omega = {omega:.10g} rad per physical time unit")
    print(f"Estimated period = {2*np.pi/abs(omega):.10g}" if omega != 0 else "Estimated period = inf")


if __name__ == "__main__":
    main()
