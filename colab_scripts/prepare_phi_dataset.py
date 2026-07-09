#!/usr/bin/env python3
"""Build the rotating-frame average electric potential and estimate spoke rotation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


# =============================================================================
# Editable calculation parameters
# =============================================================================

STATIONARY_START_STEP = 1100000
STEADY_FRACTION = 0.55
DT = 1.0

CENTER_X = None
CENTER_Y = None

DOMAIN_EPS = 0.0
DOMAIN_MIN_OCCUPANCY = 0.10
N_RADIAL_BINS = 32

USE_POSITIVE_RESIDUAL = True
SIGNED_TARGET = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data_phi", help="Directory with phi_*.txt files.")
    parser.add_argument("--out-dir", default="outputs_phi/rotating_average", help="Output directory.")
    parser.add_argument("--pattern", default="phi_*.txt", help="Input filename pattern.")
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


def load_frame(path: Path) -> np.ndarray:
    frame = np.loadtxt(path)
    if frame.ndim != 2:
        raise ValueError(f"{path} is not a 2D matrix.")
    return frame if SIGNED_TARGET else np.abs(frame)


def stationary_indices(steps: np.ndarray) -> np.ndarray:
    if STATIONARY_START_STEP is not None:
        idx = np.flatnonzero(steps >= STATIONARY_START_STEP)
    else:
        first = int(math.floor(len(steps) * STEADY_FRACTION))
        idx = np.arange(first, len(steps))
    if idx.size < 2:
        raise ValueError("Stationary interval has fewer than 2 frames.")
    return idx


def make_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float, float]:
    ny, nx = shape
    cx = (nx - 1) / 2.0 if CENTER_X is None else float(CENTER_X)
    cy = (ny - 1) / 2.0 if CENTER_Y is None else float(CENTER_Y)
    col, row = np.meshgrid(np.arange(nx), np.arange(ny))
    x = col - cx
    y = row - cy
    return x, y, cx, cy


def estimate_domain_mask(paths: list[Path], frame_indices: np.ndarray) -> np.ndarray:
    occupancy = None
    for idx in frame_indices:
        frame = load_frame(paths[idx])
        nonzero = np.abs(frame) > DOMAIN_EPS
        if occupancy is None:
            occupancy = np.zeros(frame.shape, dtype=float)
        elif occupancy.shape != frame.shape:
            raise ValueError(f"Inconsistent frame shape in {paths[idx]}.")
        occupancy += nonzero
    if occupancy is None:
        raise ValueError("Cannot estimate plasma domain without frames.")
    mask = occupancy / len(frame_indices) >= DOMAIN_MIN_OCCUPANCY
    if not np.any(mask):
        raise ValueError("Plasma-domain mask is empty. Lower DOMAIN_EPS or DOMAIN_MIN_OCCUPANCY.")
    return mask


def subtract_radial_background(frame: np.ndarray, r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    residual = np.zeros_like(frame, dtype=float)
    background = np.zeros_like(frame, dtype=float)

    r_valid = r[mask]
    bins = np.linspace(float(r_valid.min()), float(r_valid.max()), N_RADIAL_BINS + 1)
    bin_id = np.digitize(r, bins) - 1

    for k in range(N_RADIAL_BINS):
        selected = mask & (bin_id == k)
        if np.any(selected):
            background[selected] = np.mean(frame[selected])

    residual[mask] = frame[mask] - background[mask]
    return residual


def estimate_phase(frame: np.ndarray, r: np.ndarray, theta: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    residual = subtract_radial_background(frame, r, mask)
    if USE_POSITIVE_RESIDUAL:
        weights = np.where(mask, np.maximum(residual, 0.0), 0.0)
    else:
        weights = np.where(mask, residual, 0.0)

    weight_sum = float(np.sum(np.abs(weights)))
    if weight_sum <= 0.0:
        weights = np.where(mask, frame, 0.0)
        weight_sum = float(np.sum(np.abs(weights)))

    harmonic = np.sum(weights * np.exp(1j * theta))
    phase = float(np.angle(harmonic))
    strength = float(np.abs(harmonic) / (weight_sum + 1e-30))
    return phase, strength


def rotate_to_spoke_frame(frame: np.ndarray, alpha: float, x: np.ndarray, y: np.ndarray, cx: float, cy: float) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    x_lab = x * cos_a - y * sin_a
    y_lab = x * sin_a + y * cos_a
    cols = x_lab + cx
    rows = y_lab + cy

    return map_coordinates(
        frame,
        [rows.ravel(), cols.ravel()],
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(frame.shape)


def save_phase_plot(out_dir: Path, times: np.ndarray, phases: np.ndarray, fit: np.ndarray, omega: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    ax[0].plot(times, phases, "o", ms=4, label="measured phase")
    ax[0].plot(times, fit, "-", lw=2, label="linear fit")
    ax[0].set_title(f"Potential spoke rotation fit: omega = {omega:.12g}")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("unwrapped phase, rad")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(times, phases - fit, "o-", ms=4)
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("phase residual, rad")
    ax[1].grid(True, alpha=0.3)

    fig.savefig(out_dir / "phase_fit.png", dpi=180)
    plt.close(fig)


def save_average_plot(out_dir: Path, mean_pattern: np.ndarray, std_pattern: np.ndarray, last_frame: np.ndarray, mask: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    field_lim = max(
        float(np.nanpercentile(np.abs(last_frame[mask]), 99.5)),
        float(np.nanpercentile(np.abs(mean_pattern[np.isfinite(mean_pattern)]), 99.5)),
        1e-12,
    )

    im0 = ax[0].imshow(np.where(mask, last_frame, np.nan), origin="lower", cmap="coolwarm", vmin=-field_lim, vmax=field_lim)
    ax[0].set_title("last stationary frame")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(mean_pattern, origin="lower", cmap="coolwarm", vmin=-field_lim, vmax=field_lim)
    ax[1].set_title("mean rotating-frame potential")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(std_pattern, origin="lower", cmap="viridis")
    ax[2].set_title("std after alignment")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    for axis in ax:
        axis.set_xlabel("x cell")
        axis.set_ylabel("y cell")

    fig.savefig(out_dir / "rotating_average_2d.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_paths(Path(args.data_dir), args.pattern)
    steps = np.array([extract_step(path) for path in paths], dtype=float)
    times = steps * DT
    steady_idx = stationary_indices(steps)

    first = load_frame(paths[steady_idx[0]])
    x, y, cx, cy = make_grid(first.shape)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    domain_mask = estimate_domain_mask(paths, steady_idx)

    phases = []
    strengths = []
    frames = []
    for idx in steady_idx:
        frame = load_frame(paths[idx])
        if frame.shape != first.shape:
            raise ValueError(f"Inconsistent frame shape: {paths[idx]} has {frame.shape}, expected {first.shape}.")
        phase, strength = estimate_phase(frame, r, theta, domain_mask)
        phases.append(phase)
        strengths.append(strength)
        frames.append(frame)

    phases = np.asarray(phases, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    stationary_steps = steps[steady_idx]
    stationary_times = times[steady_idx]
    phases_unwrapped = np.unwrap(phases)
    omega, phi0 = np.polyfit(stationary_times, phases_unwrapped, 1)
    phase_fit = omega * stationary_times + phi0

    aligned_frames = []
    for frame, time_value in zip(frames, stationary_times):
        alpha = omega * time_value + phi0
        aligned_frames.append(rotate_to_spoke_frame(frame, alpha, x, y, cx, cy))
    aligned_frames = np.asarray(aligned_frames, dtype=float)

    valid_count = np.sum(np.isfinite(aligned_frames), axis=0)
    mean_pattern = np.nanmean(aligned_frames, axis=0)
    std_pattern = np.nanstd(aligned_frames, axis=0)
    mean_pattern = np.where(valid_count > 0.5 * len(aligned_frames), mean_pattern, np.nan)

    coordinate_scale = float(max(np.max(np.abs(x[domain_mask])), np.max(np.abs(y[domain_mask])), 1.0))
    phase_rmse = float(np.sqrt(np.mean((phases_unwrapped - phase_fit) ** 2)))

    np.savez_compressed(
        out_dir / "rotating_average.npz",
        mean_pattern=mean_pattern,
        std_pattern=std_pattern,
        aligned_frames=aligned_frames,
        steps=stationary_steps,
        times=stationary_times,
        phases=phases,
        phases_unwrapped=phases_unwrapped,
        phase_fit=phase_fit,
        harmonic_strengths=strengths,
        omega=float(omega),
        phi0=float(phi0),
        center_x=float(cx),
        center_y=float(cy),
        coordinate_scale=coordinate_scale,
        domain_mask=domain_mask,
    )

    metadata = {
        "data_dir": str(Path(args.data_dir)),
        "pattern": args.pattern,
        "shape": list(first.shape),
        "stationary_steps": stationary_steps.tolist(),
        "stationary_times": stationary_times.tolist(),
        "stationary_start_step": STATIONARY_START_STEP,
        "steady_fraction": STEADY_FRACTION,
        "dt": DT,
        "center_x_cell": float(cx),
        "center_y_cell": float(cy),
        "coordinate_scale": coordinate_scale,
        "domain_eps": DOMAIN_EPS,
        "domain_min_occupancy": DOMAIN_MIN_OCCUPANCY,
        "domain_cell_count": int(np.count_nonzero(domain_mask)),
        "n_radial_bins": N_RADIAL_BINS,
        "use_positive_residual": USE_POSITIVE_RESIDUAL,
        "signed_target": SIGNED_TARGET,
        "omega": float(omega),
        "phi0": float(phi0),
        "phase_rmse": phase_rmse,
        "mean_harmonic_strength": float(np.mean(strengths)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_phase_plot(out_dir, stationary_times, phases_unwrapped, phase_fit, float(omega))
    save_average_plot(out_dir, mean_pattern, std_pattern, frames[-1], domain_mask)

    print(f"Saved rotating potential average to {out_dir / 'rotating_average.npz'}")
    print(f"Saved metadata to {out_dir / 'metadata.json'}")
    print(f"Stationary frames: {len(steady_idx)} / {len(paths)}")
    print(f"Domain cells: {metadata['domain_cell_count']} / {first.size}")
    print(f"omega = {omega:.12g}")
    print(f"phi0 = {phi0:.12g}")
    print(f"phase_rmse = {phase_rmse:.6g}")
    print(f"mean_harmonic_strength = {metadata['mean_harmonic_strength']:.6g}")


if __name__ == "__main__":
    main()
