#!/usr/bin/env python3
"""Prepare a direct x, y, t -> rho_i dataset for PySR in Colab."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


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
    parser.add_argument("--domain-eps", type=float, default=0.0, help="Cells with abs(rho_i) <= this are excluded.")
    parser.add_argument("--domain-min-occupancy", type=float, default=0.10, help="Fraction of stationary frames where a cell must be nonzero to enter the domain.")
    parser.add_argument("--samples-per-frame", type=int, default=300, help="Random samples drawn per stationary frame.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation fraction.")
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
    return x, y, float(cx), float(cy)


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
    if idx.size < 2:
        raise ValueError("Stationary interval has fewer than 2 frames. Relax --steady-fraction/--t-start/--t-end.")
    return idx


def load_target(path: Path, signed_target: bool) -> np.ndarray:
    frame = np.loadtxt(path)
    if frame.ndim != 2:
        raise ValueError(f"{path} is not a 2D matrix.")
    return frame if signed_target else np.abs(frame)


def estimate_domain_mask(
    paths: list[Path],
    frame_indices: np.ndarray,
    signed_target: bool,
    domain_eps: float,
    min_occupancy: float,
) -> np.ndarray:
    occupancy = None
    for idx in frame_indices:
        frame = load_target(paths[idx], signed_target)
        nonzero = np.abs(frame) > domain_eps
        if occupancy is None:
            occupancy = np.zeros(nonzero.shape, dtype=float)
        elif occupancy.shape != nonzero.shape:
            raise ValueError(f"Inconsistent frame shape in {paths[idx]}.")
        occupancy += nonzero
    if occupancy is None:
        raise ValueError("Cannot estimate domain mask without frames.")
    mask = occupancy / len(frame_indices) >= min_occupancy
    if not np.any(mask):
        raise ValueError("Estimated plasma-domain mask is empty. Lower --domain-eps or --domain-min-occupancy.")
    return mask


def sample_frame(rng: np.random.Generator, frame: np.ndarray, domain_mask: np.ndarray, n_samples: int) -> np.ndarray:
    candidates = np.flatnonzero((domain_mask & (np.abs(frame) > 0.0)).ravel())
    if candidates.size == 0:
        candidates = np.flatnonzero(domain_mask.ravel())
    if candidates.size == 0:
        raise ValueError("Sampling mask is empty.")
    replace = n_samples > candidates.size
    return rng.choice(candidates, size=n_samples, replace=replace)


def train_validation_split(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray,
    validation_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("--validation-size must be between 0 and 1.")
    indices = rng.permutation(len(y))
    n_val = max(1, int(round(len(y) * validation_size)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if train_idx.size == 0:
        raise ValueError("Training split is empty. Lower --validation-size or increase samples.")
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx], meta[train_idx], meta[val_idx]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_paths(Path(args.data_dir), args.pattern)
    steps = np.array([extract_step(path) for path in paths], dtype=float)
    times = steps * args.dt
    steady_idx = stationary_indices(times, args.steady_fraction, args.t_start, args.t_end)

    first = load_target(paths[0], args.signed_target)
    x, y, center_x, center_y = make_grid(first.shape, args.dx, args.dy, args.center_x, args.center_y)
    domain_mask = estimate_domain_mask(paths, steady_idx, args.signed_target, args.domain_eps, args.domain_min_occupancy)
    if domain_mask.shape != first.shape:
        raise ValueError(f"Domain mask has shape {domain_mask.shape}, expected {first.shape}.")

    x_scale = float(np.max(np.abs(x[domain_mask])))
    y_scale = float(np.max(np.abs(y[domain_mask])))
    t0 = float(times[steady_idx[0]])
    t_scale = float(max(times[steady_idx[-1]] - t0, 1.0))
    coordinate_scale = float(max(x_scale, y_scale, 1.0))

    feature_rows = []
    target_rows = []
    meta_rows = []
    for idx in steady_idx:
        frame = load_target(paths[idx], args.signed_target)
        if frame.shape != first.shape:
            raise ValueError(f"Inconsistent frame shape: {paths[idx]} has {frame.shape}, expected {first.shape}.")
        picked = sample_frame(rng, frame, domain_mask, args.samples_per_frame)
        iy, ix = np.unravel_index(picked, frame.shape)
        x_n = x[iy, ix] / coordinate_scale
        y_n = y[iy, ix] / coordinate_scale
        t_phys = np.full_like(x_n, times[idx], dtype=float)
        t_n = (t_phys - t0) / t_scale
        feature_rows.append(np.column_stack([x_n, y_n, t_n]))
        target_rows.append(frame[iy, ix])
        meta_rows.append(np.column_stack([np.full_like(x_n, steps[idx]), t_phys, x[iy, ix], y[iy, ix]]))

    X = np.vstack(feature_rows)
    y_target = np.concatenate(target_rows)
    sample_meta = np.vstack(meta_rows)
    target_scale = float(np.percentile(np.abs(y_target), 99.0))
    if target_scale <= 0:
        target_scale = 1.0
    y_scaled = y_target / target_scale

    X_train, X_val, y_train, y_val, meta_train, meta_val = train_validation_split(
        rng, X, y_scaled, sample_meta, args.validation_size
    )

    feature_names = np.array(["x_n", "y_n", "t_n"])
    np.savez_compressed(
        out_dir / "dataset.npz",
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
        "coordinate_scale": coordinate_scale,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "t0": t0,
        "t_scale": t_scale,
        "domain_eps": args.domain_eps,
        "domain_min_occupancy": args.domain_min_occupancy,
        "domain_cell_count": int(np.count_nonzero(domain_mask)),
        "target_scale": target_scale,
        "target_kind": "signed_rho_i" if args.signed_target else "abs_rho_i",
        "feature_names": feature_names.tolist(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved dataset to {out_dir / 'dataset.npz'}")
    print(f"Saved metadata to {out_dir / 'metadata.json'}")
    print(f"Stationary frames: {steady_idx.size} / {len(paths)}")
    print(f"Domain cells: {metadata['domain_cell_count']} / {first.size}")
    print(f"Samples: train={len(y_train)}, validation={len(y_val)}")


if __name__ == "__main__":
    main()
