#!/usr/bin/env python3
"""Evaluate a fitted PySR spoke formula on full frames."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="colab_outputs/pysr_run/model.pkl")
    parser.add_argument("--metadata", default="colab_outputs/prepared/metadata.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pattern", default="rho_i_*.txt")
    parser.add_argument("--out-dir", default="colab_outputs/evaluation")
    parser.add_argument("--frames", nargs="*", type=int, default=None, help="Iteration numbers to render.")
    parser.add_argument("--smooth-sigma", type=float, default=0.8, help="Match preparation smoothing.")
    parser.add_argument("--signed-target", action="store_true", help="Evaluate signed rho_i instead of abs(rho_i).")
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot extract iteration number from {path.name}")
    return int(match.group(1))


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def make_features(shape: tuple[int, int], step: int, metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = shape
    j, i = np.meshgrid(np.arange(nx), np.arange(ny))
    x = (j - metadata["center_x_cell"]) * metadata["dx"]
    y = (i - metadata["center_y_cell"]) * metadata["dy"]
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    t = step * metadata["dt"]
    psi = wrap_angle(theta - metadata["omega"] * t - metadata["phase0"])
    r_n = r / metadata["r_scale"]
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    X = np.column_stack([
        r_n.ravel(),
        cpsi.ravel(),
        spsi.ravel(),
        (r_n * cpsi).ravel(),
        (r_n * spsi).ravel(),
    ])
    return X, x, y


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    paths = sorted(Path(args.data_dir).glob(args.pattern), key=extract_step)
    by_step = {extract_step(path): path for path in paths}
    if not by_step:
        raise FileNotFoundError(f"No files matched {Path(args.data_dir) / args.pattern}")

    if args.frames:
        selected_steps = args.frames
    else:
        stationary = metadata["stationary_steps"]
        selected_steps = [int(stationary[i]) for i in np.linspace(0, len(stationary) - 1, 4, dtype=int)]

    formula_lines = [
        "Selected PySR formula for scaled density:",
        str(model.get_best()["equation"]),
        "",
        "Physical mapping:",
        f"rho_i ~= {metadata['target_scale']:.12g} * F(r_n, cpsi, spsi, xrot_n, yrot_n)",
        f"r_n = sqrt(x^2 + y^2) / {metadata['r_scale']:.12g}",
        f"theta = atan2(y, x)",
        f"psi = theta - ({metadata['omega']:.12g})*t - ({metadata['phase0']:.12g})",
        "cpsi = cos(psi)",
        "spsi = sin(psi)",
        "xrot_n = r_n*cpsi",
        "yrot_n = r_n*spsi",
    ]
    (out_dir / "formula.txt").write_text("\n".join(formula_lines), encoding="utf-8")

    metric_rows = []
    for step in selected_steps:
        if step not in by_step:
            print(f"Skipping missing frame {step}")
            continue
        raw = np.loadtxt(by_step[step])
        target = raw if args.signed_target else np.abs(raw)
        if args.smooth_sigma > 0:
            target = gaussian_filter(target, sigma=args.smooth_sigma)
        X, x, y = make_features(target.shape, step, metadata)
        pred = model.predict(X).reshape(target.shape) * metadata["target_scale"]
        residual = pred - target
        flat_target = target.ravel()
        flat_pred = pred.ravel()
        metric_rows.append(
            {
                "step": step,
                "mae": mean_absolute_error(flat_target, flat_pred),
                "rmse": mean_squared_error(flat_target, flat_pred, squared=False),
                "r2": r2_score(flat_target, flat_pred),
                "target_max": float(np.max(target)),
                "pred_max": float(np.max(pred)),
            }
        )

        vmax = np.percentile(target[target > 0], 99.5) if np.any(target > 0) else np.max(target)
        lim = max(np.percentile(np.abs(residual), 99.0), 1e-12)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        im0 = ax[0].imshow(target, origin="lower", cmap="magma", vmax=vmax)
        ax[0].set_title(f"data step={step}")
        fig.colorbar(im0, ax=ax[0], fraction=0.046)
        im1 = ax[1].imshow(pred, origin="lower", cmap="magma", vmax=vmax)
        ax[1].set_title("PySR formula")
        fig.colorbar(im1, ax=ax[1], fraction=0.046)
        im2 = ax[2].imshow(residual, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
        ax[2].set_title("prediction - data")
        fig.colorbar(im2, ax=ax[2], fraction=0.046)
        for axis in ax:
            axis.set_xlabel("x cell")
            axis.set_ylabel("y cell")
        fig.savefig(out_dir / f"comparison_{step}.png", dpi=180)
        plt.close(fig)

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "mae", "rmse", "r2", "target_max", "pred_max"])
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"Saved formula to {out_dir / 'formula.txt'}")
    print(f"Saved metrics to {out_dir / 'metrics.csv'}")
    for row in metric_rows:
        print(f"step={row['step']} mae={row['mae']:.6g} rmse={row['rmse']:.6g} r2={row['r2']:.4f}")


if __name__ == "__main__":
    main()
