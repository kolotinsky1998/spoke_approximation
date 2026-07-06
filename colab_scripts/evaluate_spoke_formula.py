#!/usr/bin/env python3
"""Evaluate a fitted rotating-coordinate PySR formula on full frames."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="colab_outputs/pysr_run/model.pkl")
    parser.add_argument("--metadata", default="colab_outputs/pysr_run/metadata.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pattern", default="rho_i_*.txt")
    parser.add_argument("--out-dir", default="colab_outputs/evaluation")
    parser.add_argument("--frames", nargs="*", type=int, default=None, help="Iteration numbers to render.")
    parser.add_argument("--signed-target", action="store_true", help="Evaluate signed rho_i instead of abs(rho_i).")
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot extract iteration number from {path.name}")
    return int(match.group(1))


def make_grid(shape: tuple[int, int], metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    j, i = np.meshgrid(np.arange(nx), np.arange(ny))
    x = (j - metadata["center_x_cell"]) * metadata["dx"]
    y = (i - metadata["center_y_cell"]) * metadata["dy"]
    return x, y


def rotating_features(shape: tuple[int, int], step: int, metadata: dict) -> np.ndarray:
    x, y = make_grid(shape, metadata)
    x_n = x.ravel() / metadata["coordinate_scale"]
    y_n = y.ravel() / metadata["coordinate_scale"]
    t = step * metadata["dt"]
    alpha = metadata["omega"] * t
    c = np.cos(alpha)
    s = np.sin(alpha)
    u_n = x_n * c + y_n * s
    v_n = -x_n * s + y_n * c
    return np.column_stack([u_n, v_n])


def load_target(path: Path, signed_target: bool) -> np.ndarray:
    frame = np.loadtxt(path)
    if frame.ndim != 2:
        raise ValueError(f"{path} is not a 2D matrix.")
    return frame if signed_target else np.abs(frame)


def evaluation_mask(frame: np.ndarray, metadata: dict) -> np.ndarray:
    domain_eps = float(metadata.get("domain_eps", 0.0))
    mask = np.abs(frame) > domain_eps
    if not np.any(mask):
        mask = np.ones(frame.shape, dtype=bool)
    return mask


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0.0 else float("nan")
    return mae, rmse, r2


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
        f"rho(x,y,t) ~= {metadata['target_scale']:.12g} * F(u_n, v_n)",
        f"u_n = x_n*cos(({metadata['omega']:.12g})*t) + y_n*sin(({metadata['omega']:.12g})*t)",
        f"v_n = -x_n*sin(({metadata['omega']:.12g})*t) + y_n*cos(({metadata['omega']:.12g})*t)",
        f"x_n = x / {metadata['coordinate_scale']:.12g}",
        f"y_n = y / {metadata['coordinate_scale']:.12g}",
    ]
    (out_dir / "formula.txt").write_text("\n".join(formula_lines), encoding="utf-8")

    metric_rows = []
    for step in selected_steps:
        if step not in by_step:
            print(f"Skipping missing frame {step}")
            continue
        target = load_target(by_step[step], args.signed_target)
        mask = evaluation_mask(target, metadata)
        X = rotating_features(target.shape, step, metadata)
        pred = model.predict(X).reshape(target.shape) * metadata["target_scale"]
        residual = pred - target

        flat_target = target[mask]
        flat_pred = pred[mask]
        mae, rmse, r2 = regression_metrics(flat_target, flat_pred)
        metric_rows.append(
            {
                "step": step,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "target_max": float(np.max(target[mask])),
                "pred_max": float(np.max(pred[mask])),
            }
        )

        vmax = np.percentile(target[mask], 99.5) if np.any(mask) else np.max(target)
        lim = max(np.percentile(np.abs(residual[mask]), 99.0), 1e-12)
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
