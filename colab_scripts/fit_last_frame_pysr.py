#!/usr/bin/env python3
"""Fit the last rho_i frame as rho_i(x, y) with PySR."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with rho_i_*.txt files.")
    parser.add_argument("--pattern", default="rho_i_*.txt", help="Input filename pattern.")
    parser.add_argument("--out-dir", default="colab_outputs/pysr_last_frame", help="Output directory.")
    parser.add_argument("--dx", type=float, default=1.0, help="Physical x spacing per cell.")
    parser.add_argument("--dy", type=float, default=1.0, help="Physical y spacing per cell.")
    parser.add_argument("--center-x", type=float, default=None, help="Grid center x in cell-index units.")
    parser.add_argument("--center-y", type=float, default=None, help="Grid center y in cell-index units.")
    parser.add_argument("--domain-eps", type=float, default=0.0, help="Cells with abs(rho_i) <= this are excluded.")
    parser.add_argument("--n-samples", type=int, default=0, help="Random samples from the last frame. Use 0 to use all domain cells.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--signed-target", action="store_true", help="Fit signed rho_i instead of abs(rho_i).")
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--populations", type=int, default=12)
    parser.add_argument("--maxsize", type=int, default=40)
    parser.add_argument("--parsimony", type=float, default=0.001)
    parser.add_argument(
        "--model-selection",
        choices=["accuracy", "best", "score"],
        default="accuracy",
        help="How PySR selects the final equation.",
    )
    parser.add_argument("--top-equations", type=int, default=8, help="Number of best equations to print by loss.")
    parser.add_argument("--include-trig", action="store_true", help="Also allow sin/cos of spatial expressions.")
    parser.add_argument("--batch-size", type=int, default=None, help="Use PySR mini-batches of this size.")
    parser.add_argument("--timeout-minutes", type=float, default=None, help="PySR time budget in minutes.")
    parser.add_argument("--show-pysr-output", action="store_true", help="Show raw PySR/Julia output instead of writing it to a log file.")
    parser.add_argument("--turbo", action="store_true", help="Enable PySR turbo mode. Faster, but may print Julia LoopVectorization warnings.")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--procs", type=int, default=0, help="Julia worker processes. 0 lets PySR choose.")
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot extract iteration number from {path.name}")
    return int(match.group(1))


def load_last_frame(data_dir: Path, pattern: str, signed_target: bool) -> tuple[Path, int, np.ndarray]:
    paths = sorted(data_dir.glob(pattern), key=extract_step)
    if not paths:
        raise FileNotFoundError(f"No files matched {data_dir / pattern}")
    path = paths[-1]
    frame = np.loadtxt(path)
    if frame.ndim != 2:
        raise ValueError(f"{path} is not a 2D matrix.")
    target = frame if signed_target else np.abs(frame)
    return path, extract_step(path), target


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


def split_dataset(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    validation_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("--validation-size must be between 0 and 1.")
    indices = rng.permutation(len(y))
    n_val = max(1, int(round(len(y) * validation_size)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if train_idx.size == 0:
        raise ValueError("Training split is empty. Lower --validation-size or increase --n-samples.")
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def choose_samples(
    rng: np.random.Generator,
    candidates: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    if n_samples <= 0 or n_samples >= candidates.size:
        return candidates
    return rng.choice(candidates, size=n_samples, replace=False)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_scale: float, prefix: str) -> dict:
    residual = y_pred - y_true
    rmse_scaled = float(np.sqrt(np.mean(residual**2)))
    mae_scaled = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0.0 else float("nan")
    return {
        f"{prefix}_mae_scaled": mae_scaled,
        f"{prefix}_rmse_scaled": rmse_scaled,
        f"{prefix}_r2": r2,
        f"{prefix}_mae_physical": mae_scaled * target_scale,
        f"{prefix}_rmse_physical": rmse_scaled * target_scale,
    }


@contextlib.contextmanager
def redirect_process_output(log_path: Path, enabled: bool):
    if not enabled:
        yield
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with log_path.open("ab") as log_file:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        raise SystemExit(
            "PySR is not installed. In Colab run:\n"
            '  %pip install -U "pysr==1.5.9" numpy pandas matplotlib joblib'
        ) from exc

    frame_path, step, target = load_last_frame(Path(args.data_dir), args.pattern, args.signed_target)
    x, y_grid, center_x, center_y = make_grid(target.shape, args.dx, args.dy, args.center_x, args.center_y)
    domain_mask = np.abs(target) > args.domain_eps
    if not np.any(domain_mask):
        raise ValueError("Domain mask is empty. Lower --domain-eps.")

    coordinate_scale = float(max(np.max(np.abs(x[domain_mask])), np.max(np.abs(y_grid[domain_mask])), 1.0))
    target_scale = float(np.percentile(np.abs(target[domain_mask]), 99.0))
    if target_scale <= 0:
        target_scale = 1.0

    candidates = np.flatnonzero(domain_mask.ravel())
    picked = choose_samples(rng, candidates, args.n_samples)
    iy, ix = np.unravel_index(picked, target.shape)
    X = np.column_stack([x[iy, ix] / coordinate_scale, y_grid[iy, ix] / coordinate_scale])
    y_scaled = target[iy, ix] / target_scale
    X_train, X_val, y_train, y_val = split_dataset(rng, X, y_scaled, args.validation_size)

    unary_operators = ["sqrt", "abs", "exp"]
    constraints = {
        "sqrt": 9,
        "exp": 9,
        "/": (-1, 9),
    }
    nested_constraints = {
        "exp": {"exp": 0},
    }
    if args.include_trig:
        unary_operators += ["sin", "cos"]
        constraints.update({"sin": 9, "cos": 9})
        nested_constraints.update(
            {
                "sin": {"sin": 0, "cos": 0, "exp": 0},
                "cos": {"sin": 0, "cos": 0, "exp": 0},
            }
        )

    model_kwargs = dict(
        niterations=args.niterations,
        populations=args.populations,
        maxsize=args.maxsize,
        parsimony=args.parsimony,
        model_selection=args.model_selection,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=unary_operators,
        constraints=constraints,
        nested_constraints=nested_constraints,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=args.random_state,
        turbo=args.turbo,
        progress=False,
        verbosity=0,
        output_directory=str(out_dir / "pysr_outputs"),
    )
    if args.procs > 0:
        model_kwargs["procs"] = args.procs
    if args.timeout_minutes is not None:
        model_kwargs["timeout_in_seconds"] = args.timeout_minutes * 60.0
    if args.batch_size is not None:
        model_kwargs["batching"] = True
        model_kwargs["batch_size"] = args.batch_size

    log_path = out_dir / "pysr.log"
    print(f"Fitting last frame: {frame_path.name} (step={step})")
    print(f"Samples: train={len(y_train)}, validation={len(y_val)}, domain_cells={int(np.count_nonzero(domain_mask))}")
    print(f"PySR output is written to: {log_path}")
    start = time.monotonic()
    with redirect_process_output(log_path, enabled=not args.show_pysr_output):
        model = PySRRegressor(**model_kwargs)
        model.fit(X_train, y_train, variable_names=["x_n", "y_n"])
    print(f"PySR finished in {(time.monotonic() - start) / 60.0:.2f} min")

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    metrics = {
        **regression_metrics(y_train, pred_train, target_scale, "train"),
        **regression_metrics(y_val, pred_val, target_scale, "val"),
        "target_scale": target_scale,
    }

    full_X = np.column_stack([x.ravel() / coordinate_scale, y_grid.ravel() / coordinate_scale])
    pred = model.predict(full_X).reshape(target.shape) * target_scale
    residual = pred - target
    mask = domain_mask

    metadata = {
        "frame_path": str(frame_path),
        "step": step,
        "shape": list(target.shape),
        "dx": args.dx,
        "dy": args.dy,
        "center_x_cell": center_x,
        "center_y_cell": center_y,
        "coordinate_scale": coordinate_scale,
        "domain_eps": args.domain_eps,
        "domain_cell_count": int(np.count_nonzero(domain_mask)),
        "target_scale": target_scale,
        "target_kind": "signed_rho_i" if args.signed_target else "abs_rho_i",
        "feature_names": ["x_n", "y_n"],
        "model_selection": args.model_selection,
        "unary_operators": unary_operators,
        "metrics": metrics,
    }

    joblib.dump(model, out_dir / "model.pkl")
    model.equations_.to_csv(out_dir / "equations.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    equation_rows = model.equations_.sort_values("loss", ascending=True).head(args.top_equations)
    top_lines = [
        f"complexity={row['complexity']} loss={row['loss']:.6g} equation={row['equation']}"
        for _, row in equation_rows.iterrows()
    ]

    formula_text = [
        "Selected PySR formula for scaled last-frame density:",
        str(model.get_best()["equation"]),
        "",
        "Physical mapping:",
        f"rho(x,y) ~= {target_scale:.12g} * F(x_n, y_n)",
        f"x_n = x / {coordinate_scale:.12g}",
        f"y_n = y / {coordinate_scale:.12g}",
        "",
        "Top equations by loss:",
        *top_lines,
        "",
        "Validation metrics:",
        json.dumps(metrics, indent=2),
    ]
    (out_dir / "formula.txt").write_text("\n".join(formula_text), encoding="utf-8")

    vmax = np.percentile(target[mask], 99.5)
    lim = max(np.percentile(np.abs(residual[mask]), 99.0), 1e-12)
    target_plot = np.where(mask, target, np.nan)
    pred_plot = np.where(mask, pred, np.nan)
    residual_plot = np.where(mask, residual, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    im0 = ax[0].imshow(target_plot, origin="lower", cmap="magma", vmax=vmax)
    ax[0].set_title(f"data step={step}")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(pred_plot, origin="lower", cmap="magma", vmax=vmax)
    ax[1].set_title("PySR formula")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(residual_plot, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[2].set_title("prediction - data")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)
    for axis in ax:
        axis.set_xlabel("x cell")
        axis.set_ylabel("y cell")
    fig.savefig(out_dir / "comparison_last_frame.png", dpi=180)
    plt.close(fig)

    print("\n".join(formula_text))
    print(f"\nSaved model to {out_dir / 'model.pkl'}")
    print(f"Saved comparison to {out_dir / 'comparison_last_frame.png'}")


if __name__ == "__main__":
    main()
