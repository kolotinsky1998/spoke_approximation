#!/usr/bin/env python3
"""Fit the rotating-frame average electric potential with PySR in polar coordinates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


# =============================================================================
# Editable calculation parameters
# =============================================================================

PHI_EPS = 1e-12
RANDOM_STATE = 7
VALIDATION_SIZE = 0.2

NITERATIONS = 300000
MAXSIZE = 35
POPULATIONS = 20
PARSIMONY = 0.0003
TIMEOUT_MINUTES = 1200
PROCS = 20

POTENTIAL_WEIGHT = 0.0
POTENTIAL_WEIGHT_POWER = 1.0

BINARY_OPERATORS = ["+", "-", "*", "/"]
UNARY_OPERATORS = ["exp", "erf", "sin", "cos"]

CONSTRAINTS = {
    "exp": 10,
    "erf": 10,
    "sin": 8,
    "cos": 8,
    "/": (-1, 10),
}

NESTED_CONSTRAINTS = {
    "exp": {"exp": 0, "sin": 0, "cos": 0, "erf": 0},
    "erf": {"exp": 0, "sin": 0, "cos": 0, "erf": 0},
    "sin": {"exp": 0, "sin": 0, "cos": 0, "erf": 0},
    "cos": {"exp": 0, "sin": 0, "cos": 0, "erf": 0},
}

COMPLEXITY_OF_OPERATORS = {
    "exp": 3,
    "erf": 3,
    "sin": 3,
    "cos": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--average-file", default="outputs_phi/rotating_average/rotating_average.npz")
    parser.add_argument("--metadata", default="outputs_phi/rotating_average/metadata.json")
    parser.add_argument("--out-dir", default="outputs_phi/pysr_polar")
    return parser.parse_args()


def train_validation_split(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    validation_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("VALIDATION_SIZE must be between 0 and 1.")
    indices = rng.permutation(len(y))
    n_val = max(1, int(round(len(y) * validation_size)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if train_idx.size == 0:
        raise ValueError("Training split is empty.")
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


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


def potential_weights(y_scaled: np.ndarray) -> np.ndarray:
    return 1.0 + POTENTIAL_WEIGHT * np.abs(y_scaled) ** POTENTIAL_WEIGHT_POWER


def load_metadata(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def polar_features(phi: np.ndarray, metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    ny, nx = phi.shape
    cx = float(metadata.get("center_x_cell", (nx - 1) / 2.0))
    cy = float(metadata.get("center_y_cell", (ny - 1) / 2.0))

    col, row = np.meshgrid(np.arange(nx), np.arange(ny))
    x = col - cx
    y = row - cy

    coordinate_scale = float(metadata.get("coordinate_scale", max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y)), 1.0)))
    x_n = x / coordinate_scale
    y_n = y / coordinate_scale
    r = np.sqrt(x_n**2 + y_n**2)
    theta = np.arctan2(y_n, x_n)

    mask = np.isfinite(phi) & (np.abs(phi) > PHI_EPS)
    weights_for_angle = np.where(mask, np.abs(phi), 0.0)
    harmonic = np.sum(weights_for_angle * np.exp(1j * theta))
    theta_spoke = float(np.angle(harmonic))

    theta_shifted = theta - theta_spoke
    theta_shifted = np.arctan2(np.sin(theta_shifted), np.cos(theta_shifted))

    X = np.column_stack(
        [
            r[mask],
            theta_shifted[mask],
            np.sin(theta_shifted[mask]),
            np.cos(theta_shifted[mask]),
        ]
    )
    y_physical = phi[mask]

    grid_features = np.column_stack(
        [
            r.ravel(),
            theta_shifted.ravel(),
            np.sin(theta_shifted.ravel()),
            np.cos(theta_shifted.ravel()),
        ]
    )

    feature_metadata = {
        "center_x_cell": cx,
        "center_y_cell": cy,
        "coordinate_scale": coordinate_scale,
        "theta_spoke": theta_spoke,
        "feature_names": ["r", "theta", "sin_theta", "cos_theta"],
    }
    return X, y_physical, grid_features, mask, feature_metadata


def save_comparison_plot(out_dir: Path, phi: np.ndarray, pred: np.ndarray, residual: np.ndarray, mask: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = np.nanpercentile(np.abs(phi[mask]), 99.5)
    lim = max(float(np.nanpercentile(np.abs(residual[mask]), 99.0)), 1e-12)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    im0 = ax[0].imshow(np.where(mask, phi, np.nan), origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[0].set_title("rotating average potential")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(pred, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax[1].set_title("PySR formula")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(residual, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[2].set_title("prediction - data")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    for axis in ax:
        axis.set_xlabel("x cell")
        axis.set_ylabel("y cell")

    fig.savefig(out_dir / "comparison.png", dpi=180)
    plt.close(fig)


def save_surface_html(out_dir: Path, pred: np.ndarray, mask: np.ndarray) -> None:
    import plotly.graph_objects as go

    ny, nx = pred.shape
    x_plot, y_plot = np.meshgrid(np.arange(nx), np.arange(ny))
    lim = max(float(np.nanpercentile(np.abs(pred[mask]), 99.5)), 1e-12)
    fig = go.Figure(
        data=[
            go.Surface(
                x=x_plot,
                y=y_plot,
                z=np.where(mask, pred, np.nan),
                colorscale="RdBu",
                cmin=-lim,
                cmax=lim,
                colorbar=dict(title="phi formula"),
            )
        ]
    )
    fig.update_layout(
        title="PySR analytic potential formula in polar coordinates",
        scene=dict(
            xaxis_title="x cell",
            yaxis_title="y cell",
            zaxis_title="phi formula",
            aspectratio=dict(x=1, y=1, z=0.45),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(out_dir / "formula_surface.html", include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        from pysr import PySRRegressor
    except ImportError as exc:
        raise SystemExit("Install PySR and joblib in the active Python environment before running this script.") from exc

    average_data = np.load(args.average_file)
    phi = average_data["mean_pattern"].astype(float)
    metadata = load_metadata(Path(args.metadata))

    X_all, y_physical, X_grid, mask, feature_metadata = polar_features(phi, metadata)
    target_scale = float(np.percentile(np.abs(y_physical), 99.0))
    if target_scale <= 0.0:
        target_scale = 1.0
    y_all = y_physical / target_scale

    rng = np.random.default_rng(RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_validation_split(rng, X_all, y_all, VALIDATION_SIZE)
    weights_train = potential_weights(y_train)

    pysr_kwargs = dict(
        niterations=NITERATIONS,
        populations=POPULATIONS,
        maxsize=MAXSIZE,
        parsimony=PARSIMONY,
        model_selection="accuracy",
        binary_operators=BINARY_OPERATORS,
        unary_operators=UNARY_OPERATORS,
        constraints=CONSTRAINTS,
        nested_constraints=NESTED_CONSTRAINTS,
        complexity_of_operators=COMPLEXITY_OF_OPERATORS,
        elementwise_loss="loss(prediction, target, weight) = weight * (prediction - target)^2",
        batching=False,
        random_state=RANDOM_STATE,
        timeout_in_seconds=TIMEOUT_MINUTES * 60,
        progress=True,
        verbosity=1,
        output_directory=str(out_dir / "pysr_outputs"),
    )
    if PROCS and PROCS > 0:
        pysr_kwargs["procs"] = int(PROCS)

    print(f"Loaded rotating potential average: {phi.shape}")
    print(f"Nonzero finite cells: {len(y_all)}")
    print(f"Train points: {len(y_train)}")
    print(f"Validation points: {len(y_val)}")
    print(f"target_scale = {target_scale:.12g}")
    print(f"PySR procs = {pysr_kwargs.get('procs', 'default')}")

    model = PySRRegressor(**pysr_kwargs)
    model.fit(
        X_train,
        y_train,
        weights=weights_train,
        variable_names=feature_metadata["feature_names"],
    )

    pred_train = np.asarray(model.predict(X_train), dtype=float)
    pred_val = np.asarray(model.predict(X_val), dtype=float)
    metrics = {
        **regression_metrics(y_train, pred_train, target_scale, "train"),
        **regression_metrics(y_val, pred_val, target_scale, "val"),
        "target_scale": target_scale,
        "potential_weight": POTENTIAL_WEIGHT,
        "potential_weight_power": POTENTIAL_WEIGHT_POWER,
    }

    best = model.get_best()
    equations = model.equations_.sort_values("loss", ascending=True)
    equations.to_csv(out_dir / "equations.csv", index=False)
    joblib.dump(model, out_dir / "model.pkl")

    pred_grid = np.asarray(model.predict(X_grid), dtype=float).reshape(phi.shape) * target_scale
    pred_grid = np.where(mask, pred_grid, np.nan)
    residual_grid = pred_grid - phi

    metrics_path = out_dir / "metrics.json"
    output_metadata = {
        **metadata,
        **feature_metadata,
        "average_file": str(args.average_file),
        "phi_eps": PHI_EPS,
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "niterations": NITERATIONS,
        "maxsize": MAXSIZE,
        "populations": POPULATIONS,
        "parsimony": PARSIMONY,
        "timeout_minutes": TIMEOUT_MINUTES,
        "procs": PROCS,
        "binary_operators": BINARY_OPERATORS,
        "unary_operators": UNARY_OPERATORS,
        "best_equation": str(best["equation"]),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")

    formula_text = [
        "Best equation for scaled rotating-frame electric potential in polar coordinates:",
        str(best["equation"]),
        "",
        "Physical mapping:",
        f"phi_mean(r, theta) ~= {target_scale:.12g} * F(r, theta, sin(theta), cos(theta))",
        f"x_n = (x_cell - {feature_metadata['center_x_cell']:.12g}) / {feature_metadata['coordinate_scale']:.12g}",
        f"y_n = (y_cell - {feature_metadata['center_y_cell']:.12g}) / {feature_metadata['coordinate_scale']:.12g}",
        "r = sqrt(x_n^2 + y_n^2)",
        f"theta = atan2(y_n, x_n) - ({feature_metadata['theta_spoke']:.12g})",
        "theta is wrapped to [-pi, pi]",
        "",
        "Rotation found before averaging:",
        f"omega = {metadata.get('omega', 'unknown')}",
        f"phi0 = {metadata.get('phi0', 'unknown')}",
        "",
        "Validation metrics:",
        json.dumps(metrics, indent=2),
    ]
    (out_dir / "formula.txt").write_text("\n".join(formula_text), encoding="utf-8")

    save_comparison_plot(out_dir, phi, pred_grid, residual_grid, mask)
    save_surface_html(out_dir, pred_grid, mask)

    print("\n".join(formula_text))
    print(f"\nSaved model to {out_dir / 'model.pkl'}")
    print(f"Saved equations to {out_dir / 'equations.csv'}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved comparison to {out_dir / 'comparison.png'}")
    print(f"Saved 3D surface to {out_dir / 'formula_surface.html'}")


if __name__ == "__main__":
    main()
