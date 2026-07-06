#!/usr/bin/env python3
"""Fit rho_i(x, y, t) as F(rotating coordinates) with PySR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="colab_outputs/prepared/dataset.npz")
    parser.add_argument("--metadata", default="colab_outputs/prepared/metadata.json")
    parser.add_argument("--out-dir", default="colab_outputs/pysr_run")
    parser.add_argument("--omega-min", type=float, required=True, help="Minimum omega in rad per physical time unit.")
    parser.add_argument("--omega-max", type=float, required=True, help="Maximum omega in rad per physical time unit.")
    parser.add_argument("--omega-count", type=int, default=17, help="Number of omega values in the grid search.")
    parser.add_argument("--niterations", type=int, default=80)
    parser.add_argument("--populations", type=int, default=8)
    parser.add_argument("--maxsize", type=int, default=30)
    parser.add_argument("--parsimony", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=None, help="Use PySR mini-batches of this size.")
    parser.add_argument("--timeout-minutes", type=float, default=None, help="Approximate total PySR time budget across the omega scan.")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--procs", type=int, default=0, help="Julia worker processes. 0 lets PySR choose.")
    return parser.parse_args()


def rotating_features(X: np.ndarray, omega: float, metadata: dict) -> np.ndarray:
    x_n = X[:, 0]
    y_n = X[:, 1]
    t = X[:, 2] * metadata["t_scale"] + metadata["t0"]
    alpha = omega * t
    c = np.cos(alpha)
    s = np.sin(alpha)
    u_n = x_n * c + y_n * s
    v_n = -x_n * s + y_n * c
    return np.column_stack([u_n, v_n])


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, target_scale: float, prefix: str) -> dict:
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        raise SystemExit(
            "PySR is not installed. In Colab run:\n"
            '  %pip install -U "pysr==1.5.9" numpy pandas scipy scikit-learn matplotlib joblib'
        ) from exc

    data = np.load(args.dataset, allow_pickle=True)
    X_train_raw = data["X_train"]
    y_train = data["y_train"]
    X_val_raw = data["X_val"]
    y_val = data["y_val"]
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    target_scale = float(metadata["target_scale"])

    if args.omega_count < 1:
        raise ValueError("--omega-count must be >= 1")
    omega_values = np.linspace(args.omega_min, args.omega_max, args.omega_count)

    model_kwargs = dict(
        niterations=args.niterations,
        populations=args.populations,
        maxsize=args.maxsize,
        parsimony=args.parsimony,
        model_selection="best",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "abs", "exp"],
        constraints={
            "sqrt": 9,
            "exp": 9,
            "/": (-1, 9),
        },
        nested_constraints={
            "exp": {"exp": 0},
        },
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=args.random_state,
        turbo=True,
        verbosity=1,
    )
    if args.procs > 0:
        model_kwargs["procs"] = args.procs
    if args.timeout_minutes is not None:
        model_kwargs["timeout_in_seconds"] = args.timeout_minutes * 60.0 / args.omega_count
    if args.batch_size is not None:
        model_kwargs["batching"] = True
        model_kwargs["batch_size"] = args.batch_size

    runs = []
    best_record = None
    for run_idx, omega in enumerate(omega_values):
        run_dir = out_dir / "pysr_outputs" / f"omega_{run_idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        X_train = rotating_features(X_train_raw, float(omega), metadata)
        X_val = rotating_features(X_val_raw, float(omega), metadata)

        model = PySRRegressor(output_directory=str(run_dir), **model_kwargs)
        model.fit(X_train, y_train, variable_names=["u_n", "v_n"])
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        train_metrics = metrics_dict(y_train, pred_train, target_scale, "train")
        val_metrics = metrics_dict(y_val, pred_val, target_scale, "val")
        record = {
            "omega": float(omega),
            "equation": str(model.get_best()["equation"]),
            **train_metrics,
            **val_metrics,
        }
        runs.append(record)
        if best_record is None or record["val_rmse_scaled"] < best_record["metrics"]["val_rmse_scaled"]:
            best_record = {"omega": float(omega), "model": model, "metrics": record}

        print(
            f"omega={omega:.12g} "
            f"val_rmse={record['val_rmse_physical']:.6g} "
            f"val_r2={record['val_r2']:.4f}"
        )

    if best_record is None:
        raise RuntimeError("No PySR runs completed.")

    best_model = best_record["model"]
    best_omega = best_record["omega"]
    best_metrics = best_record["metrics"]
    joblib.dump(best_model, out_dir / "model.pkl")
    best_model.equations_.to_csv(out_dir / "equations.csv", index=False)
    (out_dir / "omega_scan_metrics.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

    output_metadata = {
        **metadata,
        "omega": best_omega,
        "omega_min": args.omega_min,
        "omega_max": args.omega_max,
        "omega_count": args.omega_count,
        "pysr_feature_names": ["u_n", "v_n"],
        "pysr_formula_mapping": (
            "rho(x,y,t) ~= target_scale * F("
            "x_n*cos(omega*t) + y_n*sin(omega*t), "
            "-x_n*sin(omega*t) + y_n*cos(omega*t))"
        ),
        "best_metrics": best_metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")

    formula_text = [
        "Selected PySR formula for scaled density:",
        str(best_model.get_best()["equation"]),
        "",
        "Physical mapping:",
        f"rho(x,y,t) ~= {target_scale:.12g} * F(u_n, v_n)",
        f"u_n = x_n*cos(({best_omega:.12g})*t) + y_n*sin(({best_omega:.12g})*t)",
        f"v_n = -x_n*sin(({best_omega:.12g})*t) + y_n*cos(({best_omega:.12g})*t)",
        f"x_n = x / {metadata['coordinate_scale']:.12g}",
        f"y_n = y / {metadata['coordinate_scale']:.12g}",
        "",
        "Best omega and validation metrics:",
        json.dumps(best_metrics, indent=2),
    ]
    (out_dir / "formula.txt").write_text("\n".join(formula_text), encoding="utf-8")

    print("\n".join(formula_text))
    print(f"\nSaved model to {out_dir / 'model.pkl'}")
    print(f"Saved metadata to {out_dir / 'metadata.json'}")
    print(f"Saved omega scan metrics to {out_dir / 'omega_scan_metrics.json'}")


if __name__ == "__main__":
    main()
