#!/usr/bin/env python3
"""Fit an explicit stationary-spoke formula with PySR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="colab_outputs/prepared/spoke_dataset.npz")
    parser.add_argument("--metadata", default="colab_outputs/prepared/metadata.json")
    parser.add_argument("--out-dir", default="colab_outputs/pysr_run")
    parser.add_argument("--niterations", type=int, default=800)
    parser.add_argument("--populations", type=int, default=24)
    parser.add_argument("--maxsize", type=int, default=45)
    parser.add_argument("--parsimony", type=float, default=0.002)
    parser.add_argument("--timeout-minutes", type=float, default=None, help="Stop PySR after this many minutes.")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--procs", type=int, default=0, help="Julia worker processes. 0 lets PySR choose.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        raise SystemExit(
            "PySR is not installed. In Colab run:\n"
            '  pip install -U "pysr==1.5.9" numpy pandas scipy scikit-learn matplotlib joblib'
        ) from exc

    data = np.load(args.dataset, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    feature_names = [str(x) for x in data["feature_names"]]
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))

    model_kwargs = dict(
        niterations=args.niterations,
        populations=args.populations,
        maxsize=args.maxsize,
        parsimony=args.parsimony,
        model_selection="best",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sin",
            "cos",
            "exp",
            "sqrt",
            "abs",
        ],
        constraints={
            "exp": 9,
            "sin": 9,
            "cos": 9,
            "sqrt": 9,
            "/": (-1, 9),
        },
        nested_constraints={
            "sin": {"sin": 0, "cos": 0, "exp": 0},
            "cos": {"sin": 0, "cos": 0, "exp": 0},
            "exp": {"exp": 0},
        },
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=args.random_state,
        turbo=True,
        output_directory=str(out_dir / "pysr_outputs"),
        verbosity=1,
    )
    if args.procs > 0:
        model_kwargs["procs"] = args.procs
    if args.timeout_minutes is not None:
        model_kwargs["timeout_in_seconds"] = args.timeout_minutes * 60.0

    model = PySRRegressor(**model_kwargs)
    model.fit(X_train, y_train, variable_names=feature_names)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    metrics = {
        "train_mae_scaled": float(mean_absolute_error(y_train, pred_train)),
        "train_rmse_scaled": float(mean_squared_error(y_train, pred_train, squared=False)),
        "train_r2": float(r2_score(y_train, pred_train)),
        "val_mae_scaled": float(mean_absolute_error(y_val, pred_val)),
        "val_rmse_scaled": float(mean_squared_error(y_val, pred_val, squared=False)),
        "val_r2": float(r2_score(y_val, pred_val)),
        "target_scale": metadata["target_scale"],
        "val_mae_physical": float(mean_absolute_error(y_val, pred_val) * metadata["target_scale"]),
        "val_rmse_physical": float(mean_squared_error(y_val, pred_val, squared=False) * metadata["target_scale"]),
    }

    joblib.dump(model, out_dir / "model.pkl")
    model.equations_.to_csv(out_dir / "equations.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    best = model.get_best()
    formula_text = [
        "Selected PySR formula for scaled density:",
        str(best["equation"]),
        "",
        "Physical mapping:",
        f"rho_i ~= {metadata['target_scale']:.12g} * F(r_n, cpsi, spsi, xrot_n, yrot_n)",
        f"r_n = r / {metadata['r_scale']:.12g}",
        f"psi = theta - ({metadata['omega']:.12g})*t - ({metadata['phase0']:.12g})",
        "cpsi = cos(psi)",
        "spsi = sin(psi)",
        "xrot_n = r_n*cpsi",
        "yrot_n = r_n*spsi",
        "",
        "Validation metrics:",
        json.dumps(metrics, indent=2),
    ]
    (out_dir / "formula.txt").write_text("\n".join(formula_text), encoding="utf-8")

    print("\n".join(formula_text))
    print(f"\nSaved model to {out_dir / 'model.pkl'}")
    print(f"Saved equation table to {out_dir / 'equations.csv'}")


if __name__ == "__main__":
    main()
