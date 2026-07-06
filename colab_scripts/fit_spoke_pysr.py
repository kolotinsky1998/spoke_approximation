#!/usr/bin/env python3
"""Fit stationary rotating rho_i(x, y, t) with a PySR template expression."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


BINARY_OPERATORS = ["+", "-", "*", "/"]

# sin/cos are needed by the fixed time-rotation template. abs is deliberately
# excluded because it created ray-like piecewise artifacts in last-frame tests.
UNARY_OPERATORS = ["sqrt", "exp", "sin", "cos"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="colab_outputs/prepared/dataset.npz")
    parser.add_argument("--metadata", default="colab_outputs/prepared/metadata.json")
    parser.add_argument("--out-dir", default="colab_outputs/pysr_run")
    parser.add_argument("--niterations", type=int, default=800)
    parser.add_argument("--populations", type=int, default=24)
    parser.add_argument("--maxsize", type=int, default=80)
    parser.add_argument("--parsimony", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--timeout-minutes", type=float, default=None)
    parser.add_argument("--density-weight", type=float, default=20.0, help="Extra loss weight for high-density points. Use 0 for plain MSE.")
    parser.add_argument("--density-weight-power", type=float, default=2.0)
    parser.add_argument("--denoise", action="store_true", help="Use PySR Gaussian-process denoising.")
    parser.add_argument("--tensorboard-log-dir", default=None, help="Optional TensorBoard log directory.")
    parser.add_argument("--tensorboard-log-interval", type=int, default=10)
    parser.add_argument("--top-equations", type=int, default=10)
    parser.add_argument("--show-pysr-output", action="store_true", help="Show raw PySR/Julia output instead of writing it to a log file.")
    parser.add_argument("--turbo", action="store_true", help="Enable PySR turbo mode. Faster, but may print Julia LoopVectorization warnings.")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--procs", type=int, default=0, help="Julia worker processes. 0 lets PySR choose.")
    return parser.parse_args()


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


def make_weights(y_scaled: np.ndarray, density_weight: float, power: float) -> np.ndarray | None:
    if density_weight <= 0.0:
        return None
    signal = np.clip(y_scaled, 0.0, None)
    return 1.0 + density_weight * signal**power


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


def log_tail(path: Path, n_lines: int = 60) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def safe_text(fn) -> str:
    try:
        return str(fn())
    except Exception as exc:  # PySR exports can fail for some template expressions.
        return f"<unavailable: {exc}>"


def compact_value(value) -> str:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return f"{float(value):.12g}"
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=8, threshold=20)
    text = str(value)
    if len(text) > 500:
        return text[:500] + "..."
    return text


def parameter_lines(best_row) -> list[str]:
    lines = []
    for key in best_row.index:
        key_lower = str(key).lower()
        if key_lower in {"omega", "phi", "parameters", "parameter_values"} or "parameter" in key_lower:
            lines.append(f"{key}: {compact_value(best_row[key])}")
    return lines


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        from pysr import PySRRegressor, TemplateExpressionSpec
        from pysr import TensorBoardLoggerSpec
    except ImportError as exc:
        raise SystemExit(
            "PySR is not installed. In Colab run:\n"
            '  %pip install -U "pysr==1.5.9" numpy pandas matplotlib joblib tensorboard'
        ) from exc

    data = np.load(args.dataset, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    target_scale = float(metadata["target_scale"])

    if X_train.shape[1] != 3:
        raise ValueError("Expected dataset features [x_n, y_n, t_n]. Re-run prepare_spoke_dataset.py.")

    weights_train = make_weights(y_train, args.density_weight, args.density_weight_power)
    n_points_total = len(y_train) + len(y_val)
    if args.denoise and n_points_total > 10000:
        print(
            "Warning: --denoise fits a Gaussian process before symbolic regression. "
            "For large datasets this can be slow; consider lowering --samples-per-frame "
            "in prepare_spoke_dataset.py if Colab stalls."
        )

    # tau is normalized time: tau = (t_phys - t0) / t_scale.
    # PySR optimizes omega and phi directly in the template below.
    angle = "omega[1] * tau + phi[1]"
    combine = (
        "f("
        f"x*cos({angle}) + y*sin({angle}), "
        f"-x*sin({angle}) + y*cos({angle})"
        ")"
    )
    expression_spec = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x", "y", "tau"],
        parameters={"omega": 1, "phi": 1},
        combine=combine,
    )

    logger_spec = None
    if args.tensorboard_log_dir:
        logger_spec = TensorBoardLoggerSpec(
            log_dir=args.tensorboard_log_dir,
            log_interval=args.tensorboard_log_interval,
            overwrite=False,
        )

    model_kwargs = dict(
        expression_spec=expression_spec,
        niterations=args.niterations,
        populations=args.populations,
        maxsize=args.maxsize,
        parsimony=args.parsimony,
        model_selection="accuracy",
        binary_operators=BINARY_OPERATORS,
        unary_operators=UNARY_OPERATORS,
        constraints={
            "sqrt": 9,
            "exp": 9,
            "sin": 9,
            "cos": 9,
            "/": (-1, 9),
        },
        nested_constraints={
            "exp": {"exp": 0},
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
        },
        complexity_of_operators={
            "exp": 3,
            "sqrt": 2,
            "sin": 3,
            "cos": 3,
        },
        elementwise_loss=(
            "loss(prediction, target, weight) = weight * (prediction - target)^2"
            if weights_train is not None
            else "loss(prediction, target) = (prediction - target)^2"
        ),
        denoise=args.denoise,
        random_state=args.random_state,
        turbo=args.turbo,
        progress=False,
        verbosity=0,
        input_stream="devnull",
        output_directory=str(out_dir / "pysr_outputs"),
        logger_spec=logger_spec,
    )
    if args.procs > 0:
        model_kwargs["procs"] = args.procs
    if args.timeout_minutes is not None:
        model_kwargs["timeout_in_seconds"] = args.timeout_minutes * 60.0
    if args.batch_size is not None and args.batch_size > 0:
        model_kwargs["batching"] = True
        model_kwargs["batch_size"] = args.batch_size

    log_path = out_dir / "pysr.log"
    print("Fitting rotating-template PySR model")
    print(f"Train points: {len(y_train)}, validation points: {len(y_val)}")
    print(f"Template: rho_scaled = {combine}")
    print(f"Operators: binary={BINARY_OPERATORS}, unary={UNARY_OPERATORS}")
    print(f"Density weighting: {args.density_weight:g} * target^{args.density_weight_power:g}")
    print(f"Denoise: {args.denoise}")
    if logger_spec is not None:
        print(f"TensorBoard log dir: {args.tensorboard_log_dir}")
    print(f"PySR output is written to: {log_path}")

    start = time.monotonic()
    try:
        with redirect_process_output(log_path, enabled=not args.show_pysr_output):
            model = PySRRegressor(**model_kwargs)
            if weights_train is None:
                model.fit(X_train, y_train, variable_names=["x", "y", "tau"])
            else:
                model.fit(X_train, y_train, variable_names=["x", "y", "tau"], weights=weights_train)
    except Exception as exc:
        print(f"PySR failed: {exc}", flush=True)
        tail = log_tail(log_path)
        if tail:
            print(f"Last PySR log lines from {log_path}:\n{tail}", flush=True)
        raise
    elapsed_min = (time.monotonic() - start) / 60.0
    print(f"PySR finished in {elapsed_min:.2f} min")

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    metrics = {
        **regression_metrics(y_train, pred_train, target_scale, "train"),
        **regression_metrics(y_val, pred_val, target_scale, "val"),
        "target_scale": target_scale,
    }
    if weights_train is not None:
        val_weights = make_weights(y_val, args.density_weight, args.density_weight_power)
        metrics["train_weighted_mse_scaled"] = float(np.average((pred_train - y_train) ** 2, weights=weights_train))
        metrics["val_weighted_mse_scaled"] = float(np.average((pred_val - y_val) ** 2, weights=val_weights))

    best_row = model.get_best()
    equations = model.equations_.sort_values("loss", ascending=True)
    equations.to_csv(out_dir / "equations.csv", index=False)
    top_lines = [
        f"complexity={row['complexity']} loss={row['loss']:.6g} equation={row['equation']}"
        for _, row in equations.head(args.top_equations).iterrows()
    ]

    joblib.dump(model, out_dir / "model.pkl")
    output_metadata = {
        **metadata,
        "pysr_feature_names": ["x", "y", "tau"],
        "template_combine": combine,
        "template_parameters": {"omega": 1, "phi": 1},
        "time_coordinate": "tau = (t_phys - t0) / t_scale",
        "physical_omega_note": "If template omega is Omega_tau, then physical angular speed is Omega_tau / t_scale.",
        "density_weight": args.density_weight,
        "density_weight_power": args.density_weight_power,
        "denoise": args.denoise,
        "tensorboard_log_dir": args.tensorboard_log_dir,
        "best_equation": compact_value(best_row["equation"]),
        "best_parameter_columns": {
            str(key): compact_value(best_row[key])
            for key in best_row.index
            if "parameter" in str(key).lower() or str(key).lower() in {"omega", "phi"}
        },
        "best_metrics": metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")

    formula_text = [
        "Selected PySR rotating-template formula for scaled density:",
        str(best_row["equation"]),
        "",
        "SymPy export:",
        safe_text(model.sympy),
        "",
        "Physical mapping:",
        f"rho(x,y,t) ~= {target_scale:.12g} * template(x_n, y_n, tau)",
        "tau = (t_phys - t0) / t_scale",
        f"t0 = {metadata['t0']:.12g}",
        f"t_scale = {metadata['t_scale']:.12g}",
        f"x_n = x / {metadata['coordinate_scale']:.12g}",
        f"y_n = y / {metadata['coordinate_scale']:.12g}",
        "template(x,y,tau) = F(x*cos(omega*tau + phi) + y*sin(omega*tau + phi), "
        "-x*sin(omega*tau + phi) + y*cos(omega*tau + phi))",
        "",
        "Top equations by loss:",
        *top_lines,
        "",
        "Validation metrics:",
        json.dumps(metrics, indent=2),
    ]
    learned_parameters = parameter_lines(best_row)
    if learned_parameters:
        insert_at = formula_text.index("Physical mapping:")
        formula_text[insert_at:insert_at] = ["Learned template parameters:", *learned_parameters, ""]
    (out_dir / "formula.txt").write_text("\n".join(formula_text), encoding="utf-8")

    print("\n".join(formula_text))
    print(f"\nSaved model to {out_dir / 'model.pkl'}")
    print(f"Saved metadata to {out_dir / 'metadata.json'}")
    print(f"Saved equations to {out_dir / 'equations.csv'}")


if __name__ == "__main__":
    main()
