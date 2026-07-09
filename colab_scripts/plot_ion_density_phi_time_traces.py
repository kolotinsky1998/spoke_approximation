#!/usr/bin/env python3
"""Plot ion-density and electric-potential time traces at Figure-4 points."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


# =============================================================================
# Editable calculation parameters
# =============================================================================

RHO_PATTERN = "rho_i_*.txt"
PHI_PATTERN = "phi_*.txt"

CENTER_X = None
CENTER_Y = None
R0_CELLS = None
DOMAIN_EPS = 0.0

POINTS = [
    ("R0_over_2", 0.50, 0.0),
    ("R0_0p99", 0.99, 0.0),
]

DT = 1.0
TIME_OFFSET = 0.0
TIME_LABEL = "step"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho-dir", default="data", help="Directory with rho_i_*.txt files.")
    parser.add_argument("--phi-dir", default="data_phi", help="Directory with phi_*.txt files.")
    parser.add_argument("--out-dir", default="outputs_phi/time_traces", help="Output directory.")
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot extract iteration number from {path.name}")
    return int(match.group(1))


def load_paths_by_step(data_dir: Path, pattern: str) -> dict[int, Path]:
    paths = sorted(data_dir.glob(pattern), key=extract_step)
    if not paths:
        raise FileNotFoundError(f"No files matched {data_dir / pattern}")
    return {extract_step(path): path for path in paths}


def load_frame(path: Path) -> np.ndarray:
    frame = np.loadtxt(path)
    if frame.ndim != 2:
        raise ValueError(f"{path} is not a 2D matrix.")
    return frame


def infer_geometry(rho_frame: np.ndarray) -> tuple[float, float, float]:
    ny, nx = rho_frame.shape
    cx = (nx - 1) / 2.0 if CENTER_X is None else float(CENTER_X)
    cy = (ny - 1) / 2.0 if CENTER_Y is None else float(CENTER_Y)

    if R0_CELLS is not None:
        return cx, cy, float(R0_CELLS)

    col, row = np.meshgrid(np.arange(nx), np.arange(ny))
    x = col - cx
    y = row - cy
    mask = np.abs(rho_frame) > DOMAIN_EPS
    if not np.any(mask):
        raise ValueError("Cannot infer R0: ion-density frame has an empty nonzero domain.")
    r0 = float(max(np.max(np.abs(x[mask])), np.max(np.abs(y[mask])), 1.0))
    return cx, cy, r0


def find_geometry_frame(common_steps: list[int], rho_paths: dict[int, Path], expected_shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    if R0_CELLS is not None:
        return load_frame(rho_paths[common_steps[0]]), int(common_steps[0])

    for step in common_steps:
        frame = load_frame(rho_paths[step])
        if frame.shape != expected_shape:
            raise ValueError(f"Inconsistent rho shape in {rho_paths[step]}.")
        if np.any(np.abs(frame) > DOMAIN_EPS):
            return frame, int(step)

    raise ValueError("Cannot infer R0: all ion-density frames have an empty nonzero domain.")


def sample_bilinear(frame: np.ndarray, x_cell: float, y_cell: float) -> float:
    from scipy.ndimage import map_coordinates

    value = map_coordinates(
        frame,
        [[y_cell], [x_cell]],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    return float(value[0])


def collect_traces(
    rho_paths: dict[int, Path],
    phi_paths: dict[int, Path],
) -> tuple[list[dict], dict]:
    common_steps = sorted(set(rho_paths) & set(phi_paths))
    if not common_steps:
        raise ValueError("No matching iteration numbers between ion-density and potential files.")

    first_rho = load_frame(rho_paths[common_steps[0]])
    first_phi = load_frame(phi_paths[common_steps[0]])
    if first_rho.shape != first_phi.shape:
        raise ValueError(f"Shape mismatch: rho {first_rho.shape}, phi {first_phi.shape}.")

    geometry_rho, geometry_step = find_geometry_frame(common_steps, rho_paths, first_rho.shape)
    cx, cy, r0 = infer_geometry(geometry_rho)
    sample_points = []
    for name, x_over_r0, y_over_r0 in POINTS:
        sample_points.append(
            {
                "name": name,
                "x_over_r0": float(x_over_r0),
                "y_over_r0": float(y_over_r0),
                "x_cell": float(cx + x_over_r0 * r0),
                "y_cell": float(cy + y_over_r0 * r0),
            }
        )

    rows = []
    for step in common_steps:
        rho = load_frame(rho_paths[step])
        phi = load_frame(phi_paths[step])
        if rho.shape != first_rho.shape:
            raise ValueError(f"Inconsistent rho shape in {rho_paths[step]}.")
        if phi.shape != first_phi.shape:
            raise ValueError(f"Inconsistent phi shape in {phi_paths[step]}.")

        row = {
            "step": step,
            "time": TIME_OFFSET + DT * step,
        }
        for point in sample_points:
            name = point["name"]
            x_cell = point["x_cell"]
            y_cell = point["y_cell"]
            row[f"rho_i_{name}"] = sample_bilinear(rho, x_cell, y_cell)
            row[f"phi_{name}"] = sample_bilinear(phi, x_cell, y_cell)
        rows.append(row)

    metadata = {
        "center_x_cell": cx,
        "center_y_cell": cy,
        "r0_cells": r0,
        "points": sample_points,
        "dt": DT,
        "time_offset": TIME_OFFSET,
        "time_label": TIME_LABEL,
        "n_steps": len(common_steps),
        "first_step": int(common_steps[0]),
        "last_step": int(common_steps[-1]),
        "geometry_step": geometry_step,
    }
    return rows, metadata


def save_csv(out_dir: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with (out_dir / "ion_density_phi_time_traces.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(out_dir: Path, rows: list[dict], metadata: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.array([row["time"] for row in rows], dtype=float)
    fig, axes = plt.subplots(len(POINTS), 2, figsize=(13, 3.8 * len(POINTS)), constrained_layout=True)
    if len(POINTS) == 1:
        axes = np.array([axes])

    for i, point in enumerate(metadata["points"]):
        name = point["name"]
        label = f"({point['x_over_r0']:.2g} R0, {point['y_over_r0']:.2g} R0)"
        rho_values = np.array([row[f"rho_i_{name}"] for row in rows], dtype=float)
        phi_values = np.array([row[f"phi_{name}"] for row in rows], dtype=float)

        axes[i, 0].plot(time, rho_values, lw=1.5, color="tab:orange", label="ion density")
        axes[i, 0].set_title(f"Ion density at {label}")
        axes[i, 0].set_xlabel(TIME_LABEL)
        axes[i, 0].set_ylabel("rho_i")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend()

        axes[i, 1].plot(time, phi_values, lw=1.5, color="tab:blue", label="electric potential")
        axes[i, 1].set_title(f"Electric potential at {label}")
        axes[i, 1].set_xlabel(TIME_LABEL)
        axes[i, 1].set_ylabel("phi")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()

    fig.savefig(out_dir / "ion_density_phi_time_traces.png", dpi=180)
    plt.close(fig)


def save_interactive_html(out_dir: Path, rows: list[dict], metadata: dict) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    time = np.array([row["time"] for row in rows], dtype=float)
    fig = make_subplots(
        rows=len(POINTS),
        cols=2,
        subplot_titles=[
            title
            for point in metadata["points"]
            for title in (
                f"rho_i at ({point['x_over_r0']:.2g} R0, {point['y_over_r0']:.2g} R0)",
                f"phi at ({point['x_over_r0']:.2g} R0, {point['y_over_r0']:.2g} R0)",
            )
        ],
    )

    for i, point in enumerate(metadata["points"], start=1):
        name = point["name"]
        rho_values = np.array([row[f"rho_i_{name}"] for row in rows], dtype=float)
        phi_values = np.array([row[f"phi_{name}"] for row in rows], dtype=float)
        fig.add_trace(go.Scatter(x=time, y=rho_values, mode="lines", name=f"rho_i {name}"), row=i, col=1)
        fig.add_trace(go.Scatter(x=time, y=phi_values, mode="lines", name=f"phi {name}"), row=i, col=2)

    fig.update_xaxes(title_text=TIME_LABEL)
    fig.update_yaxes(title_text="rho_i", col=1)
    fig.update_yaxes(title_text="phi", col=2)
    fig.update_layout(
        title="Ion-density and electric-potential time traces at Figure-4 points",
        height=max(450, 330 * len(POINTS)),
        margin=dict(l=60, r=30, t=80, b=40),
    )
    fig.write_html(out_dir / "ion_density_phi_time_traces.html", include_plotlyjs="cdn")


def save_metadata(out_dir: Path, metadata: dict) -> None:
    import json

    (out_dir / "time_trace_points_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rho_paths = load_paths_by_step(Path(args.rho_dir), RHO_PATTERN)
    phi_paths = load_paths_by_step(Path(args.phi_dir), PHI_PATTERN)
    rows, metadata = collect_traces(rho_paths, phi_paths)

    save_csv(out_dir, rows)
    save_metadata(out_dir, metadata)
    save_plot(out_dir, rows, metadata)
    save_interactive_html(out_dir, rows, metadata)

    print(f"Saved CSV to {out_dir / 'ion_density_phi_time_traces.csv'}")
    print(f"Saved metadata to {out_dir / 'time_trace_points_metadata.json'}")
    print(f"Saved plot to {out_dir / 'ion_density_phi_time_traces.png'}")
    print(f"Saved interactive plot to {out_dir / 'ion_density_phi_time_traces.html'}")
    print(f"Points sampled with center=({metadata['center_x_cell']:.6g}, {metadata['center_y_cell']:.6g})")
    print(f"R0 = {metadata['r0_cells']:.6g} cells")
    print(f"R0 inferred from step = {metadata['geometry_step']}")


if __name__ == "__main__":
    main()
