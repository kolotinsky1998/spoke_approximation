#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate ion-density maps from rho_i_*.txt files."
    )
    parser.add_argument(
        "--pattern",
        default="rho_i_*.txt",
        help="Glob pattern for input files. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="rho_i_evolution.mp4",
        help="Output animation path (.mp4 or .gif). Default: %(default)s",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second. Default: %(default)s",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI. Default: %(default)s",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap. Default: %(default)s",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the animation window after saving.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Optional circle radius in cell units to draw on top of the map.",
    )
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_frames(paths: list[Path]) -> list[np.ndarray]:
    frames = []
    shape = None
    for path in paths:
        frame = np.loadtxt(path)
        if frame.ndim != 2:
            raise ValueError(f"{path} is not a 2D matrix.")
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise ValueError(
                f"Inconsistent frame shape: {path} has {frame.shape}, expected {shape}."
            )
        frames.append(frame)
    return frames


def compute_norm(frames_abs: list[np.ndarray]) -> LogNorm:
    positive = np.concatenate([frame[frame > 0.0] for frame in frames_abs if np.any(frame > 0.0)])
    if positive.size == 0:
        return LogNorm(vmin=1.0e-12, vmax=1.0)

    vmin = max(np.percentile(positive, 2.0), 1.0e-12)
    vmax = max(np.percentile(positive, 99.7), vmin * 10.0)
    return LogNorm(vmin=vmin, vmax=vmax)


def add_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#0b1020",
            "savefig.facecolor": "#0b1020",
            "axes.edgecolor": "#d0d7ff",
            "axes.labelcolor": "#f4f7ff",
            "text.color": "#f4f7ff",
            "xtick.color": "#d7deff",
            "ytick.color": "#d7deff",
            "font.size": 13,
            "axes.titleweight": "bold",
        }
    )


def main() -> None:
    args = parse_args()
    add_style()

    paths = sorted(Path(".").glob(args.pattern), key=extract_step)
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    steps = [extract_step(path) for path in paths]
    frames = load_frames(paths)
    frames_abs = [np.abs(frame) for frame in frames]
    ny, nx = frames[0].shape
    norm = compute_norm(frames_abs)

    fig, ax = plt.subplots(figsize=(9.5, 8.4), constrained_layout=True)
    image = ax.imshow(
        frames_abs[0],
        origin="lower",
        cmap=args.cmap,
        norm=norm,
        interpolation="bicubic",
    )

    ax.set_title("Ion Density Evolution", fontsize=22, pad=14)
    ax.set_xlabel("x cell index", fontsize=15)
    ax.set_ylabel("z cell index", fontsize=15)
    ax.set_aspect("equal")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if args.radius is not None:
        ax.add_patch(
            Circle(
                ((nx - 1) / 2.0, (ny - 1) / 2.0),
                radius=args.radius,
                fill=False,
                lw=1.5,
                ls="--",
                ec="#8bd3ff",
                alpha=0.8,
            )
        )

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        bbox={"facecolor": "#111933", "edgecolor": "#8bd3ff", "alpha": 0.85, "boxstyle": "round,pad=0.35"},
    )
    stat_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox={"facecolor": "#111933", "edgecolor": "#d8b4fe", "alpha": 0.8, "boxstyle": "round,pad=0.35"},
    )

    cbar = fig.colorbar(image, ax=ax, pad=0.02, shrink=0.94)
    cbar.set_label(r"$|\rho_i|$ (arbitrary units)", fontsize=14)

    def update(frame_idx: int):
        frame_abs = frames_abs[frame_idx]
        image.set_data(frame_abs)
        time_text.set_text(f"iteration = {steps[frame_idx]}")
        stat_text.set_text(
            f"max |rho_i| = {frame_abs.max():.3e}\nmean |rho_i| = {frame_abs.mean():.3e}"
        )
        return image, time_text, stat_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames_abs),
        interval=1000 / max(args.fps, 1),
        blit=False,
        repeat=True,
    )

    output_path = Path(args.output)
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        anim.save(output_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    elif suffix == ".mp4":
        if FFMpegWriter.isAvailable():
            anim.save(
                output_path,
                writer=FFMpegWriter(fps=args.fps, bitrate=2400),
                dpi=args.dpi,
            )
        else:
            fallback_path = output_path.with_suffix(".gif")
            print(
                "ffmpeg is unavailable, saving GIF instead: "
                f"{fallback_path.resolve()}"
            )
            anim.save(fallback_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
            output_path = fallback_path
    else:
        raise SystemExit(
            f"Unsupported output extension: {suffix}. Use .mp4 or .gif."
        )

    print(f"Saved animation to {output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
