# Cluster pipeline for spoke approximation

This workflow has two stages:

1. Build the rotating-frame average density and estimate the spoke angular
   velocity.
2. Fit an analytic PySR formula for the averaged density in polar coordinates.

The scripts are intentionally simple for cluster runs. Command-line arguments
are used only for paths. Calculation parameters are constants near the top of
each Python file.

## 1. Rotating-frame average

Edit the calculation constants at the top of:

```text
colab_scripts/prepare_spoke_dataset.py
```

Then run:

```bash
python colab_scripts/prepare_spoke_dataset.py \
  --data-dir data \
  --out-dir outputs/rotating_average
```

Outputs:

- `outputs/rotating_average/rotating_average.npz`
- `outputs/rotating_average/metadata.json`
- `outputs/rotating_average/phase_fit.png`
- `outputs/rotating_average/rotating_average_2d.png`

The script estimates the spoke phase using the first azimuthal harmonic after
subtracting a radial background, fits

```text
phase(t) ~= omega*t + phi0
```

and averages all stationary frames in the co-rotating frame.

## 2. PySR fit in polar coordinates

Edit the calculation constants at the top of:

```text
colab_scripts/fit_spoke_pysr.py
```

Then run:

```bash
python colab_scripts/fit_spoke_pysr.py \
  --average-file outputs/rotating_average/rotating_average.npz \
  --metadata outputs/rotating_average/metadata.json \
  --out-dir outputs/pysr_polar
```

Outputs:

- `outputs/pysr_polar/model.pkl`
- `outputs/pysr_polar/equations.csv`
- `outputs/pysr_polar/formula.txt`
- `outputs/pysr_polar/metrics.json`
- `outputs/pysr_polar/comparison.png`
- `outputs/pysr_polar/formula_surface.html`

The fitted model uses nonzero finite density cells only and searches

```text
rho_mean(r, theta) ~= target_scale * F(r, theta, sin(theta), cos(theta))
```

where

```text
r = sqrt(x_n^2 + y_n^2)
theta = atan2(y_n, x_n) - theta_spoke
```

The learned averaged formula can be lifted back to the lab frame with the
rotation parameters saved in `metadata.json`.

## Notes

- The direct time-dependent `TemplateExpressionSpec` approach is no longer the
  main pipeline.
- PySR parallelism is controlled by the `PROCS` constant in
  `fit_spoke_pysr.py`.
- Batching is disabled by default because the averaged 2D profile contains only
  a few thousand nonzero cells.
