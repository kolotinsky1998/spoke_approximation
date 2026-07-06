# Symbolic regression for the rotating ion-density spoke

These scripts are written for Google Colab. They convert the raw
`rho_i_*.txt` matrices into a compact regression dataset, fit an explicit
formula with PySR, and render diagnostic plots.

## 1. Download the GitHub repository in Colab

Run this in a separate Colab cell:

```python
REPO_URL = "https://github.com/kolotinsky1998/spoke_approximation.git"
PROJECT_DIR = "/content/spoke_approximation"

import os

if not os.path.exists(PROJECT_DIR):
    !git clone {REPO_URL} {PROJECT_DIR}
else:
    %cd {PROJECT_DIR}
    !git pull
```

Then:

```python
%cd /content/spoke_approximation
```

If you prefer Google Drive storage for outputs, clone into Drive instead:

```python
from google.colab import drive
drive.mount("/content/drive")

REPO_URL = "https://github.com/kolotinsky1998/spoke_approximation.git"
PROJECT_DIR = "/content/drive/MyDrive/spoke_approximation"

import os

if not os.path.exists(PROJECT_DIR):
    !git clone {REPO_URL} {PROJECT_DIR}
else:
    %cd {PROJECT_DIR}
    !git pull

%cd {PROJECT_DIR}
```

## Alternative: Upload data to Colab

Zip the project folder locally or upload just the `data/` folder. In Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then put the files, for example, in:

```text
/content/drive/MyDrive/spoke_approximation/data/rho_i_0.txt
/content/drive/MyDrive/spoke_approximation/data/rho_i_10000.txt
...
```

Copy this `colab_scripts/` folder into the same project folder.

## 2. Install dependencies

PySR uses Julia under the hood. The first run in Colab can take several
minutes while Julia packages are compiled.

```python
%pip install -U "pysr==1.5.9" numpy pandas scipy scikit-learn matplotlib joblib
```

## 3. Prepare the stationary rotating dataset

```python
!python colab_scripts/prepare_spoke_dataset.py \
  --data-dir data \
  --out-dir colab_outputs/prepared \
  --steady-fraction 0.55 \
  --samples-per-frame 1200 \
  --random-state 7
```

Outputs:

- `spoke_dataset.npz`: sampled points and train/validation split.
- `metadata.json`: grid center, inferred angular velocity, phase, scaling.
- `angle_fit.png`: phase tracking and fitted rotation rate.
- `sampled_points.png`: what points were sampled for regression.

By default coordinates are cell-index coordinates centered on the image.
If you know physical scales, pass `--dx`, `--dy`, and `--dt`.

## 4. Fit PySR

Start with a quick smoke-test run:

```python
!python colab_scripts/fit_spoke_pysr.py \
  --dataset colab_outputs/prepared/spoke_dataset.npz \
  --metadata colab_outputs/prepared/metadata.json \
  --out-dir colab_outputs/pysr_run \
  --niterations 80 \
  --maxsize 25 \
  --populations 8 \
  --operator-set fast \
  --batch-size 1024 \
  --timeout-minutes 12
```

For a longer search:

```python
!python colab_scripts/fit_spoke_pysr.py \
  --dataset colab_outputs/prepared/spoke_dataset.npz \
  --metadata colab_outputs/prepared/metadata.json \
  --out-dir colab_outputs/pysr_run_long \
  --niterations 2500 \
  --maxsize 55 \
  --populations 40 \
  --operator-set balanced \
  --batch-size 2048 \
  --timeout-minutes 90
```

The fitted expression uses normalized features:

```text
r_n      = r / r_scale
cpsi     = cos(theta - omega*t - phase0)
spsi     = sin(theta - omega*t - phase0)
xrot_n   = r_n*cpsi
yrot_n   = r_n*spsi
```

The physical formula is therefore:

```text
rho_i(x, y, t) ~= target_scale * F(r_n, cpsi, spsi, xrot_n, yrot_n)
```

where `omega`, `phase0`, `r_scale`, and `target_scale` are stored in
`metadata.json`.

## 5. Evaluate and visualize

```python
!python colab_scripts/evaluate_spoke_formula.py \
  --model colab_outputs/pysr_run/model.pkl \
  --metadata colab_outputs/prepared/metadata.json \
  --data-dir data \
  --out-dir colab_outputs/evaluation \
  --frames 1200000 1500000 1800000 1990000
```

Outputs:

- `formula.txt`: selected PySR formula plus the physical coordinate mapping.
- `metrics.csv`: frame-by-frame errors.
- `comparison_*.png`: data, formula prediction, residuals.

## Notes

- The animation script visualizes `abs(rho_i)`, so these scripts also use
  `abs(rho_i)` by default. Pass `--signed-target` to fit signed values instead.
- If the spoke is not fully stationary at 55% of the sequence, increase
  `--steady-fraction`, for example to `0.65` or `0.75`.
- If PySR finds formulas that are too complicated, increase `--parsimony` or
  lower `--maxsize`.
- If small fluctuations dominate, raise `--smooth-sigma` in dataset preparation.
