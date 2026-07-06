# Direct PySR fit for the rotating ion-density spoke

This Colab workflow keeps preprocessing minimal. It does not smooth,
interpolate, estimate the spoke phase, or transform the data to a rotating
frame during preparation.

The dataset preparation step only builds a compact table:

```text
x_n, y_n, t_n -> rho_i
```

The rotating-pattern assumption is imposed during fitting. For each candidate
angular velocity `omega`, the fit script constructs:

```text
alpha(t) = omega*t

u_n =  x_n*cos(alpha) + y_n*sin(alpha)
v_n = -x_n*sin(alpha) + y_n*cos(alpha)

rho_i(x, y, t) ~= target_scale * F(u_n, v_n)
```

The best `omega` is selected by validation error.

## 1. Clone the repository in Colab

```python
REPO_URL = "https://github.com/kolotinsky1998/spoke_approximation.git"
PROJECT_DIR = "/content/spoke_approximation"

import os

if not os.path.exists(PROJECT_DIR):
    !git clone {REPO_URL} {PROJECT_DIR}
else:
    %cd {PROJECT_DIR}
    !git pull

%cd {PROJECT_DIR}
```

## 2. Install dependencies

The first PySR run compiles Julia packages and may take several minutes.

```python
%pip install -U "pysr==1.5.9" numpy pandas matplotlib joblib
```

## 3. Prepare the dataset

```python
!python colab_scripts/prepare_spoke_dataset.py \
  --data-dir data \
  --out-dir colab_outputs/prepared \
  --steady-fraction 0.55 \
  --samples-per-frame 300 \
  --domain-eps 0.0 \
  --random-state 7
```

Outputs:

- `colab_outputs/prepared/dataset.npz`
- `colab_outputs/prepared/metadata.json`

The preparation script removes points outside the nonzero plasma domain and
samples points from the stationary interval only.

## 4. Fit PySR with omega scan

Start with a short run:

```python
!python colab_scripts/fit_spoke_pysr.py \
  --dataset colab_outputs/prepared/dataset.npz \
  --metadata colab_outputs/prepared/metadata.json \
  --out-dir colab_outputs/pysr_run \
  --omega-min -0.0004 \
  --omega-max 0.0004 \
  --omega-count 17 \
  --niterations 80 \
  --maxsize 30 \
  --populations 8 \
  --parsimony 0.003 \
  --batch-size 512 \
  --timeout-minutes 20
```

`--timeout-minutes` is an approximate total budget for the whole omega scan;
the script divides it across the requested `--omega-count` values.

By default, raw PySR/Julia progress bars are written to log files instead of
Colab output. The notebook output only shows one compact progress line per
omega value. Add `--show-pysr-output` only when debugging PySR itself.

Outputs:

- `colab_outputs/pysr_run/model.pkl`
- `colab_outputs/pysr_run/metadata.json`
- `colab_outputs/pysr_run/formula.txt`
- `colab_outputs/pysr_run/equations.csv`
- `colab_outputs/pysr_run/omega_scan_metrics.json`
- `colab_outputs/pysr_run/pysr_logs/omega_*.log`

The PySR operator set is fixed:

```python
binary_operators = ["+", "-", "*", "/"]
unary_operators = ["sqrt", "abs", "exp"]
```

## 5. Evaluate on full frames

```python
!python colab_scripts/evaluate_spoke_formula.py \
  --model colab_outputs/pysr_run/model.pkl \
  --metadata colab_outputs/pysr_run/metadata.json \
  --data-dir data \
  --out-dir colab_outputs/evaluation
```

Display results:

```python
from IPython.display import Image, display
import glob

print(open("colab_outputs/pysr_run/formula.txt").read())

for path in sorted(glob.glob("colab_outputs/evaluation/comparison_*.png")):
    display(Image(path))
```

## Optional: fit only the last frame

This diagnostic checks whether PySR can approximate a single stationary frame
without any time dependence:

```python
!python colab_scripts/fit_last_frame_pysr.py \
  --data-dir data \
  --out-dir colab_outputs/pysr_last_frame \
  --n-samples 3000 \
  --niterations 200 \
  --maxsize 40 \
  --populations 12 \
  --parsimony 0.001 \
  --batch-size 512 \
  --timeout-minutes 15
```

Display:

```python
from IPython.display import Image, display

print(open("colab_outputs/pysr_last_frame/formula.txt").read())
display(Image("colab_outputs/pysr_last_frame/comparison_last_frame.png"))
```

## Notes

- No smoothing is applied.
- No interpolation is applied.
- No phase tracking is performed during preparation.
- `phi0` is omitted in the first version because a constant phase shift is
  absorbed by the learned spatial function `F(u_n, v_n)`.
