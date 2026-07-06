# Direct PySR fit for the rotating ion-density spoke

This Colab workflow keeps preprocessing minimal. It does not smooth,
interpolate, estimate the spoke phase, or transform the data to a rotating
frame during preparation.

The dataset preparation step only builds a compact table:

```text
x_n, y_n, t_n -> rho_i
```

The rotating-pattern assumption is imposed during fitting with
`TemplateExpressionSpec`. PySR optimizes the angular phase parameters inside
the search rather than scanning `omega` on a fixed grid:

```text
tau = (t - t0) / t_scale
alpha(t) = omega*tau + phi

u_n =  x_n*cos(alpha) + y_n*sin(alpha)
v_n = -x_n*sin(alpha) + y_n*cos(alpha)

rho_i(x, y, t) ~= target_scale * F(u_n, v_n)
```

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
%pip install -U "pysr==1.5.9" numpy pandas matplotlib joblib tensorboard
```

## 3. Prepare the dataset

```python
!python colab_scripts/prepare_spoke_dataset.py \
  --data-dir data \
  --out-dir colab_outputs/prepared \
  --steady-fraction 0.55 \
  --samples-per-frame 120 \
  --domain-eps 0.0 \
  --random-state 7
```

Outputs:

- `colab_outputs/prepared/dataset.npz`
- `colab_outputs/prepared/metadata.json`

The preparation script removes points outside the nonzero plasma domain and
samples points from the stationary interval only.

If you run without `--denoise`, you can raise `--samples-per-frame` to 300-1200.
With `--denoise`, start smaller because PySR first builds a Gaussian-process
denoising model.

## 4. Fit PySR with a rotating template

Start with a short run:

```python
!python colab_scripts/fit_spoke_pysr.py \
  --dataset colab_outputs/prepared/dataset.npz \
  --metadata colab_outputs/prepared/metadata.json \
  --out-dir colab_outputs/pysr_run \
  --niterations 800 \
  --maxsize 80 \
  --populations 24 \
  --parsimony 0.0001 \
  --density-weight 20 \
  --density-weight-power 2 \
  --denoise \
  --batch-size 512 \
  --timeout-minutes 45 \
  --tensorboard-log-dir colab_outputs/tensorboard/spoke_template \
  --tensorboard-log-interval 10
```

The script uses a structured expression:

```text
template(x,y,tau) = F(
    x*cos(omega*tau + phi) + y*sin(omega*tau + phi),
   -x*sin(omega*tau + phi) + y*cos(omega*tau + phi)
)
```

`omega` and `phi` are optimized PySR parameters. Since `tau` is normalized,
the physical angular speed is the learned `omega / t_scale`.

By default, raw PySR/Julia progress bars are written to log files instead of
Colab output. Add `--show-pysr-output` only when debugging PySR itself.

To view TensorBoard in Colab:

```python
%load_ext tensorboard
%tensorboard --logdir colab_outputs/tensorboard
```

Outputs:

- `colab_outputs/pysr_run/model.pkl`
- `colab_outputs/pysr_run/metadata.json`
- `colab_outputs/pysr_run/formula.txt`
- `colab_outputs/pysr_run/equations.csv`
- `colab_outputs/pysr_run/pysr.log`

The time-dependent PySR operator set is fixed. `sin` and `cos` are included
because the template itself contains the rotating coordinate transform:

```python
binary_operators = ["+", "-", "*", "/"]
unary_operators = ["sqrt", "exp", "sin", "cos"]
```

`abs` is excluded by default because the last-frame experiments showed
piecewise ray-like artifacts.

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
  --n-samples 0 \
  --niterations 500 \
  --maxsize 60 \
  --populations 20 \
  --parsimony 0.0003 \
  --model-selection accuracy \
  --batch-size 512 \
  --timeout-minutes 25
```

By default this last-frame diagnostic excludes `abs`, because it often creates
piecewise ray-like artifacts. Add `--include-abs` only when you deliberately
want piecewise formulas.

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
- The fit template includes `phi`, because it helps PySR align the stationary
  pattern while still keeping time dependence physically constrained.
