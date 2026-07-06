# %% [markdown]
# # Rotating spoke symbolic regression with PySR
#
# Copy this file into a Colab notebook cell-by-cell. The first cell downloads
# the GitHub repository into Colab.

# %%
REPO_URL = "https://github.com/kolotinsky1998/spoke_approximation.git"
PROJECT_DIR = "/content/spoke_approximation"

# %%
import os

if not os.path.exists(PROJECT_DIR):
    !git clone {REPO_URL} {PROJECT_DIR}
else:
    %cd {PROJECT_DIR}
    !git pull

# %%
%pip install -U "pysr==1.5.9" numpy pandas scipy scikit-learn matplotlib joblib

# %%
%cd {PROJECT_DIR}

# %%
!python colab_scripts/prepare_spoke_dataset.py \
  --data-dir data \
  --out-dir colab_outputs/prepared \
  --steady-fraction 0.55 \
  --samples-per-frame 1200 \
  --smooth-sigma 0.8 \
  --random-state 7

# %%
!python colab_scripts/fit_spoke_pysr.py \
  --dataset colab_outputs/prepared/spoke_dataset.npz \
  --metadata colab_outputs/prepared/metadata.json \
  --out-dir colab_outputs/pysr_run \
  --niterations 80 \
  --maxsize 25 \
  --populations 8 \
  --parsimony 0.004 \
  --timeout-minutes 12

# %%
!python colab_scripts/evaluate_spoke_formula.py \
  --model colab_outputs/pysr_run/model.pkl \
  --metadata colab_outputs/prepared/metadata.json \
  --data-dir data \
  --out-dir colab_outputs/evaluation

# %%
from IPython.display import Image, display

display(Image("colab_outputs/prepared/angle_fit.png"))
display(Image("colab_outputs/prepared/sampled_points.png"))

# %%
import glob

for path in sorted(glob.glob("colab_outputs/evaluation/comparison_*.png")):
    display(Image(path))

# %%
print(open("colab_outputs/pysr_run/formula.txt").read())
