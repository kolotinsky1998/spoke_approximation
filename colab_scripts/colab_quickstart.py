# %% [markdown]
# # Rotating spoke symbolic regression with PySR
#
# Copy this file into a Colab notebook cell-by-cell, or run the shell commands
# directly from a Colab terminal. Adjust `PROJECT_DIR` if your data is elsewhere.

# %%
from google.colab import drive
drive.mount("/content/drive")

# %%
PROJECT_DIR = "/content/drive/MyDrive/spoke_approximation"

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
  --niterations 800 \
  --maxsize 45 \
  --populations 24 \
  --parsimony 0.002

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
