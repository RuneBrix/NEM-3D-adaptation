#!/bin/bash
#SBATCH -J nem3d-viz-extremes
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail
set -x
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.03-py3.10 || module load anaconda3/2022.10
eval "$($(command -v conda) shell.bash hook)"
conda activate nem3d

python -V
python -c "import numpy as np; import pandas as pd; import matplotlib; import torch; print('numpy', np.__version__, '| pandas', pd.__version__, '| matplotlib', matplotlib.__version__, '| torch', torch.__version__)"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TORCH_NUM_INTEROP_THREADS=1

export MPLBACKEND=Agg
export PYTHONPATH="$(pwd)"

export LUNA_ROOT=Luna16
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d

export VIZ_AXIS=ax
export VIZ_MODE=shared_best

export VIZ_CONTEXT=1
export VIZ_CONTEXT_FACTOR=2.0

export VIZ_VOL_PCT_LO=1.0
export VIZ_VOL_PCT_HI=99.5
export VIZ_HEAT_PCT_LO=1.0
export VIZ_HEAT_PCT_HI=99.5

export VIZ_TOPK_IG=0.20
export VIZ_TOPK_NEM=0.20
export VIZ_TOPK_RISE=0.20

# IG
export VIZ_IG_SHRINK_Q=0.80
export VIZ_IG_SHRINK_POWER=1.15
export VIZ_ALPHA_IG=0.45
export VIZ_GAMMA_IG=1.35
export VIZ_EPS_IG=0.10

# NEM
export VIZ_ALPHA_NEM=0.55
export VIZ_GAMMA_NEM=1.80
export VIZ_EPS_NEM=0.08

# RISE
export VIZ_ALPHA_RISE=0.55
export VIZ_GAMMA_RISE=1.35
export VIZ_EPS_RISE=0.08
export VIZ_RISE_SHRINK_Q=0.70
export VIZ_RISE_SHRINK_POWER=1.25
export VIZ_RISE_SMOOTH_K=3
export VIZ_RISE_SMOOTH_SIGMA=0.9

export PYTHONPATH="$(pwd)"

srun python -u viz_patch_extremes.py
