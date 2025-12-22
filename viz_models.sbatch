#!/bin/bash
#SBATCH -J nem3d-viz
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o slurm-viz-%j.out
#SBATCH -e slurm-viz-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.03-py3.10 || module load anaconda3/2022.10
eval "$($(command -v conda) shell.bash hook)"
conda activate nem3d

export LUNA_ROOT=Luna16
export NEM_BATCH_SIZE=1
export NEM_NUM_WORKERS=4
export NEM_VAL_SPLIT=0.2
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d
export NEM_USE_CKPT=1
export NEM_FORCE_TRAIN=0

export VIZ_POS_LABEL=1
export VIZ_POS_ONLY=1
export VIZ_MIN_PROB=0.5
export VIZ_REQUIRE_TP=1
export VIZ_N_SAMPLES=8
export VIZ_PROGRESS_EVERY=30

export VIZ_MIP_AXIS=ax
export VIZ_HEAT_PROJ_MODE=q
export VIZ_HEAT_PROJ_Q=0.99
export VIZ_HEAT_GAMMA=1.5

export VIZ_NEM_SHOW=delete

export VIZ_NEM_DISP_CLIP="1,99.5"
export VIZ_NEM_GAMMA=2.0

export VIZ_NEM_BIN=0
export VIZ_NEM_BIN_FRAC=0.02

export VIZ_NEM_CORE=0
export VIZ_NEM_CORE_FRAC=0.02

export VIZ_DEBUG_STATS=1
export VIZ_MAKE_GIFS=1
export VIZ_MAKE_NIFTI=0

export PYTHONPATH=$(pwd)

srun python -u viz_nem_vs_baselines.py
