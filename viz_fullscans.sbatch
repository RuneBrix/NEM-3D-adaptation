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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONPATH="$(pwd)"

# data/model
export LUNA_ROOT=Luna16
export LUNA_FOLD_ID=0
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d

export NEM_BATCH_SIZE=1
export NEM_NUM_WORKERS=4
export NEM_VAL_SPLIT=0.2
export NEM_USE_CKPT=1
export NEM_FORCE_TRAIN=0
export RISE_N_MASKS=4096

export NEM_USE_TTA=1
export NEM_USE_TEMP=1
export NEM_DECISION_THR=0.20

export VIZ_POS_LABEL=1
export VIZ_POS_ONLY=1
export VIZ_MIN_PROB=0.3
export VIZ_REQUIRE_TP=1
export VIZ_N_SAMPLES=8
export VIZ_PROGRESS_EVERY=50

export VIZ_MAKE_GIFS=1
export VIZ_MAKE_NIFTI=1

export VIZ_TOPK_RISE=0.20
export VIZ_TOPK_IG=0.20
export VIZ_TOPK_MODE=slice

export VIZ_OVERLAY_ALPHA=0.55
export VIZ_OVERLAY_GAMMA=1.8
export VIZ_HEAT_EPS=0.08

export VIZ_OVERLAY_ALPHA_IG=0.80
export VIZ_OVERLAY_GAMMA_IG=0.8
export VIZ_HEAT_EPS_IG=0.0

export VIZ_VOL_CLIP_LO=1
export VIZ_VOL_CLIP_HI=99.5

srun python -u viz_fullscan_nem_vs_baselines.py
