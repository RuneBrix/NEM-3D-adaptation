#!/bin/bash
#SBATCH -J nem3d-debug
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o slurm-debug-%j.out
#SBATCH -e slurm-debug-%j.err

set -euo pipefail
set -x
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.03-py3.10 || module load anaconda3/2022.10
eval "$($(command -v conda) shell.bash hook)"
conda activate nem3d

python -V
python -c "import torch; print('torch', torch.__version__, '| CUDA?', torch.cuda.is_available(), '| CUDA build', torch.version.cuda)"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Data / dataset setup
export LUNA_ROOT=Luna16
export NEM_BATCH_SIZE=1
export NEM_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-4}
export NEM_VAL_SPLIT=0.2
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d
export LUNA_FOLD_ID=0

export NEM_USE_TTA=1
export NEM_USE_TEMP=1
export NEM_DECISION_THR=0.20

export NEM_USE_CKPT=1
export NEM_FORCE_TRAIN=0

export IG_N_STEPS=64

export MON_POS_LABEL=1
export MON_NR_SAMPLES=3
export MON_STEPS=32
export MON_EPS=1e-5


export DETAIL_STYLE=eval_pos_only
export DETAIL_SEED=0

export DETAIL_SCAN_MAX_VAL=150995
export DETAIL_POS_POOL_MAX=1000
export DETAIL_NEG_POOL_MAX=0

export DETAIL_N_POS=128
export DETAIL_N_NEG=0

export DETAIL_POS_MIN_PROB=0.2
export DETAIL_POS_REQUIRE_CORRECT=0
export DETAIL_TOP_REQUIRE_CORRECT=0

export DETAIL_TOPK_WORST_NEM=10
export DETAIL_TOPK_BEST_NEM=10
export DETAIL_TOPK_IG_BETTER=10
export DETAIL_TOPK_NEM_BETTER=10
export DETAIL_TOPK_BEST_IG=10
export DETAIL_TOPK_WORST_IG=10

export PYTHONPATH="$(pwd)"

srun python -u debug_nem3d_masks.py
