#!/bin/bash
#SBATCH -J nem3d-detailed
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail
set -x
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.03-py3.10 || module load anaconda3/2022.10
eval "$($(command -v conda) shell.bash hook)"
conda activate nem3d

python -V
python -c "import torch; print('torch', torch.__version__, '| CUDA?', torch.cuda.is_available(), '| built with CUDA', torch.version.cuda)"
python -c "import quantus; print('quantus', quantus.__version__)"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDNN_BENCHMARK=1
export NEM_MATMUL_PRECISION=high
export NVIDIA_TF32_OVERRIDE=0
export USE_CHANNELS_LAST_3D=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LUNA_ROOT=Luna16
export NEM_BATCH_SIZE=1
export NEM_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}
export NEM_VAL_SPLIT=0.2
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d
export LUNA_FOLD_ID=0

export NEM_USE_TTA=1
export NEM_USE_TEMP=1
export NEM_DECISION_THR=0.20

export NEM_USE_CKPT=1
export NEM_FORCE_TRAIN=0

export DETAIL_PROGRESS_EVERY=10

# Detailed eval
export DETAIL_STYLE=eval_pos_only
export DETAIL_SEED=0
export DETAIL_SCAN_MAX_VAL=150995

export DETAIL_POS_POOL_MAX=200
export DETAIL_N_POS=200
export DETAIL_POS_MIN_PROB=0.2
export DETAIL_POS_REQUIRE_CORRECT=0

# Faster prediction scan
export DETAIL_PRED_BATCH_SIZE=8
export DETAIL_PIN_MEMORY=1

# Explainers
export IG_N_STEPS=64
export RISE_N_MASKS=4096

# Monotonicity
export MON_NR_SAMPLES=3
export MON_STEPS=32
export MON_EPS=1e-5
export MON_POS_LABEL=1
export MON_BATCH=8

# Top-k folders
export DETAIL_TOPK_WORST_NEM=10
export DETAIL_TOPK_BEST_NEM=10
export DETAIL_TOPK_IG_BETTER=10
export DETAIL_TOPK_NEM_BETTER=10
export DETAIL_TOPK_BEST_IG=10
export DETAIL_TOPK_WORST_IG=10

export DETAIL_TOPK_RISE_BETTER=10
export DETAIL_TOPK_NEM_BETTER_RISE=10
export DETAIL_TOPK_BEST_RISE=10
export DETAIL_TOPK_WORST_RISE=10

export PYTHONPATH="$(pwd)"

srun python -u eval_detailed_examples.py
