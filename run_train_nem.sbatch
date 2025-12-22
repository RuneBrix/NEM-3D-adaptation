#!/bin/bash
#SBATCH -J nem3d-train
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
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
nvidia-smi || true

# Threads / perf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDNN_BENCHMARK=1
export NEM_MATMUL_PRECISION=high
export NVIDIA_TF32_OVERRIDE=0
export USE_CHANNELS_LAST_3D=1

# Data
export LUNA_ROOT=Luna16
export NEM_BATCH_SIZE=1
export NEM_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}
export LUNA_FOLD_ID=0
export NEM_TBLOG=1

# NEM training knobs
export NEM_POS_ONLY=1
unset NEM_MAX_NEG_POS_RATIO

export NEM_EPOCHS=4
unset NEM_LIMIT_TRAIN
export NEM_LOG_EVERY=50
export NEM_AMP=1
export NEM_KEEP_TARGET=0.97
export NEMT_LAMBDA=1
export NEM_ACCUM=4
export NEM_CLIP=1.0

# Smoothing knobs
export NEM_SMOOTH_MASK=0
export NEM_SMOOTH_K=5
export NEM_SMOOTH_SIGMA=1.0

# Checkpoint policy – training run
export NEM_USE_CKPT=0
export NEM_FORCE_TRAIN=1

# Debug
export NEM_DEBUG=1

# Scope
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d

export PYTHONPATH=$(pwd)

srun python -u train_nem.py