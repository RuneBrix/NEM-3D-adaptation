#!/bin/bash
#SBATCH -J nem3d-debug-rise
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o slurm-debug-rise-%j.out
#SBATCH -e slurm-debug-rise-%j.err

set -euo pipefail
set -x
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.03-py3.10 || module load anaconda3/2022.10
eval "$($(command -v conda) shell.bash hook)"
conda activate nem3d

python -V
python -c "import torch; print('torch', torch.__version__, '| CUDA?', torch.cuda.is_available(), '| built with CUDA', torch.version.cuda)"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDNN_BENCHMARK=1
export NEM_MATMUL_PRECISION=high
export NVIDIA_TF32_OVERRIDE=0
export USE_CHANNELS_LAST_3D=1

# Data / model
export LUNA_ROOT=Luna16
export NEM_BATCH_SIZE=1
export NEM_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}
export NEM_VAL_SPLIT=0.2
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d

export MON_POS_LABEL=1

# RISE + debug parameters
export RISE_N_MASKS=8192
export DEBUG_RISE_N_POS=8
export DEBUG_RISE_MAX_VAL_SCAN=400
export DEBUG_RISE_NR_SAMPLES=5
export DEBUG_RISE_STEPS=32

export EVAL_MAX_SAMPLES=10

export PYTHONPATH="$(pwd)"

srun python -u debug_rise3d_analysis.py
