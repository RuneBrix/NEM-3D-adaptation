#!/bin/bash
#SBATCH -J nem3d-eval
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
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

export LUNA_ROOT=Luna16
export NEM_BATCH_SIZE=1
export NEM_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d
export NEM_DECISION_THR=0.20
export NEM_USE_TTA=1
export NEM_USE_TEMP=1
export LUNA_FOLD_ID=0

export NEM_USE_CKPT=1
export NEM_FORCE_TRAIN=0

export MON_POS_LABEL=1
export MON_MAX_POS_SAMPLES=10
export MON_NR_SAMPLES=3
export MON_STEPS=32
export MON_REQUIRE_CORRECT=0
export MON_MIN_PROB=0.2
export MON_SCAN_MAX_VAL=1000

export IG_N_STEPS=64
export RISE_N_MASKS=8192

export PYTHONPATH="$(pwd)"

srun python -u eval_models.py
