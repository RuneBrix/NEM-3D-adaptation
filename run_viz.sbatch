#!/bin/bash
#SBATCH -J nem3d-viz
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --requeue

set -euo pipefail
set -x
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.03-py3.10 || module load anaconda3/2022.10
eval "$($(command -v conda) shell.bash hook)"
conda activate nem3d

python -V
python -c "import torch; print('torch', torch.__version__, '| CUDA?', torch.cuda.is_available(), '| built with CUDA', torch.version.cuda)"
python -c "import imageio; print('imageio', imageio.__version__)"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDNN_BENCHMARK=1
export NEM_MATMUL_PRECISION=high
export NVIDIA_TF32_OVERRIDE=0
export USE_CHANNELS_LAST_3D=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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

export NEM_KEEP_TARGET=0.97

export VIZ_POS_LABEL=1
export VIZ_MAX_SAMPLES=5
export VIZ_MIN_PROB=0.5
export VIZ_REQUIRE_CORRECT=0
export VIZ_SCAN_MAX_VAL=1000

export VIZ_TOPK_NEM=0
export VIZ_TOPK_MODE_NEM=volume
export VIZ_NEM_ATTR_KIND=remove

export VIZ_SLICE_BY=nem
export VIZ_SLICE_TOP_FRAC=0.02
export VIZ_HEAT_EPS=0.08
export VIZ_OVERLAY_GAMMA=1.8
export VIZ_ATTR_CLIP_LO=0
export VIZ_ATTR_CLIP_HI=100

export VIZ_NUM_SLICES=5
export VIZ_CMAP=magma
export VIZ_OVERLAY_ALPHA=0.55

# Debug overlap metrics/grid 
export VIZ_DEBUG_OVERLAP_RATIO=0.05

# GIF controls
export VIZ_GIF_MAX_FRAMES=64
export VIZ_GIF_FPS=8
export VIZ_MAKE_PANEL_GIF=1

export VIZ_IG_PREP=abs
export VIZ_RISE_PREP=raw
export VIZ_NEM_PREP=raw

export VIZ_IG_INTERNAL_BATCH=1
export IG_N_STEPS=32
export RISE_N_MASKS=8192

export PYTHONPATH="$(pwd)"

srun python -u visualize_models.py
