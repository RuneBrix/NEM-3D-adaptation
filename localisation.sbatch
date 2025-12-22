#!/bin/bash
#SBATCH -J nem3d-loc-topk-rise
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o slurm-loc-topk-rise-%j.out
#SBATCH -e slurm-loc-topk-rise-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.03-py3.10 || module load anaconda3/2022.10
eval "$($(command -v conda) shell.bash hook)"
conda activate nem3d

export MPLBACKEND=Agg
export PYTHONPATH="$(pwd)"

export LUNA_ROOT=Luna16
export NEM_DATASETS=luna3d
export NEM_MODELS=phase3_3d

NPZ_ROOT="experiments/luna3d/phase3_3d/eval_detailed_examples/top_examples"
ANN_CSV="Luna16/annotations.csv"
OUT_DIR="experiments/luna3d/phase3_3d/localisation_topk_rise"

SPACING_MM=1.0
MAX_MATCH_DIST_MM=10.0
ONLY_LABEL=1
DEDUPE_GI=1

TOPK_FRAC=0.01

VIZ_MAX=24
VIZ_ALPHA=0.45

VIZ_SORT_BY="rise_topk_relevancy_mass"

srun python -u localisation_from_npz.py \
  --npz-root "$NPZ_ROOT" \
  --annotations-csv "$ANN_CSV" \
  --out-dir "$OUT_DIR" \
  --spacing-mm "$SPACING_MM" \
  --max-match-dist-mm "$MAX_MATCH_DIST_MM" \
  --only-label "$ONLY_LABEL" \
  --dedupe-gi "$DEDUPE_GI" \
  --topk-frac "$TOPK_FRAC" \
  --viz-max "$VIZ_MAX" \
  --viz-alpha "$VIZ_ALPHA" \
  --viz-sort-by "$VIZ_SORT_BY"
