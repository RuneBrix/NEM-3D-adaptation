# Extending Neural Explanation Masks to 3D Medical Images

> MSc thesis in Computer Science, University of Copenhagen. Submitted 22 December 2025. Grade: 12.

This repository contains the implementation and experiments for adapting targeted Neural Explanation Masks (NEMt) from 2D images to volumetric medical-image classifiers. The work uses lung CT data to study whether a learned 3D masking network can produce coherent explanations quickly enough to be useful while remaining competitive with established attribution methods.

## Main contribution

- implemented a 3D NEMt-style masking network with a 3D U-Net-style decoder and extractor interfaces for 3D classifier backbones;
- trained the explainer against a frozen 3D classifier to generate segmentation-like explanation masks in a forward pass;
- implemented and compared 3D Integrated Gradients and RISE baselines;
- evaluated methods using complexity, a grouped-perturbation monotonicity-style faithfulness score, localisation proxies, and runtime;
- built reproducible experiment, evaluation, visualisation, and Slurm batch-script workflows in PyTorch.

## Data and classifier

The experiments use the LUNA16 lung-nodule CT dataset and its official ten-fold split. Candidate-centred patches are resampled to isotropic 1 mm spacing and represented as 64 × 96 × 96 voxel volumes. The classifier is a 3D DenseNet-121 implemented with MONAI; the reported validation AUC was 0.8645.

Data from LUNA16 is not redistributed by this repository. Several scripts contain cluster-specific paths and expect prepared data and checkpoints, so local reproduction requires adapting `exp_config.py` and the batch scripts.

## Results

Under the thesis evaluation protocol, Integrated Gradients achieved the highest average ranking-based faithfulness score (0.5918), followed by RISE (0.5792) and NEM (0.5013). NEM produced more coherent, region-like masks and was substantially faster once trained: 0.03 seconds per sample in the reported timing run, compared with 0.64 seconds for Integrated Gradients and 35.67 seconds for RISE.

These timings and rankings are experiment-specific. The central result is a trade-off: the learned masks were fast and visually coherent, while Integrated Gradients was stronger on the selected quantitative faithfulness measure.

## Repository structure

- `attrs/nem_utils/` — 3D NEM implementation, masking network, model extractors, and method wrapper;
- `attrs/` — Integrated Gradients, RISE, Grad-CAM, saliency, and shared attribution interfaces;
- `exp_utils/` — data loading, experiment orchestration, and metrics;
- `train_nem.py` — NEM training entry point;
- `eval_models.py` and `eval_detailed_examples.py` — quantitative evaluation;
- `visualize_models.py` and `viz_*.py` — explanation visualisation;
- `*.sbatch` — Slurm jobs used on the University of Copenhagen cluster;
- `Luna16Classifier.ipynb` — classifier-development notebook;
- `environment.yml` — Conda environment with Python 3.10, PyTorch 2.2, CUDA 11.8, MONAI, Captum, and related packages.

## Environment

```bash
conda env create -f environment.yml
conda activate nem3d
```

On the original cluster, experiments were launched through the checked-in Slurm scripts, for example:

```bash
sbatch run_train_nem.sbatch
```

Review and change dataset, checkpoint, output, account, and partition paths before running. The repository includes selected model outputs and evaluation artifacts, but it is not a self-contained copy of the dataset or cluster environment.

## Scientific boundaries

The evaluation focuses on patch-level explanations for confident positive lung-nodule cases, not full clinical scans or an end-to-end diagnostic system. The reported faithfulness result depends on the chosen perturbation semantics and should not be interpreted as absolute ground truth. The thesis also identifies limitations in subset selection, hyperparameter sensitivity, out-of-distribution perturbations, target specificity, and transfer to other datasets.
