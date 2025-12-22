import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from exp_config import CHOSEN_DATASETS, CHOSEN_MODELS
from attrs.intgrad3d import itg3d_atr
from attrs.rise3d import rs3d_atr
from attrs.nem_utils.method_nemt3d import NEMT3DMethod

dataset_name = os.environ.get("NEM_DATASETS", "luna3d").split(",")[0]
model_name   = os.environ.get("NEM_MODELS", "phase3_3d").split(",")[0]

N_VIZ_SAMPLES = int(os.environ.get("VIZ_N_SAMPLES", "8"))

VIZ_NEM_CORE      = os.environ.get("VIZ_NEM_CORE", "1") == "1"
VIZ_NEM_CORE_FRAC = float(os.environ.get("VIZ_NEM_CORE_FRAC", "0.02"))
VIZ_HEAT_GAMMA    = float(os.environ.get("VIZ_HEAT_GAMMA", "2.0"))

# Debug stats
VIZ_DEBUG_STATS = os.environ.get("VIZ_DEBUG_STATS", "0") == "1"

AXIS = os.environ.get("VIZ_MIP_AXIS", "ax")

# Whether to create GIFs and NIfTIs (0/1 via env)
MAKE_GIFS   = os.environ.get("VIZ_MAKE_GIFS", "0") == "1"
MAKE_NIFTIS = os.environ.get("VIZ_MAKE_NIFTI", "0") == "1"

# Where to save visualisations
OUT_VIZ_DIR = Path("experiments") / dataset_name / model_name / "viz_nem_vs_baselines"

# Visual parameters
ALPHA      = 0.45
VOL_CLIP   = (1, 99.5)
HEAT_CLIP  = (1, 99.5)

POS_LABEL = int(os.environ.get("VIZ_POS_LABEL", "1"))
VIZ_POS_ONLY = os.environ.get("VIZ_POS_ONLY", "1") == "1"
VIZ_MIN_PROB = float(os.environ.get("VIZ_MIN_PROB", "0.0"))
VIZ_PRINT_SKIPPED = os.environ.get("VIZ_PRINT_SKIPPED", "0") == "1"
VIZ_PROGRESS_EVERY = int(os.environ.get("VIZ_PROGRESS_EVERY", "200"))

SCAN_PRED_CSV = os.environ.get(
    "VIZ_SCAN_PRED_CSV",
    "luna16_cls3d/luna16_cls3d/fold_0/val_scan_predictions_phase3.csv",
)
SCAN_POS_LABEL = int(os.environ.get("VIZ_SCAN_POS_LABEL", "1"))
SCAN_MIN_PROB  = float(os.environ.get("VIZ_SCAN_MIN_PROB", "0.0"))
USE_SCAN_FILTER = os.environ.get("VIZ_USE_SCAN_FILTER", "0") == "1"

HEAT_PROJ_MODE = os.environ.get("VIZ_HEAT_PROJ_MODE", "max").lower()
HEAT_PROJ_Q    = float(os.environ.get("VIZ_HEAT_PROJ_Q", "0.95"))

NEM_DISP_CLIP = tuple(map(float, os.environ.get("VIZ_NEM_DISP_CLIP", "1,99.5").split(",")))
NEM_GAMMA = float(os.environ.get("VIZ_NEM_GAMMA", "2.0"))

NEM_SMOOTH = os.environ.get("VIZ_NEM_SMOOTH", "0") == "1"
NEM_SMOOTH_SIGMA = float(os.environ.get("VIZ_NEM_SMOOTH_SIGMA", "0.8"))

def keep_topk_frac(vol, frac):
    """
    Keep EXACTLY the top-k fraction of voxels (by value), zero out the rest.
    This avoids the 'quantile ties' problem when many voxels share the same value.
    """
    v = np.asarray(vol, np.float32)
    flat = v.reshape(-1)
    n = flat.size
    k = max(1, int(frac * n))
    if k >= n:
        return v.copy()

    idx = np.argpartition(flat, n - k)[n - k:]
    out = np.zeros_like(flat, dtype=np.float32)
    out[idx] = flat[idx]
    return out.reshape(v.shape)

def norm01_nonzero_by_hi(vol, hi_q=99.5, eps=1e-9):
    """
    Normalize a nonnegative map to [0,1] by dividing by a robust high percentile
    of NONZERO values. Zeros stay zero (so sparsity stays visible).
    """
    v = np.asarray(vol, np.float32)
    v = np.maximum(v, 0.0)
    nz = v[v > 0]
    if nz.size == 0:
        return np.zeros_like(v, np.float32)
    hi = float(np.percentile(nz, hi_q))
    if not np.isfinite(hi) or hi < eps:
        hi = float(nz.max())
    return np.clip(v / (hi + eps), 0.0, 1.0)

def axis_reduce_for(axis):
    return 0 if axis == "ax" else (1 if axis == "co" else 2)

def mip_coverage_gated(vol3d_01, axis="ax"):
    """
    MIP(max) but gated by coverage along the ray:
      out = mip(max) * mean( vol>0 along axis )
    This keeps structure that persists across slices and suppresses random hits.
    """
    vol3d_01 = np.asarray(vol3d_01, np.float32)
    m = mip(vol3d_01, axis)  # max projection
    rax = axis_reduce_for(axis)
    cov = (vol3d_01 > 0).mean(axis=rax).astype(np.float32)  # [0..1]
    return np.clip(m * cov, 0.0, 1.0)

def to01(a, clip=None):
    a = np.asarray(a, np.float32)
    if clip is not None:
        lo, hi = np.percentile(a, clip)
        if hi > lo:
            a = np.clip(a, lo, hi)
    rng = float(a.max() - a.min())
    if not np.isfinite(rng) or rng < 1e-9:
        return np.zeros_like(a, np.float32)
    return (a - a.min()) / (rng + 1e-9)

def gamma01(h, gamma: float):
    h = np.clip(np.asarray(h, np.float32), 0.0, 1.0)
    if gamma is None or abs(gamma - 1.0) < 1e-6:
        return h
    if gamma <= 0:
        return h
    return np.power(h, gamma)

def norm3d(a, clip=None):
    a = np.asarray(a, np.float32)
    if clip is not None:
        lo, hi = np.percentile(a, clip)
        if hi > lo:
            a = np.clip(a, lo, hi)
    amin = float(a.min())
    amax = float(a.max())
    rng = amax - amin
    if not np.isfinite(rng) or rng < 1e-9:
        return np.zeros_like(a, np.float32)
    return (a - amin) / (rng + 1e-9)

def overlay_red01(gray2d, heat2d, alpha=ALPHA):
    g = np.clip(gray2d, 0.0, 1.0)
    h = np.clip(heat2d, 0.0, 1.0)
    base = np.dstack([g, g, g])
    red  = np.dstack([h, np.zeros_like(h), np.zeros_like(h)])
    return np.uint8(np.clip((1 - alpha) * base + alpha * red, 0, 1) * 255)

def overlay_red(gray2d, heat2d, alpha=ALPHA, already_01: bool = False):
    g = np.asarray(gray2d, np.float32)
    h = np.asarray(heat2d, np.float32)
    if not already_01:
        g = to01(g, VOL_CLIP)
        h = to01(h, HEAT_CLIP)
    base = np.dstack([g, g, g])
    red  = np.dstack([h, np.zeros_like(h), np.zeros_like(h)])
    return np.uint8(np.clip((1 - alpha) * base + alpha * red, 0, 1) * 255)

def mip(vol3d, axis="ax"):
    if axis == "ax":
        return vol3d.max(axis=0)
    if axis == "co":
        return vol3d.max(axis=1)
    return vol3d.max(axis=2)

def proj(vol3d, axis="ax", mode="mean", q=0.95):
    vol3d = np.asarray(vol3d)
    if axis == "ax":
        reduce_axis = 0
    elif axis == "co":
        reduce_axis = 1
    else:
        reduce_axis = 2

    if mode == "max":
        return vol3d.max(axis=reduce_axis)
    if mode == "mean":
        return vol3d.mean(axis=reduce_axis)
    return np.quantile(vol3d, q, axis=reduce_axis)

def to_3d_volume(vol: np.ndarray) -> np.ndarray:
    arr = np.asarray(vol)
    if arr.ndim == 5:
        if arr.shape[0] != 1:
            print(f"[warn] to_3d_volume: batch dim >1 ({arr.shape[0]}), using first sample only.")
        arr = arr[0]  # [C,D,H,W]
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]  # [D,H,W]
        else:
            arr = arr.mean(axis=0)  # average channels
    if arr.ndim != 3:
        raise ValueError(f"to_3d_volume: expected 3D array (D,H,W), got shape {arr.shape}")
    return arr.astype(np.float32)

def print_stats_3d(uid, name, vol):
    v = np.asarray(vol, np.float32)
    if v.size == 0:
        print(f"[viz-debug][{uid}] {name}: EMPTY")
        return
    v_flat = v.reshape(-1)
    print(
        f"[viz-debug][{uid}] {name}: "
        f"shape={v.shape}, "
        f"min={np.nanmin(v_flat):.4g}, "
        f"max={np.nanmax(v_flat):.4g}, "
        f"mean={np.nanmean(v_flat):.4g}, "
        f"std={np.nanstd(v_flat):.4g}, "
        f"frac>0.5={(v_flat > 0.5).mean():.3f}, "
        f"frac>0.9={(v_flat > 0.9).mean():.3f}"
    )

def print_stats_2d(uid, name, vol2d):
    v = np.asarray(vol2d, np.float32)
    v_flat = v.reshape(-1)
    print(
        f"[viz-debug][{uid}] {name} [2D]: "
        f"shape={v.shape}, "
        f"min={np.nanmin(v_flat):.4g}, "
        f"max={np.nanmax(v_flat):.4g}, "
        f"mean={np.nanmean(v_flat):.4g}, "
        f"std={np.nanstd(v_flat):.4g}, "
        f"frac>0.5={(v_flat > 0.5).mean():.3f}, "
        f"frac>0.9={(v_flat > 0.9).mean():.3f}"
    )

def topk_overlap(uid, name_a, a, name_b, b, frac=0.01):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    n = a.size
    k = max(1, int(frac * n))

    idx_a = np.argpartition(-a, k-1)[:k]
    idx_b = np.argpartition(-b, k-1)[:k]

    mask_a = np.zeros(n, dtype=bool)
    mask_b = np.zeros(n, dtype=bool)
    mask_a[idx_a] = True
    mask_b[idx_b] = True

    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = inter / union if union > 0 else 0.0

    print(
        f"[viz-debug][{uid}] top-{frac*100:.1f}% IoU({name_a}, {name_b}) = "
        f"{iou:.3f} (inter={inter}, union={union})"
    )
    return iou

def save_scroll_gif(vol, imp, out_gif, plane="axial", alpha=0.45, fps=12, imp_already_01=False):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[warn] imageio not installed; skipping GIF:", out_gif)
        return

    vol = np.asarray(vol, np.float32)
    imp = np.asarray(imp, np.float32)

    vol_n = to01(vol, VOL_CLIP)
    if imp_already_01:
        imp_n = np.clip(imp, 0.0, 1.0)
    else:
        imp_n = to01(imp, HEAT_CLIP)

    frames = []

    if plane == "axial":
        indices = range(vol_n.shape[0])
        slicer = lambda a, i: a[i, :, :]
    elif plane == "coronal":
        indices = range(vol_n.shape[1])
        slicer = lambda a, i: a[:, i, :]
    else:
        indices = range(vol_n.shape[2])
        slicer = lambda a, i: a[:, :, i]

    for i in indices:
        img2d = slicer(vol_n, i)
        heat2d = slicer(imp_n, i)
        frame = overlay_red(img2d, heat2d, alpha=alpha, already_01=True)
        frames.append(frame)

    out_gif = str(out_gif)
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, frames, duration=1.0 / float(fps))
    print("[ok] wrote GIF", out_gif)

def save_nifti(vol, imp, out_dir, keep=None, delete=None, logits=None, voxel_spacing=None):
    try:
        import nibabel as nib
    except Exception:
        print("[warn] nibabel not installed; skipping NIfTI export:", out_dir)
        return

    vol = np.asarray(vol, np.float32)
    imp = np.asarray(imp, np.float32)
    os.makedirs(out_dir, exist_ok=True)

    aff = np.eye(4, dtype=np.float32)
    if voxel_spacing is not None and len(voxel_spacing) == 3:
        sx, sy, sz = voxel_spacing
        aff[0, 0], aff[1, 1], aff[2, 2] = sx, sy, sz

    nib.save(nib.Nifti1Image(vol, aff), os.path.join(out_dir, "volume.nii.gz"))
    nib.save(nib.Nifti1Image(imp, aff), os.path.join(out_dir, "importance.nii.gz"))

    if keep is not None:
        nib.save(nib.Nifti1Image(np.asarray(keep, np.float32), aff), os.path.join(out_dir, "keep.nii.gz"))
    if delete is not None:
        nib.save(nib.Nifti1Image(np.asarray(delete, np.float32), aff), os.path.join(out_dir, "delete.nii.gz"))
    if logits is not None:
        nib.save(nib.Nifti1Image(np.asarray(logits, np.float32), aff), os.path.join(out_dir, "mask_logits.nii.gz"))

    print("[ok] wrote NIfTI to", out_dir)

print(f"[info] Using dataset='{dataset_name}', model='{model_name}' for visualisation.")
OUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

data_obj = CHOSEN_DATASETS[dataset_name]()
train_loader, val_loader = data_obj.get_data()
if val_loader is None:
    raise RuntimeError("Validation loader not found in dataset.")

model = CHOSEN_MODELS[model_name]().eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"[info] Model device: {device}")

use_predicted = True
IG_explainer   = itg3d_atr(model, train_data=None, use_predicted_labels=use_predicted)
RISE_explainer = rs3d_atr(model, train_data=None, use_predicted_labels=use_predicted)
nem_explainer  = NEMT3DMethod(model, train_loader=None, train_or_load=False, device=str(device))
print("[info] Explainers initialised (IG, RISE, NEM).")

# Optional: Scan-level CSV filter
positive_scans = None
if USE_SCAN_FILTER and Path(SCAN_PRED_CSV).is_file():
    df_scans = pd.read_csv(SCAN_PRED_CSV)
    if "seriesuid" not in df_scans.columns:
        raise RuntimeError(f"{SCAN_PRED_CSV} does not contain 'seriesuid' column.")
    df_scans["seriesuid"] = df_scans["seriesuid"].astype(str)

    n_label_pos = (df_scans["label"] == SCAN_POS_LABEL).sum()
    pos_mask = (df_scans["label"] == SCAN_POS_LABEL)
    if "prob_max" in df_scans.columns:
        pos_mask &= (df_scans["prob_max"] >= SCAN_MIN_PROB)

    positive_scans = set(df_scans.loc[pos_mask, "seriesuid"].tolist())
    print(
        f"[info] Loaded scan-level preds from {SCAN_PRED_CSV}: "
        f"{len(df_scans)} scans total, {n_label_pos} with label={SCAN_POS_LABEL}, "
        f"{len(positive_scans)} of them with prob_max >= {SCAN_MIN_PROB}."
    )
elif USE_SCAN_FILTER:
    print(
        f"[warn] USE_SCAN_FILTER=1 but no scan-level CSV found at {SCAN_PRED_CSV}; "
        "no scan-level restriction will be applied."
    )
else:
    print("[info] Scan-level CSV not used (USE_SCAN_FILTER=0).")

# Build list of positive validation candidates
val_dataset = val_loader.dataset
if hasattr(val_dataset, "indices"):
    full_dataset = val_dataset.dataset
    val_indices = list(val_dataset.indices)
else:
    full_dataset = val_dataset
    val_indices = list(range(len(full_dataset)))

if not hasattr(full_dataset, "rows"):
    raise RuntimeError("Expected LunaCandidates3DDataset with 'rows' attribute as full_dataset.")

pos_val_indices = []
for gi in val_indices:
    row = full_dataset.rows[gi]
    lbl = int(row.get("label", 0))
    if lbl != POS_LABEL:
        continue

    if USE_SCAN_FILTER and positive_scans is not None:
        uid = str(row.get("uid", row.get("seriesuid", "")))
        if uid not in positive_scans:
            continue

    pos_val_indices.append(gi)

print(
    f"[info] Validation positives ({'after scan filter' if USE_SCAN_FILTER else 'no scan filter'}): "
    f"{len(pos_val_indices)} candidates (POS_LABEL={POS_LABEL})."
)

if len(pos_val_indices) == 0:
    raise RuntimeError("No positive candidates found in validation set.")

candidate_infos = []
print(f"[info] Starting prediction pass over {len(pos_val_indices)} positive validation candidates...")

with torch.no_grad():
    for j, gi in enumerate(pos_val_indices):
        X_patch, y = full_dataset[gi]  # X_patch: [1,D,H,W]
        label = int(y.item()) if torch.is_tensor(y) else int(y)

        if VIZ_POS_ONLY and label != POS_LABEL:
            continue

        X_sample = X_patch.unsqueeze(0).to(device)  # [1,1,D,H,W]
        logits = model(X_sample)

        if logits.ndim == 1 or logits.shape[1] == 1:
            logit_scalar = float(logits.view(-1)[0].item())
            prob_pos = float(torch.sigmoid(logits.view(-1))[0].item())
            pred_label = int((logits.view(-1) > 0).item())
        else:
            probs = torch.softmax(logits, dim=1)
            pred_label = int(torch.argmax(probs, dim=1).item())
            prob_pos = float(probs[0, POS_LABEL].item())
            logit_scalar = float(logits[0, POS_LABEL].item())

        row = full_dataset.rows[gi]
        uid = str(row.get("uid", row.get("seriesuid", f"sample-{gi:06d}")))

        candidate_infos.append({
            "idx": gi,
            "uid": uid,
            "label": label,
            "pred_label": pred_label,
            "prob_pos": prob_pos,
            "logit_scalar": logit_scalar,
        })

        if VIZ_PROGRESS_EVERY > 0 and (j + 1) % VIZ_PROGRESS_EVERY == 0:
            print(f"[progress] scanned {j+1}/{len(pos_val_indices)} positives, collected {len(candidate_infos)} preds.")

print(f"[info] Computed predictions for {len(candidate_infos)} positive validation candidates.")

top10 = sorted(candidate_infos, key=lambda c: c["prob_pos"], reverse=True)[:10]
print("[debug] Top-10 positive validation candidates by prob_pos:")
for c in top10:
    print(
        f"  uid={c['uid']} | prob_pos={c['prob_pos']:.3f} | "
        f"label={c['label']} | pred={c['pred_label']} | logit={c['logit_scalar']:.3f}"
    )
REQUIRE_TP = os.environ.get("VIZ_REQUIRE_TP", "1") == "1"
pool = candidate_infos

if REQUIRE_TP:
    tp_pool = [c for c in candidate_infos if c["label"] == POS_LABEL and c["pred_label"] == POS_LABEL]
    if tp_pool:
        print(f"[info] Restricting viz pool to {len(tp_pool)} true positives (label={POS_LABEL}, pred={POS_LABEL}).")
        pool = tp_pool
    else:
        print("[warn] No true positives found; falling back to label-only positives.")
        pool = candidate_infos

if VIZ_MIN_PROB > 0.0:
    thresholded = [c for c in pool if c["prob_pos"] >= VIZ_MIN_PROB]
    print(f"[info] {len(thresholded)} / {len(pool)} in pool have prob_pos >= {VIZ_MIN_PROB:.3f}.")
    if thresholded:
        strong = sorted(thresholded, key=lambda c: c["prob_pos"], reverse=True)[:N_VIZ_SAMPLES]
        effective_min_prob = VIZ_MIN_PROB
    else:
        print(f"[info] None reached threshold; using top {N_VIZ_SAMPLES} by prob_pos in pool.")
        strong = sorted(pool, key=lambda c: c["prob_pos"], reverse=True)[:N_VIZ_SAMPLES]
        effective_min_prob = 0.0
else:
    strong = sorted(pool, key=lambda c: c["prob_pos"], reverse=True)[:N_VIZ_SAMPLES]
    effective_min_prob = 0.0

print(
    f"[info] Selected {len(strong)} candidates for visualisation "
    f"(VIZ_MIN_PROB={VIZ_MIN_PROB}, effective_min_prob={effective_min_prob}, N_VIZ_SAMPLES={N_VIZ_SAMPLES})."
)

saved = 0

if HEAT_PROJ_MODE not in ("max", "mean", "q"):
    print(f"[warn] Unknown VIZ_HEAT_PROJ_MODE='{HEAT_PROJ_MODE}', defaulting to 'max'.")
    HEAT_PROJ_MODE = "max"

for info in strong:
    gi           = info["idx"]
    uid          = info["uid"]
    label        = info["label"]
    pred_label   = info["pred_label"]
    prob_pos     = info["prob_pos"]
    logit_scalar = info["logit_scalar"]

    if prob_pos < effective_min_prob:
        if VIZ_PRINT_SKIPPED:
            print(f"[skip] uid={uid}: prob_pos={prob_pos:.3f} < effective_min_prob={effective_min_prob:.3f}")
        continue

    X_patch, y = full_dataset[gi]
    X_sample = X_patch.unsqueeze(0).to(device)  # [1,1,D,H,W]

    if VIZ_POS_ONLY and label != POS_LABEL:
        if VIZ_PRINT_SKIPPED:
            print(f"[skip] uid={uid}: label={label}, want POS_LABEL={POS_LABEL}")
        continue

    print(
        f"[viz] uid={uid} | label={label} | pred={pred_label} | "
        f"logit_pos={logit_scalar:.3f} | prob_pos={prob_pos:.3f} (POS_LABEL={POS_LABEL})"
    )

    # Base volume
    vol3d = to_3d_volume(X_sample.detach().cpu().numpy())

    # IG
    ig_attr = IG_explainer.gen_attr(X_sample)
    ig3d = to_3d_volume(ig_attr)
    ig_pos = ig3d if label == POS_LABEL else -ig3d
    ig_pos = np.maximum(ig_pos, 0.0)

    # RISE (target = predicted label)
    rise_attr = RISE_explainer.gen_attr(X_sample, int(pred_label))
    rise3d = to_3d_volume(rise_attr)

    # NEM
    mask_logits, X_masked, mask_tensor = nem_explainer.gen_mask(X_sample)

    def pos_prob(logits):
        if logits.ndim == 1 or logits.shape[1] == 1:
            return float(torch.sigmoid(logits.view(-1))[0].item())
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, POS_LABEL].item())

    with torch.no_grad():
        p_orig = pos_prob(model(X_sample))
        p_mask = pos_prob(model(X_masked))

        inv = 1.0 - mask_tensor
        baseline = X_sample.mean(dim=(2, 3, 4), keepdim=True)
        X_masked_inv = X_sample * inv + (1.0 - inv) * baseline
        p_mask_inv = pos_prob(model(X_masked_inv))

    # If inverted mask preserves prediction better, flip semantics
    nem_keep_raw = to_3d_volume(mask_tensor.detach().cpu().numpy())  # [0,1]
    mask_flipped = False
    if p_mask_inv > p_mask:
        nem_keep_raw = 1.0 - nem_keep_raw
        X_masked = X_masked_inv
        p_mask = p_mask_inv
        mask_flipped = True

    print(f"[nem-check][{uid}] p_orig={p_orig:.3f} p_masked={p_mask:.3f} flipped={mask_flipped}")

    nem_keep_raw3d = np.clip(nem_keep_raw.astype(np.float32), 0.0, 1.0)
    nem_del_raw3d  = 1.0 - nem_keep_raw3d

    if VIZ_NEM_CORE:
        frac = float(np.clip(VIZ_NEM_CORE_FRAC, 1e-6, 1.0))
        thr_keep = np.quantile(nem_keep_raw3d, 1.0 - frac)
        nem_keep_raw3d = np.where(nem_keep_raw3d >= thr_keep, nem_keep_raw3d, 0.0)

        thr_del = np.quantile(nem_del_raw3d, 1.0 - frac)
        nem_del_raw3d = np.where(nem_del_raw3d >= thr_del, nem_del_raw3d, 0.0)

    # Logits map (optional viz/debug)
    mask_logits3d = to_3d_volume(mask_logits.detach().cpu().numpy())
    if mask_flipped:
        mask_logits3d = -mask_logits3d

    print(
        f"[nem-area][{uid}] del_mean={nem_del_raw3d.mean():.3f} "
        f"keep_mean={nem_keep_raw3d.mean():.3f} logits_std={mask_logits3d.std():.3f}"
    )

    nem_keep_disp3d = to01(nem_keep_raw3d, clip=NEM_DISP_CLIP)
    nem_del_disp3d  = to01(nem_del_raw3d,  clip=NEM_DISP_CLIP)

    if NEM_SMOOTH:
        try:
            from scipy.ndimage import gaussian_filter
            nem_keep_disp3d = gaussian_filter(nem_keep_disp3d, sigma=NEM_SMOOTH_SIGMA)
            nem_del_disp3d  = gaussian_filter(nem_del_disp3d,  sigma=NEM_SMOOTH_SIGMA)
            nem_keep_disp3d = np.clip(nem_keep_disp3d, 0.0, 1.0)
            nem_del_disp3d  = np.clip(nem_del_disp3d,  0.0, 1.0)
        except Exception as e:
            print(f"[warn] NEM_SMOOTH=1 but could not import/apply scipy gaussian_filter: {e}")

    nem_keep3d_n   = nem_keep_disp3d.astype(np.float32)
    nem_delete3d_n = nem_del_disp3d.astype(np.float32)

    vol3d_n  = norm3d(vol3d, clip=VOL_CLIP)
    ig3d_n   = norm3d(ig_pos, clip=HEAT_CLIP)
    rise3d_n = norm3d(rise3d, clip=HEAT_CLIP)

    img_mip_max  = mip(vol3d_n, AXIS)
    ig_mip_max   = mip(ig3d_n, AXIS)
    rise_mip_max = mip(rise3d_n, AXIS)

    nem_keep_mip_max = mip(nem_keep3d_n, AXIS)
    nem_del_mip_max  = mip(nem_delete3d_n, AXIS)

    img_proj = proj(vol3d_n, AXIS, mode="max")

    if HEAT_PROJ_MODE == "max":
        ig_proj       = proj(ig3d_n, AXIS, mode="max")
        rise_proj     = proj(rise3d_n, AXIS, mode="max")
        nem_keep_proj = proj(nem_keep3d_n, AXIS, mode="max")
        nem_del_proj  = proj(nem_delete3d_n, AXIS, mode="max")
        heat_label = "max"
    elif HEAT_PROJ_MODE == "mean":
        ig_proj       = proj(ig3d_n, AXIS, mode="mean")
        rise_proj     = proj(rise3d_n, AXIS, mode="mean")
        nem_keep_proj = proj(nem_keep3d_n, AXIS, mode="mean")
        nem_del_proj  = proj(nem_delete3d_n, AXIS, mode="mean")
        heat_label = "mean"
    else:
        ig_proj       = proj(ig3d_n, AXIS, mode="quantile", q=HEAT_PROJ_Q)
        rise_proj     = proj(rise3d_n, AXIS, mode="quantile", q=HEAT_PROJ_Q)
        nem_keep_proj = proj(nem_keep3d_n, AXIS, mode="quantile", q=HEAT_PROJ_Q)
        nem_del_proj  = proj(nem_delete3d_n, AXIS, mode="quantile", q=HEAT_PROJ_Q)
        heat_label = f"q{HEAT_PROJ_Q:.2f}"

    if VIZ_DEBUG_STATS:
        print_stats_3d(uid, "IG_pos3d", ig3d_n)
        print_stats_3d(uid, "RISE3d", rise3d_n)
        print_stats_3d(uid, "NEM_keep_disp3d", nem_keep3d_n)
        print_stats_3d(uid, "NEM_del_disp3d", nem_delete3d_n)

        print_stats_2d(uid, "IG_MIP_max", ig_mip_max)
        print_stats_2d(uid, "RISE_MIP_max", rise_mip_max)
        print_stats_2d(uid, "NEM_keep_MIP_max", nem_keep_mip_max)
        print_stats_2d(uid, "NEM_del_MIP_max", nem_del_mip_max)

        print_stats_2d(uid, f"IG_PROJ_{heat_label}", ig_proj)
        print_stats_2d(uid, f"RISE_PROJ_{heat_label}", rise_proj)
        print_stats_2d(uid, f"NEMkeep_PROJ_{heat_label}", nem_keep_proj)
        print_stats_2d(uid, f"NEMdel_PROJ_{heat_label}", nem_del_proj)

        topk_overlap(uid, "IG_pos3d", ig3d_n, "NEM_keep_disp3d", nem_keep3d_n, frac=0.01)
        topk_overlap(uid, "IG_pos3d", ig3d_n, "NEM_del_disp3d", nem_delete3d_n, frac=0.01)

    panels = [
        ("IG — MIP(max)",       ig_mip_max,       img_mip_max),
        ("RISE — MIP(max)",     rise_mip_max,     img_mip_max),
        ("NEM keep — MIP(max)", nem_keep_mip_max, img_mip_max),
        ("NEM del — MIP(max)",  nem_del_mip_max,  img_mip_max),

        (f"IG — PROJ({heat_label})",       ig_proj,       img_proj),
        (f"RISE — PROJ({heat_label})",     rise_proj,     img_proj),
        (f"NEM keep — PROJ({heat_label})", nem_keep_proj, img_proj),
        (f"NEM del — PROJ({heat_label})",  nem_del_proj,  img_proj),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(22, 10))
    axs = axs.reshape(-1)

    for ax, (title, heat2d, base2d) in zip(axs, panels):
        g = NEM_GAMMA if title.startswith("NEM") else VIZ_HEAT_GAMMA
        ax.imshow(overlay_red01(base2d, gamma01(heat2d, g)), interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{title} — [{AXIS}]")

    fig.suptitle(
        f"UID={uid} | label={label} | pred={pred_label} | "
        f"logit_pos={logit_scalar:.3f} | prob_pos={prob_pos:.3f} | "
        f"heat_proj={heat_label}",
        y=0.98
    )
    plt.tight_layout()

    out_png = OUT_VIZ_DIR / f"{uid}.mipproj-{AXIS}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")

    if MAKE_GIFS:
        gif_dir = OUT_VIZ_DIR / "gifs" / uid
        gif_dir.mkdir(parents=True, exist_ok=True)
        save_scroll_gif(vol3d, ig_pos,          gif_dir / f"{uid}_ig_axial.gif",       plane="axial")
        save_scroll_gif(vol3d, rise3d,          gif_dir / f"{uid}_rise_axial.gif",     plane="axial")
        save_scroll_gif(vol3d, nem_keep3d_n,    gif_dir / f"{uid}_nem_keep_axial.gif", plane="axial", imp_already_01=True)
        save_scroll_gif(vol3d, nem_delete3d_n,  gif_dir / f"{uid}_nem_del_axial.gif",  plane="axial", imp_already_01=True)

    if MAKE_NIFTIS:
        base_dir = OUT_VIZ_DIR / "nifti" / uid
        save_nifti(
            vol3d,
            nem_keep3d_n,
            str(base_dir / "nem_display"),
            keep=nem_keep_raw3d,
            delete=nem_del_raw3d,
            logits=mask_logits3d,
            voxel_spacing=None,
        )

    saved += 1

print(
    f"[done] saved {saved} visualised samples under {OUT_VIZ_DIR} "
    f"(from {len(candidate_infos)} positive validation candidates)."
)
