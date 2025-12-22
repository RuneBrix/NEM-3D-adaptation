import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp_config import CHOSEN_DATASETS, CHOSEN_MODELS, TEMPERATURE
from attrs.intgrad3d import itg3d_atr
from attrs.rise3d import rs3d_atr
from attrs.nem_utils.method_nemt3d import NEMT3DMethod

# For loading full CT volumes at 1mm spacing
from exp_utils.luna_monai3d import CachedLoadPreprocessd, SPACING_PATCH

def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if v in ("", "None", None):
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v in ("", "None", None):
        return default
    return int(v)

def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if v in ("", "None", None):
        return default
    return float(v)

def env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    if v in ("", "None", None):
        return default
    return str(v)


def call_gen_attr(explainer, x5d):
    """
    prefer gen_attr(x, None) or gen_attr(x) (no class index).
    Fallbacks included for older signatures.
    """
    try:
        return explainer.gen_attr(x5d, None)
    except TypeError:
        pass
    try:
        return explainer.gen_attr(x5d)
    except TypeError:
        pass
    return explainer.gen_attr(x5d, 0)


dataset_name = os.environ.get("NEM_DATASETS", "luna3d").split(",")[0]
model_name   = os.environ.get("NEM_MODELS", "phase3_3d").split(",")[0]

LUNA_ROOT = Path(os.environ.get("LUNA_ROOT", "Luna16"))
FOLD_ID = env_int("LUNA_FOLD_ID", 0)

# prediction logic knobs (to match eval)
USE_TTA      = env_bool("NEM_USE_TTA", False)
USE_TEMP     = env_bool("NEM_USE_TEMP", False)
DECISION_THR = env_float("NEM_DECISION_THR", 0.5)

# selection
POS_LABEL          = env_int("VIZ_POS_LABEL", 1)
VIZ_POS_ONLY       = env_bool("VIZ_POS_ONLY", True)
VIZ_MIN_PROB       = env_float("VIZ_MIN_PROB", 0.0)
REQUIRE_TP         = env_bool("VIZ_REQUIRE_TP", True)
N_VIZ_SAMPLES      = env_int("VIZ_N_SAMPLES", 8)
VIZ_PROGRESS_EVERY = env_int("VIZ_PROGRESS_EVERY", 200)
VIZ_PRINT_SKIPPED  = env_bool("VIZ_PRINT_SKIPPED", False)

# optional scan-level filter (kept, but still simple)
USE_SCAN_FILTER = env_bool("VIZ_USE_SCAN_FILTER", False)
SCAN_PRED_CSV = env_str(
    "VIZ_SCAN_PRED_CSV",
    f"luna16_cls3d/luna16_cls3d/fold_{FOLD_ID}/val_scan_predictions_phase3.csv",
)
SCAN_POS_LABEL = env_int("VIZ_SCAN_POS_LABEL", 1)
SCAN_MIN_PROB  = env_float("VIZ_SCAN_MIN_PROB", 0.0)

RISE_N_MASKS = os.environ.get("RISE_N_MASKS", "")

# outputs
MAKE_GIFS   = env_bool("VIZ_MAKE_GIFS", False)
MAKE_NIFTIS = env_bool("VIZ_MAKE_NIFTI", False)

OUT_VIZ_DIR = Path("experiments") / dataset_name / model_name / "viz_fullscan_nem_vs_baselines"
OUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# CT display normalization
VOL_CLIP_LO = env_float("VIZ_VOL_CLIP_LO", 1.0)
VOL_CLIP_HI = env_float("VIZ_VOL_CLIP_HI", 99.5)

# overlay styling (keep just a few)
ALPHA_BASE = env_float("VIZ_OVERLAY_ALPHA", 0.55)
GAMMA_BASE = env_float("VIZ_OVERLAY_GAMMA", 1.8)
HEAT_EPS   = env_float("VIZ_HEAT_EPS", 0.08)

# IG often needs different visibility
ALPHA_IG   = env_float("VIZ_OVERLAY_ALPHA_IG", 0.80)
GAMMA_IG   = env_float("VIZ_OVERLAY_GAMMA_IG", 0.8)
HEAT_EPS_IG= env_float("VIZ_HEAT_EPS_IG", 0.0)

# RISE/IG sparsification (optional)
TOPK_RISE  = env_float("VIZ_TOPK_RISE", 0.20)
TOPK_IG    = env_float("VIZ_TOPK_IG", 0.20)
TOPK_MODE  = env_str("VIZ_TOPK_MODE", "slice")  # slice or volume

NEM_MASK_SMOOTH_K     = 5
NEM_MASK_SMOOTH_SIGMA = 1.0

# Shrink area while keeping fade:
# higher Q -> smaller highlighted area
NEM_SHRINK_Q      = 0.70
NEM_SHRINK_POWER  = 1.25

# Prevent the ugly square patch border
NEM_EDGE_FADE_FRAC = 0.12  # fraction of patch dims to fade at borders

SEGMENTATION_HINTS = ("seg-lungs-luna16", "seg-lungs", "segmentation", "seg_")

print(f"[info] dataset='{dataset_name}' model='{model_name}' out='{OUT_VIZ_DIR}'")
print(f"[info] USE_TTA={USE_TTA} USE_TEMP={USE_TEMP} DECISION_THR={DECISION_THR}")
print(f"[info] NEM(remove) shrink: Q={NEM_SHRINK_Q} power={NEM_SHRINK_POWER} edge_fade={NEM_EDGE_FADE_FRAC}")

def to_3d_volume(vol: np.ndarray) -> np.ndarray:
    arr = np.asarray(vol)
    if arr.ndim == 5:
        arr = arr[0]  # [C,D,H,W]
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]  # [D,H,W]
        else:
            arr = arr.mean(axis=0)
    if arr.ndim != 3:
        raise ValueError(f"to_3d_volume: expected (D,H,W), got {arr.shape}")
    return arr.astype(np.float32)

def robust_minmax(x: np.ndarray, p_lo=1.0, p_hi=99.0):
    x = np.asarray(x, dtype=np.float32)
    lo = np.nanpercentile(x, p_lo)
    hi = np.nanpercentile(x, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        lo = float(np.nanmin(x)) if np.isfinite(np.nanmin(x)) else 0.0
        hi = float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else 1.0
        if (hi - lo) < 1e-12:
            hi = lo + 1.0
    return float(lo), float(hi)

def norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(a.min())
    hi = float(a.max())
    if (hi - lo) < 1e-12:
        return np.zeros_like(a, np.float32)
    return (a - lo) / (hi - lo + 1e-9)

def topk_ratio_map(v01: np.ndarray, ratio: float, mode: str = "volume") -> np.ndarray:
    v = np.asarray(v01, dtype=np.float32)
    if ratio <= 0.0:
        return v

    if mode == "volume":
        flat = v.reshape(-1)
        k = max(1, int(ratio * flat.size))
        if k >= flat.size:
            return v
        thr = np.partition(flat, -k)[-k]
        out = np.zeros_like(v, dtype=np.float32)
        out[v >= thr] = v[v >= thr]
        return out

    if mode == "slice":
        out = np.zeros_like(v, dtype=np.float32)
        for d in range(v.shape[0]):
            flat = v[d].reshape(-1)
            k = max(1, int(ratio * flat.size))
            if k >= flat.size:
                out[d] = v[d]
                continue
            thr = np.partition(flat, -k)[-k]
            m = v[d] >= thr
            out[d][m] = v[d][m]
        return out

    raise ValueError("mode must be 'volume' or 'slice'")

def overlay_red_alpha(gray2d: np.ndarray,
                      heat01_2d: np.ndarray,
                      vmin: float,
                      vmax: float,
                      alpha: float,
                      gamma: float,
                      heat_eps: float) -> np.ndarray:
    g = np.asarray(gray2d, dtype=np.float32)
    h = np.asarray(heat01_2d, dtype=np.float32)

    g01 = (g - vmin) / (vmax - vmin + 1e-9)
    g01 = np.clip(g01, 0.0, 1.0)
    rgb = np.repeat(g01[..., None], 3, axis=-1)

    h = np.clip(h, 0.0, 1.0)
    if heat_eps > 0.0:
        h_eff = (h - heat_eps) / (1.0 - heat_eps + 1e-9)
        h_eff = np.clip(h_eff, 0.0, 1.0)
    else:
        h_eff = h

    a = alpha * (h_eff ** gamma)

    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    out = (1.0 - a[..., None]) * rgb + a[..., None] * red
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

def smooth3d_np(vol_zyx: np.ndarray, k: int = 5, sigma: float = 1.0) -> np.ndarray:
    if k <= 1:
        return vol_zyx.astype(np.float32)

    v = torch.as_tensor(vol_zyx[None, None, ...], dtype=torch.float32)  # [1,1,Z,Y,X]
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    g = torch.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * float(sigma) ** 2))
    g = g / g.sum()
    kernel = g.view(1, 1, k, k, k)

    pad = k // 2
    out = torch.nn.functional.conv3d(v, kernel, padding=pad)
    return out[0, 0].cpu().numpy().astype(np.float32)

def _edge_fade_weights(n: int, frac: float) -> np.ndarray:
    if frac <= 0.0:
        return np.ones(n, dtype=np.float32)
    m = max(1, int(round(frac * n)))
    if 2 * m >= n:
        m = max(1, n // 3)

    w = np.ones(n, dtype=np.float32)
    t = np.linspace(0.0, np.pi / 2.0, m, dtype=np.float32)
    ramp = np.sin(t) ** 2
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w

def apply_edge_fade_3d(v: np.ndarray, frac: float) -> np.ndarray:
    v = np.asarray(v, np.float32)
    D, H, W = v.shape
    wz = _edge_fade_weights(D, frac)[:, None, None]
    wy = _edge_fade_weights(H, frac)[None, :, None]
    wx = _edge_fade_weights(W, frac)[None, None, :]
    return (v * (wz * wy * wx)).astype(np.float32)

def best_z_by_topmean(a01_3d: np.ndarray, top_frac: float = 0.02, margin: int = 12) -> int:
    v = np.asarray(a01_3d, np.float32)
    D, H, W = v.shape
    scores = np.zeros(D, dtype=np.float32)
    for d in range(D):
        sl = v[d]
        if H > 2 * margin and W > 2 * margin:
            sl = sl[margin:-margin, margin:-margin]
        flat = sl.reshape(-1)
        k = max(1, int(top_frac * flat.size))
        topk = np.partition(flat, -k)[-k:]
        scores[d] = float(np.mean(topk))
    return int(np.argmax(scores))

def build_uid_to_ct_path(luna_root: Path):
    uid2path = {}
    n_files = 0
    n_skipped = 0
    for mhd_path in luna_root.rglob("*.mhd"):
        n_files += 1
        p_str = str(mhd_path).lower()
        if any(h in p_str for h in SEGMENTATION_HINTS):
            n_skipped += 1
            continue
        uid = mhd_path.stem
        if uid not in uid2path:
            uid2path[uid] = str(mhd_path)
    print(
        f"[info] UID→CT map: {len(uid2path)} entries "
        f"(found {n_files} .mhd, skipped {n_skipped} seg files)"
    )
    return uid2path

def compute_crop_slices(meta, center_world, roi_size, spatial_shape):
    center_world = np.asarray(center_world, dtype=float)
    roi = np.asarray(roi_size, dtype=float)           # (Z,Y,X)
    spatial = np.asarray(spatial_shape, dtype=int)    # (Z,Y,X)

    origin = np.asarray(meta.get("origin", (0, 0, 0)), dtype=float)
    spacing = np.asarray(meta.get("spacing", (1, 1, 1)), dtype=float)
    direction = np.asarray(meta.get("direction", np.eye(3).reshape(-1)), dtype=float).reshape(3, 3)

    A = np.eye(4, dtype=float)
    A[:3, :3] = direction @ np.diag(spacing)
    A[:3, 3] = origin
    invA = np.linalg.inv(A)

    cidx = invA @ np.array([center_world[0], center_world[1], center_world[2], 1.0], dtype=float)
    cidx = cidx[:3]
    cidx_zyx = cidx[[2, 1, 0]]  # (z,y,x)

    start = np.floor(cidx[[2, 1, 0]] - roi / 2.0).astype(int)
    end = start + roi.astype(int)

    pad_l = np.maximum(0, -start)
    pad_r = np.maximum(0, end - spatial)

    s = np.maximum(start, 0)
    e = np.minimum(end, spatial)

    slc = (slice(s[0], e[0]), slice(s[1], e[1]), slice(s[2], e[2]))
    return slc, pad_l.astype(int), pad_r.astype(int), cidx_zyx

def embed_patch_importance_in_full(imp_patch, slc, pad_l, pad_r, full_shape):
    full_imp = np.zeros(full_shape, dtype=np.float32)

    D, H, W = imp_patch.shape
    z0, y0, x0 = pad_l
    z1, y1, x1 = D - pad_r[0], H - pad_r[1], W - pad_r[2]

    imp_unpadded = imp_patch[z0:z1, y0:y1, x0:x1]
    full_imp[slc] = imp_unpadded
    return full_imp

def parse_nem_gen_mask_output(out, x_shape_5d):
    """
    Find the keep-mask tensor in NEM outputs.
    We accept (x_masked, scores, mask) OR (scores, x_masked, mask) OR similar.
    We pick the tensor matching input shape, and closest to [0,1] range.
    """
    if not isinstance(out, (tuple, list)):
        raise RuntimeError("NEM gen_mask did not return a tuple/list.")
    tensors = [t for t in out if torch.is_tensor(t)]
    if not tensors:
        raise RuntimeError("NEM gen_mask returned no tensors.")

    shape_matches = [t for t in tensors if tuple(t.shape) == tuple(x_shape_5d)]
    if not shape_matches:
        return tensors[-1]

    best = None
    best_score = -1e18
    for t in shape_matches:
        tt = t.detach()
        mn = float(tt.min().item())
        mx = float(tt.max().item())
        score = -abs(mn - 0.0) - abs(mx - 1.0)
        if score > best_score:
            best_score = score
            best = t
    return best

def save_scroll_gif(vol3d, imp3d, out_gif, alpha, gamma, heat_eps, fps=10):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[warn] imageio not installed; skipping GIF:", out_gif)
        return

    vol3d = np.asarray(vol3d, np.float32)
    imp3d = np.asarray(imp3d, np.float32)
    vmin, vmax = robust_minmax(vol3d, VOL_CLIP_LO, VOL_CLIP_HI)

    frames = []
    for z in range(vol3d.shape[0]):
        frames.append(
            overlay_red_alpha(vol3d[z], imp3d[z], vmin, vmax, alpha=alpha, gamma=gamma, heat_eps=heat_eps)
        )

    out_gif = str(out_gif)
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=int(fps))
    print("[ok] wrote GIF", out_gif)

def save_nifti(vol, imp, out_dir, voxel_spacing=None):
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
    print("[ok] wrote NIfTI to", out_dir)

def _tta_logits_3d(model, x):
    logits = []
    logits.append(model(x))
    logits.append(model(torch.flip(x, dims=[2])))
    logits.append(model(torch.flip(x, dims=[3])))
    logits.append(model(torch.flip(x, dims=[4])))
    return torch.stack([z.view(-1) for z in logits], dim=0).mean(dim=0)  # [B]

def nem_remove_to_vis(remove01: np.ndarray) -> np.ndarray:
    """
    Simple NEM visualization normalization:
    - Smooth
    - Normalize to [0,1]
    - Soft-shrink by quantile (keeps fading, not binary)
    - Power curve to suppress weak outskirts
    - Edge-fade to prevent square patch boundary
    """
    v = np.asarray(remove01, np.float32)

    # 1) smooth (reduces blocky / square-ish artifacts)
    v = smooth3d_np(v, k=NEM_MASK_SMOOTH_K, sigma=NEM_MASK_SMOOTH_SIGMA)

    # 2) normalize to 0..1
    v = norm01(v)

    # 3) soft shrink (NOT binary)
    thr = float(np.quantile(v.reshape(-1), NEM_SHRINK_Q))
    v = (v - thr) / (1.0 - thr + 1e-9)
    v = np.clip(v, 0.0, 1.0)

    # 4) suppress weak values (still smooth fade)
    v = v ** float(NEM_SHRINK_POWER)

    # 5) fade edges so the embedded patch doesn't draw a square border
    v = apply_edge_fade_3d(v, NEM_EDGE_FADE_FRAC)

    return np.clip(v, 0.0, 1.0).astype(np.float32)

UID2PATH = build_uid_to_ct_path(LUNA_ROOT)

data_obj = CHOSEN_DATASETS[dataset_name]()
train_loader, val_loader = data_obj.get_data()
if val_loader is None:
    raise RuntimeError("Validation loader not found in dataset.")

val_dataset = val_loader.dataset
if hasattr(val_dataset, "indices"):
    full_dataset = val_dataset.dataset
    val_indices = list(val_dataset.indices)
else:
    full_dataset = val_dataset
    val_indices = list(range(len(full_dataset)))

if not hasattr(full_dataset, "rows"):
    raise RuntimeError("Expected dataset with 'rows' attribute (LunaCandidates3DDataset).")

# model
model = CHOSEN_MODELS[model_name]().eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"[info] model device: {device}")

# explainers
use_predicted = True
RISE_explainer = rs3d_atr(model, train_data=None, use_predicted_labels=use_predicted)
IG_explainer   = itg3d_atr(model, train_data=None, use_predicted_labels=use_predicted)

if RISE_N_MASKS not in ("", "None"):
    try:
        RISE_explainer.n_masks = int(RISE_N_MASKS)
        print(f"[info] RISE n_masks overridden to {RISE_explainer.n_masks} via RISE_N_MASKS")
    except ValueError:
        print(f"[warn] RISE_N_MASKS='{RISE_N_MASKS}' is not a valid int; keeping default {RISE_explainer.n_masks}")

ig_n_steps_env = os.environ.get("IG_N_STEPS", "")
if ig_n_steps_env not in ("", "None"):
    try:
        IG_explainer.n_steps = int(ig_n_steps_env)
        print(f"[info] IG n_steps overridden to {IG_explainer.n_steps} via IG_N_STEPS")
    except ValueError:
        print(f"[warn] IG_N_STEPS='{ig_n_steps_env}' invalid; keeping default {IG_explainer.n_steps}")

nem_explainer = NEMT3DMethod(model, train_loader=None, train_or_load=False, device=str(device))
print("[info] Explainers ready: RISE, IG, NEM")

# optional scan filter
positive_scans = None
if USE_SCAN_FILTER and Path(SCAN_PRED_CSV).is_file():
    df_scans = pd.read_csv(SCAN_PRED_CSV)
    if "seriesuid" not in df_scans.columns:
        raise RuntimeError(f"{SCAN_PRED_CSV} missing 'seriesuid' column.")
    df_scans["seriesuid"] = df_scans["seriesuid"].astype(str)

    pos_mask = (df_scans["label"] == SCAN_POS_LABEL)
    if "prob_max" in df_scans.columns:
        pos_mask &= (df_scans["prob_max"] >= SCAN_MIN_PROB)

    positive_scans = set(df_scans.loc[pos_mask, "seriesuid"].tolist())
    print(f"[info] scan filter enabled: {len(positive_scans)} scans pass")
else:
    if USE_SCAN_FILTER:
        print(f"[warn] USE_SCAN_FILTER=1 but CSV not found at {SCAN_PRED_CSV}; disabled")

# collect positives in val
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

print(f"[info] val positives: {len(pos_val_indices)}")
if not pos_val_indices:
    raise RuntimeError("No positive candidates found in validation set.")

# pass 1: predictions
candidate_infos = []
print(f"[info] prediction pass over {len(pos_val_indices)} positives...")
with torch.no_grad():
    for j, gi in enumerate(pos_val_indices):
        X_patch, y = full_dataset[gi]
        label = int(y.item()) if torch.is_tensor(y) else int(y)
        if VIZ_POS_ONLY and label != POS_LABEL:
            continue

        X_sample = X_patch.unsqueeze(0).to(device)  # [1,1,D,H,W]
        logits1d = _tta_logits_3d(model, X_sample) if USE_TTA else model(X_sample).view(-1)
        logit = float(logits1d[0].item())

        if USE_TEMP:
            prob = float(torch.sigmoid(logits1d / float(TEMPERATURE))[0].item())
        else:
            prob = float(torch.sigmoid(logits1d)[0].item())

        pred = int(prob >= DECISION_THR)

        row = full_dataset.rows[gi]
        uid = str(row.get("uid", row.get("seriesuid", f"sample-{gi:06d}")))

        candidate_infos.append({"idx": gi, "uid": uid, "label": label, "pred": pred, "prob": prob, "logit": logit})

        if VIZ_PROGRESS_EVERY > 0 and (j + 1) % VIZ_PROGRESS_EVERY == 0:
            print(f"[progress] {j+1}/{len(pos_val_indices)} scanned")

print(f"[info] predictions collected: {len(candidate_infos)}")

# selection
pool = candidate_infos
if REQUIRE_TP:
    tp_pool = [c for c in pool if c["label"] == POS_LABEL and c["pred"] == POS_LABEL]
    if tp_pool:
        pool = tp_pool
        print(f"[info] TP pool: {len(pool)}")
    else:
        print("[warn] No TPs found; using label-only positives")

pool2 = [c for c in pool if c["prob"] >= VIZ_MIN_PROB] if VIZ_MIN_PROB > 0 else pool
if not pool2:
    print("[warn] VIZ_MIN_PROB filtered everything; falling back to pool")
    pool2 = pool

strong = sorted(pool2, key=lambda c: c["prob"], reverse=True)[:N_VIZ_SAMPLES]
print(f"[info] selected {len(strong)} candidates for viz")

# full-volume loader
full_loader = CachedLoadPreprocessd(
    keys=("image",),
    spacing=SPACING_PATCH,
    a_min=-1000,
    a_max=400,
    max_cache=2,
    allow_missing_keys=True,
)

saved = 0

for info in strong:
    gi   = info["idx"]
    uid  = info["uid"]
    prob = info["prob"]

    X_patch, y = full_dataset[gi]
    label = int(y.item()) if torch.is_tensor(y) else int(y)

    X_sample = X_patch.unsqueeze(0).to(device)  # [1,1,D,H,W]
    patch_x3d = to_3d_volume(X_sample.detach().cpu().numpy())  # (D,H,W)
    patch_size = patch_x3d.shape

    print(f"\n[viz] uid={uid} label={label} pred={info['pred']} prob={prob:.3f} logit={info['logit']:.3f}")

    rise_attr = call_gen_attr(RISE_explainer, X_sample)
    rise3d = norm01(to_3d_volume(rise_attr))

    ig_attr = call_gen_attr(IG_explainer, X_sample)
    ig3d = norm01(np.abs(to_3d_volume(ig_attr)))

    rise3d = topk_ratio_map(rise3d, TOPK_RISE, mode=TOPK_MODE)
    ig3d   = topk_ratio_map(ig3d,   TOPK_IG,   mode=TOPK_MODE)

    out = nem_explainer.gen_mask(X_sample)
    keep_t = parse_nem_gen_mask_output(out, X_sample.shape)
    keep3d = np.clip(to_3d_volume(keep_t.detach().cpu().numpy()), 0.0, 1.0)

    # sanity print
    with torch.no_grad():
        p0 = float(torch.sigmoid(model(X_sample)).view(-1)[0].item())
        pm = float(torch.sigmoid(model(X_sample * keep_t)).view(-1)[0].item())
        pc = float(torch.sigmoid(model(X_sample * (1.0 - keep_t))).view(-1)[0].item())
    print(f"[nem] prob orig={p0:.3f} | x*keep={pm:.3f} | x*(1-keep)={pc:.3f}")

    remove3d = 1.0 - keep3d
    nem_vis_patch = nem_remove_to_vis(remove3d)

    row = full_dataset.rows[gi]
    img_path = UID2PATH.get(uid)
    if img_path is None:
        print(f"[warn] uid={uid}: CT not found under {LUNA_ROOT}")
        continue

    d_full = full_loader({"image": img_path})
    full_vol = d_full["image"]
    if torch.is_tensor(full_vol):
        full_vol = full_vol.detach().cpu().numpy()
    full_vol3d = full_vol[0]  # (Z,Y,X)
    full_meta = d_full.get("image_meta_dict", {})
    spatial_shape = full_vol3d.shape

    center_world = np.array([row["x"], row["y"], row["z"]], dtype=float)
    slc, pad_l, pad_r, cidx_zyx = compute_crop_slices(
        meta=full_meta,
        center_world=center_world,
        roi_size=patch_size,
        spatial_shape=spatial_shape,
    )

    full_rise = embed_patch_importance_in_full(rise3d, slc, pad_l, pad_r, spatial_shape)
    full_ig   = embed_patch_importance_in_full(ig3d,   slc, pad_l, pad_r, spatial_shape)
    full_nem  = embed_patch_importance_in_full(nem_vis_patch, slc, pad_l, pad_r, spatial_shape)

    z_roi_center = int(np.clip(int(round(cidx_zyx[0])), 0, spatial_shape[0] - 1))
    z_rise_best = best_z_by_topmean(full_rise) if np.any(full_rise > 0) else z_roi_center
    z_ig_best   = best_z_by_topmean(full_ig)   if np.any(full_ig > 0)   else z_roi_center
    z_nem_best  = best_z_by_topmean(full_nem)  if np.any(full_nem > 0)  else z_roi_center

    vmin, vmax = robust_minmax(full_vol3d, VOL_CLIP_LO, VOL_CLIP_HI)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].imshow(overlay_red_alpha(full_vol3d[z_roi_center], full_rise[z_roi_center], vmin, vmax,
                                       alpha=ALPHA_BASE, gamma=GAMMA_BASE, heat_eps=HEAT_EPS))
    axs[0, 1].imshow(overlay_red_alpha(full_vol3d[z_roi_center], full_ig[z_roi_center], vmin, vmax,
                                       alpha=ALPHA_IG, gamma=GAMMA_IG, heat_eps=HEAT_EPS_IG))
    axs[0, 2].imshow(overlay_red_alpha(full_vol3d[z_roi_center], full_nem[z_roi_center], vmin, vmax,
                                       alpha=ALPHA_BASE, gamma=GAMMA_BASE, heat_eps=HEAT_EPS))

    axs[0, 0].set_title(f"RISE — ROI z={z_roi_center}")
    axs[0, 1].set_title(f"IG — ROI z={z_roi_center}")
    axs[0, 2].set_title(f"NEM(remove) — ROI z={z_roi_center}")

    axs[1, 0].imshow(overlay_red_alpha(full_vol3d[z_rise_best], full_rise[z_rise_best], vmin, vmax,
                                       alpha=ALPHA_BASE, gamma=GAMMA_BASE, heat_eps=HEAT_EPS))
    axs[1, 1].imshow(overlay_red_alpha(full_vol3d[z_ig_best], full_ig[z_ig_best], vmin, vmax,
                                       alpha=ALPHA_IG, gamma=GAMMA_IG, heat_eps=HEAT_EPS_IG))
    axs[1, 2].imshow(overlay_red_alpha(full_vol3d[z_nem_best], full_nem[z_nem_best], vmin, vmax,
                                       alpha=ALPHA_BASE, gamma=GAMMA_BASE, heat_eps=HEAT_EPS))

    axs[1, 0].set_title(f"RISE — best z={z_rise_best}")
    axs[1, 1].set_title(f"IG — best z={z_ig_best}")
    axs[1, 2].set_title(f"NEM(remove) — best z={z_nem_best}")

    for r in range(2):
        for c in range(3):
            axs[r, c].axis("off")

    fig.suptitle(f"UID={uid} | label={label} | prob={prob:.3f} | logit={info['logit']:.3f}", y=0.98)
    plt.tight_layout()

    out_png = OUT_VIZ_DIR / f"{uid}.full-axial.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")

    if MAKE_GIFS:
        gif_dir = OUT_VIZ_DIR / "gifs_full" / uid
        gif_dir.mkdir(parents=True, exist_ok=True)

        save_scroll_gif(full_vol3d, full_rise, gif_dir / f"{uid}_rise_full_axial.gif",
                        alpha=ALPHA_BASE, gamma=GAMMA_BASE, heat_eps=HEAT_EPS, fps=10)
        save_scroll_gif(full_vol3d, full_ig, gif_dir / f"{uid}_ig_full_axial.gif",
                        alpha=ALPHA_IG, gamma=GAMMA_IG, heat_eps=HEAT_EPS_IG, fps=10)
        save_scroll_gif(full_vol3d, full_nem, gif_dir / f"{uid}_nem_full_axial.gif",
                        alpha=ALPHA_BASE, gamma=GAMMA_BASE, heat_eps=HEAT_EPS, fps=10)

    if MAKE_NIFTIS:
        nifti_dir = OUT_VIZ_DIR / "nifti_full" / uid
        spacing = full_meta.get("spacing")
        save_nifti(full_vol3d, full_nem, str(nifti_dir), voxel_spacing=spacing)

    saved += 1

print(f"\n[done] saved {saved} samples under {OUT_VIZ_DIR}")
