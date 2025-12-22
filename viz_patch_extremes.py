import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from exp_utils.luna_monai3d import CachedLoadPreprocessd, SPACING_PATCH
except Exception as e:
    CachedLoadPreprocessd = None
    SPACING_PATCH = None
    print(f"[warn] context crop disabled (could not import luna_monai3d): {e}")

dataset_name = os.environ.get("NEM_DATASETS", "luna3d").split(",")[0]
model_name   = os.environ.get("NEM_MODELS", "phase3_3d").split(",")[0]

BASE_DIR = Path("experiments") / dataset_name / model_name / "eval_detailed_examples"
TOP_DIR  = BASE_DIR / "top_examples"
CSV_PATH = BASE_DIR / "per_sample_scores.csv"

OUT_DIR = Path("experiments") / dataset_name / model_name / "viz_patch_extremes_ig_nem_rise"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIZ_AXIS  = os.environ.get("VIZ_AXIS", "ax")                # ax / co / sa
VIZ_MODE  = os.environ.get("VIZ_MODE", "shared_best")
VIZ_ALPHA = float(os.environ.get("VIZ_ALPHA", "0.45"))

IG_TOPK   = float(os.environ.get("VIZ_TOPK_IG",   "0.00"))
NEM_TOPK  = float(os.environ.get("VIZ_TOPK_NEM",  "0.00"))
RISE_TOPK = float(os.environ.get("VIZ_TOPK_RISE", "0.00"))

IG_ALPHA = float(os.environ.get("VIZ_ALPHA_IG", str(VIZ_ALPHA)))
IG_GAMMA = float(os.environ.get("VIZ_GAMMA_IG", "1.35"))
IG_EPS   = float(os.environ.get("VIZ_EPS_IG",   "0.10"))
IG_SHRINK_Q     = float(os.environ.get("VIZ_IG_SHRINK_Q", "0.80"))
IG_SHRINK_POWER = float(os.environ.get("VIZ_IG_SHRINK_POWER", "1.15"))

NEM_ALPHA = float(os.environ.get("VIZ_ALPHA_NEM", "0.55"))
NEM_GAMMA = float(os.environ.get("VIZ_GAMMA_NEM", "1.80"))
NEM_EPS   = float(os.environ.get("VIZ_EPS_NEM",   "0.08"))

RISE_ALPHA = float(os.environ.get("VIZ_ALPHA_RISE", "0.55"))
RISE_GAMMA = float(os.environ.get("VIZ_GAMMA_RISE", "1.35"))
RISE_EPS   = float(os.environ.get("VIZ_EPS_RISE",   "0.08"))
RISE_SHRINK_Q     = float(os.environ.get("VIZ_RISE_SHRINK_Q", "0.70"))
RISE_SHRINK_POWER = float(os.environ.get("VIZ_RISE_SHRINK_POWER", "1.25"))
RISE_SMOOTH_K     = int(os.environ.get("VIZ_RISE_SMOOTH_K", "3"))
RISE_SMOOTH_SIGMA = float(os.environ.get("VIZ_RISE_SMOOTH_SIGMA", "0.9"))

VOL_CLIP  = (float(os.environ.get("VIZ_VOL_PCT_LO", "1.0")),
             float(os.environ.get("VIZ_VOL_PCT_HI", "99.5")))
HEAT_CLIP = (float(os.environ.get("VIZ_HEAT_PCT_LO", "1.0")),
             float(os.environ.get("VIZ_HEAT_PCT_HI", "99.5")))

VIZ_CONTEXT        = int(os.environ.get("VIZ_CONTEXT", "1"))
VIZ_CONTEXT_FACTOR = float(os.environ.get("VIZ_CONTEXT_FACTOR", "2.0"))

VIZ_FILTER_PRED_POS = int(os.environ.get("VIZ_FILTER_PRED_POS", "0"))
VIZ_DECISION_THR    = float(os.environ.get("VIZ_DECISION_THR", "0.003"))

print(f"[info] dataset={dataset_name} model={model_name}")
print(f"[info] TOP_DIR={TOP_DIR}")
print(f"[info] OUT_DIR={OUT_DIR}")
print(f"[info] VIZ_AXIS={VIZ_AXIS} VIZ_MODE={VIZ_MODE}")
print(f"[info] context={VIZ_CONTEXT} factor={VIZ_CONTEXT_FACTOR}")
print(f"[info] topk: IG={IG_TOPK} NEM={NEM_TOPK} RISE={RISE_TOPK}")

NEM_MASK_SMOOTH_K     = 5
NEM_MASK_SMOOTH_SIGMA = 1.0
NEM_SHRINK_Q          = 0.70
NEM_SHRINK_POWER      = 1.25
NEM_EDGE_FADE_FRAC    = 0.12

SEGMENTATION_HINTS = ("seg-lungs-luna16", "seg-lungs", "segmentation", "seg_")

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

def to_3d(vol: np.ndarray) -> np.ndarray:
    a = np.asarray(vol)
    if a.ndim == 5:
        a = a[0, 0]
    elif a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got {a.shape}")
    return a.astype(np.float32)

def norm01_percentile(a: np.ndarray, clip=(1.0, 99.5)) -> np.ndarray:
    a = np.asarray(a, np.float32)
    if a.size == 0:
        return a
    lo, hi = np.percentile(a, clip)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        a = np.clip(a, lo, hi)
    amin, amax = float(a.min()), float(a.max())
    rng = amax - amin
    if not np.isfinite(rng) or rng < 1e-9:
        return np.zeros_like(a, np.float32)
    return (a - amin) / (rng + 1e-9)

def norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    amin, amax = float(a.min()), float(a.max())
    rng = amax - amin
    if not np.isfinite(rng) or rng < 1e-9:
        return np.zeros_like(a, np.float32)
    return (a - amin) / (rng + 1e-9)

def soft_shrink01(v01: np.ndarray, q: float, power: float) -> np.ndarray:
    v = np.asarray(v01, np.float32)
    q = float(np.clip(q, 0.0, 0.9999))
    thr = float(np.quantile(v.reshape(-1), q))
    v = (v - thr) / (1.0 - thr + 1e-9)
    v = np.clip(v, 0.0, 1.0)
    v = v ** float(power)
    return v

def apply_topk_frac(v01: np.ndarray, frac: float) -> np.ndarray:
    """Keep top frac voxels by value (0..1). frac<=0 disables."""
    frac = float(frac)
    if frac <= 0.0:
        return np.asarray(v01, np.float32)
    v = np.asarray(v01, np.float32)
    flat = v.reshape(-1)
    if flat.size == 0:
        return v
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac >= 1.0:
        return v
    thr = float(np.quantile(flat, 1.0 - frac))
    out = np.where(v >= thr, v, 0.0).astype(np.float32)
    mx = float(out.max())
    if mx > 1e-9:
        out = out / mx
    return out

def overlay_red_alpha01(gray01_2d: np.ndarray,
                        heat01_2d: np.ndarray,
                        alpha: float,
                        gamma: float,
                        heat_eps: float) -> np.ndarray:
    g = np.clip(np.asarray(gray01_2d, np.float32), 0.0, 1.0)
    h = np.clip(np.asarray(heat01_2d, np.float32), 0.0, 1.0)

    if heat_eps > 0.0:
        h = (h - heat_eps) / (1.0 - heat_eps + 1e-9)
        h = np.clip(h, 0.0, 1.0)

    a = alpha * (h ** gamma)

    rgb = np.repeat(g[..., None], 3, axis=-1)
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

def nem_remove_to_vis(remove01: np.ndarray) -> np.ndarray:
    v = np.asarray(remove01, np.float32)
    v = smooth3d_np(v, k=NEM_MASK_SMOOTH_K, sigma=NEM_MASK_SMOOTH_SIGMA)
    v = norm01(v)

    thr = float(np.quantile(v.reshape(-1), NEM_SHRINK_Q))
    v = (v - thr) / (1.0 - thr + 1e-9)
    v = np.clip(v, 0.0, 1.0)

    v = v ** float(NEM_SHRINK_POWER)
    v = apply_edge_fade_3d(v, NEM_EDGE_FADE_FRAC)
    return np.clip(v, 0.0, 1.0).astype(np.float32)

def prepare_nem_heat_3d(nem3d: np.ndarray) -> tuple[np.ndarray, str]:
    v = np.asarray(nem3d, np.float32)
    mn, mx = float(v.min()), float(v.max())

    if mn >= -0.05 and mx <= 1.05:
        m = np.clip(v, 0.0, 1.0)
        if float(m.mean()) > 0.5:
            remove01 = 1.0 - m
            kind = "keep->remove"
        else:
            remove01 = m
            kind = "remove"
    else:
        remove01 = norm01_percentile(np.maximum(v, 0.0), HEAT_CLIP)
        kind = "attr->posnorm"

    vis = nem_remove_to_vis(remove01)
    return vis, kind

def prepare_rise_heat_3d(rise3d: np.ndarray) -> tuple[np.ndarray, str]:
    """
    RISE usually behaves like a smooth positive relevance map.
    We normalise + optionally smooth + soft-shrink for display.
    """
    v = np.asarray(rise3d, np.float32)
    mn, mx = float(v.min()), float(v.max())

    if mn >= -0.05 and mx <= 1.05:
        h01 = np.clip(v, 0.0, 1.0)
        kind = "masklike"
    else:
        h01 = norm01_percentile(np.abs(v), HEAT_CLIP)
        kind = "attr->absnorm"

    if RISE_SMOOTH_K > 1:
        h01 = smooth3d_np(h01, k=int(RISE_SMOOTH_K), sigma=float(RISE_SMOOTH_SIGMA))
        h01 = norm01(h01)

    h01 = soft_shrink01(h01, q=RISE_SHRINK_Q, power=RISE_SHRINK_POWER)
    return np.clip(h01, 0.0, 1.0), kind

def proj2d_mip(vol3d: np.ndarray, axis: str) -> np.ndarray:
    v = np.asarray(vol3d, np.float32)
    if axis == "ax":
        return v.max(axis=0)
    if axis == "co":
        return v.max(axis=1)
    if axis == "sa":
        return v.max(axis=2)
    raise ValueError(f"Unknown axis: {axis}")

def slice2d(vol3d: np.ndarray, axis: str, idx: int) -> np.ndarray:
    v = np.asarray(vol3d, np.float32)
    if axis == "ax":
        return v[idx, :, :]
    if axis == "co":
        return v[:, idx, :]
    if axis == "sa":
        return v[:, :, idx]
    raise ValueError(f"Unknown axis: {axis}")

def best_index_by_topmean(v01_3d: np.ndarray, axis: str, top_frac: float = 0.02) -> int:
    v = np.asarray(v01_3d, np.float32)

    if axis == "ax":
        n = v.shape[0]
        get_sl = lambda i: v[i, :, :]
    elif axis == "co":
        n = v.shape[1]
        get_sl = lambda i: v[:, i, :]
    elif axis == "sa":
        n = v.shape[2]
        get_sl = lambda i: v[:, :, i]
    else:
        raise ValueError(f"Unknown axis: {axis}")

    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sl = get_sl(i)
        flat = sl.reshape(-1)
        k = max(1, int(top_frac * flat.size))
        topk = np.partition(flat, -k)[-k:]
        scores[i] = float(np.mean(topk))
    return int(np.argmax(scores))

def get_prob_from_meta(meta: dict) -> float | None:
    for k in ("pred_prob", "prob_pos"):
        if k in meta and meta[k] is not None:
            try:
                return float(meta[k])
            except Exception:
                pass
    return None

def is_pred_positive_from_prob(prob: float | None, thr: float) -> bool:
    if prob is None:
        return False
    return prob >= float(thr)

UID2PATH = {}
FULL_LOADER = None

if VIZ_CONTEXT and CachedLoadPreprocessd is not None:
    LUNA_ROOT = Path(os.environ.get("LUNA_ROOT", "Luna16"))
    UID2PATH = build_uid_to_ct_path(LUNA_ROOT)
    FULL_LOADER = CachedLoadPreprocessd(
        keys=("image",),
        spacing=SPACING_PATCH,
        a_min=-1000,
        a_max=400,
        max_cache=2,
        allow_missing_keys=True,
    )
else:
    if VIZ_CONTEXT:
        print("[warn] VIZ_CONTEXT=1 but CachedLoadPreprocessd unavailable; running patch-only.")

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

def extract_crop_padded(full_vol3d: np.ndarray, meta: dict, center_world, roi_size_zyx, pad_val=-1000.0):
    slc, pad_l, pad_r, _ = compute_crop_slices(meta, center_world, roi_size_zyx, full_vol3d.shape)
    crop = full_vol3d[slc]
    crop = np.pad(
        crop,
        pad_width=((pad_l[0], pad_r[0]), (pad_l[1], pad_r[1]), (pad_l[2], pad_r[2])),
        mode="constant",
        constant_values=float(pad_val),
    )
    return crop.astype(np.float32), slc, pad_l, pad_r

def embed_patch_heat_into_crop(patch_heat3d: np.ndarray,
                              slc_small, pad_l_small, pad_r_small,
                              slc_big, pad_l_big,
                              big_shape):
    out = np.zeros(big_shape, dtype=np.float32)

    D, H, W = patch_heat3d.shape
    z0, y0, x0 = pad_l_small
    z1, y1, x1 = D - pad_r_small[0], H - pad_r_small[1], W - pad_r_small[2]
    heat_unpad = patch_heat3d[z0:z1, y0:y1, x0:x1]

    rz0 = int(pad_l_big[0] + (slc_small[0].start - slc_big[0].start))
    ry0 = int(pad_l_big[1] + (slc_small[1].start - slc_big[1].start))
    rx0 = int(pad_l_big[2] + (slc_small[2].start - slc_big[2].start))

    rz1 = rz0 + heat_unpad.shape[0]
    ry1 = ry0 + heat_unpad.shape[1]
    rx1 = rx0 + heat_unpad.shape[2]

    cz0 = max(0, rz0); cy0 = max(0, ry0); cx0 = max(0, rx0)
    cz1 = min(out.shape[0], rz1); cy1 = min(out.shape[1], ry1); cx1 = min(out.shape[2], rx1)

    hz0 = cz0 - rz0; hy0 = cy0 - ry0; hx0 = cx0 - rx0
    hz1 = hz0 + (cz1 - cz0); hy1 = hy0 + (cy1 - cy0); hx1 = hx0 + (cx1 - cx0)

    if cz1 > cz0 and cy1 > cy0 and cx1 > cx0:
        out[cz0:cz1, cy0:cy1, cx0:cx1] = heat_unpad[hz0:hz1, hy0:hy1, hx0:hx1]
    return out

def _load_meta_from_npz(d) -> dict:
    meta_raw = d["meta_json"]
    if isinstance(meta_raw, np.ndarray):
        s = meta_raw[0]
        if hasattr(s, "item"):
            s = s.item()
        meta_str = s
    else:
        meta_str = meta_raw.item() if hasattr(meta_raw, "item") else str(meta_raw)
    meta = json.loads(meta_str)
    return meta

def _get_first_key(d, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in d.files:
            return k
    return None

def load_npz(npz_path: Path) -> dict:
    d = np.load(str(npz_path), allow_pickle=True)

    x_key = _get_first_key(d, ("x_5d", "x", "image_5d", "img_5d"))
    if x_key is None:
        raise RuntimeError(f"{npz_path} missing x volume (expected x_5d etc).")

    ig_key = _get_first_key(d, ("ig_attr_5d", "ig_5d", "ig_attr"))
    nem_key = _get_first_key(d, ("nem_attr_5d", "nem_5d", "nem_attr"))
    rise_key = _get_first_key(d, ("rise_attr_5d", "rise_5d", "rise_attr", "rise_map_5d"))

    x3d   = to_3d(d[x_key])
    ig3d  = to_3d(d[ig_key])  if ig_key  is not None else np.zeros_like(x3d, np.float32)
    nem3d = to_3d(d[nem_key]) if nem_key is not None else np.zeros_like(x3d, np.float32)
    rise3d = to_3d(d[rise_key]) if rise_key is not None else np.zeros_like(x3d, np.float32)

    meta = _load_meta_from_npz(d)
    meta.setdefault("gi", None)
    meta.setdefault("label", None)
    meta.setdefault("uid", meta.get("seriesuid", None))
    meta.setdefault("pred_label", None)
    meta.setdefault("pred_prob", None)
    meta.setdefault("prob_pos", None)

    return {
        "npz_path": str(npz_path),
        "x3d": x3d,
        "ig3d": ig3d,
        "nem3d": nem3d,
        "rise3d": rise3d,
        "meta": meta,
        "keys": {"x": x_key, "ig": ig_key, "nem": nem_key, "rise": rise_key},
    }

def find_first_npz_in_folder(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.npz"))
    if not files:
        return None

    if not VIZ_FILTER_PRED_POS:
        return files[0]

    for p in files:
        try:
            d = np.load(str(p), allow_pickle=True)
            meta = _load_meta_from_npz(d)
            prob = get_prob_from_meta(meta)
            if is_pred_positive_from_prob(prob, VIZ_DECISION_THR):
                return p
        except Exception:
            continue
    return None

def prepare_volumes(example: dict):
    """
    Returns:
      base_vol3d, ig_vol3d, nem_vol3d, rise_vol3d, meta_kinds, used_context(bool)
    """
    x3d    = example["x3d"]
    ig3d   = example["ig3d"]
    nem3d  = example["nem3d"]
    rise3d = example["rise3d"]
    meta   = example["meta"]

    # IG heat
    ig_heat3d = norm01_percentile(np.abs(ig3d), HEAT_CLIP)
    ig_heat3d = soft_shrink01(ig_heat3d, q=IG_SHRINK_Q, power=IG_SHRINK_POWER)
    ig_heat3d = np.clip(ig_heat3d, 0.0, 1.0)
    ig_heat3d = apply_topk_frac(ig_heat3d, IG_TOPK)

    # NEM heat
    nem_vis3d, nem_kind = prepare_nem_heat_3d(nem3d)
    nem_vis3d = apply_topk_frac(nem_vis3d, NEM_TOPK)

    # RISE heat
    rise_vis3d, rise_kind = prepare_rise_heat_3d(rise3d)
    rise_vis3d = apply_topk_frac(rise_vis3d, RISE_TOPK)

    base_vol3d = x3d
    used_context = False

    # optional context crop
    if VIZ_CONTEXT and FULL_LOADER is not None:
        uid = meta.get("uid") or meta.get("seriesuid")
        has_xyz = all(k in meta for k in ("x", "y", "z"))

        if uid is not None and has_xyz:
            img_path = UID2PATH.get(str(uid))
            if img_path is not None:
                d_full = FULL_LOADER({"image": img_path})
                full_vol = d_full["image"]
                if torch.is_tensor(full_vol):
                    full_vol = full_vol.detach().cpu().numpy()
                full_vol3d = full_vol[0]  # (Z,Y,X)
                full_meta = d_full.get("image_meta_dict", {})

                center_world = np.array([meta["x"], meta["y"], meta["z"]], dtype=float)

                patch_size = np.array(x3d.shape, dtype=int)  # (Z,Y,X)
                ctx_size = np.maximum(8, np.round(patch_size * float(VIZ_CONTEXT_FACTOR)).astype(int))

                ctx_vol3d, slc_big, pad_l_big, _ = extract_crop_padded(
                    full_vol3d, full_meta, center_world, roi_size_zyx=ctx_size, pad_val=-1000.0
                )

                slc_small, pad_l_small, pad_r_small, _ = compute_crop_slices(
                    full_meta, center_world, patch_size, full_vol3d.shape
                )

                ig_ctx   = embed_patch_heat_into_crop(ig_heat3d,   slc_small, pad_l_small, pad_r_small, slc_big, pad_l_big, ctx_vol3d.shape)
                nem_ctx  = embed_patch_heat_into_crop(nem_vis3d,   slc_small, pad_l_small, pad_r_small, slc_big, pad_l_big, ctx_vol3d.shape)
                rise_ctx = embed_patch_heat_into_crop(rise_vis3d,  slc_small, pad_l_small, pad_r_small, slc_big, pad_l_big, ctx_vol3d.shape)

                base_vol3d = ctx_vol3d
                ig_heat3d  = ig_ctx
                nem_vis3d  = nem_ctx
                rise_vis3d = rise_ctx
                used_context = True

    kinds = {"nem": nem_kind, "rise": rise_kind}
    return base_vol3d, ig_heat3d, nem_vis3d, rise_vis3d, kinds, used_context

def make_view(base_vol3d: np.ndarray, heat_vol3d: np.ndarray, idx: int | None):
    """
    Returns: base2d, heat2d

    If idx is None and VIZ_MODE == mip => do MIP.
    Otherwise slice at idx.
    """
    if idx is None and VIZ_MODE == "mip":
        base2d = proj2d_mip(base_vol3d, VIZ_AXIS)
        heat2d = proj2d_mip(heat_vol3d, VIZ_AXIS)
    else:
        assert idx is not None
        base2d = slice2d(base_vol3d, VIZ_AXIS, idx)
        heat2d = slice2d(heat_vol3d, VIZ_AXIS, idx)

    base2d = norm01_percentile(base2d, VOL_CLIP)
    heat2d = np.clip(heat2d, 0.0, 1.0)
    return base2d, heat2d

def choose_index(base_vol3d, ig_vol3d, nem_vol3d, rise_vol3d) -> dict:
    """
    Returns dict with indices per method + shared index.
    """
    if VIZ_MODE == "mip":
        return {"shared": None, "ig": None, "nem": None, "rise": None}

    # center idx
    if VIZ_AXIS == "ax":
        center_idx = base_vol3d.shape[0] // 2
    elif VIZ_AXIS == "co":
        center_idx = base_vol3d.shape[1] // 2
    else:
        center_idx = base_vol3d.shape[2] // 2

    if VIZ_MODE == "center":
        return {"shared": center_idx, "ig": center_idx, "nem": center_idx, "rise": center_idx}

    if VIZ_MODE == "best":
        return {
            "shared": None,
            "ig": best_index_by_topmean(np.clip(ig_vol3d, 0.0, 1.0), axis=VIZ_AXIS),
            "nem": best_index_by_topmean(np.clip(nem_vol3d, 0.0, 1.0), axis=VIZ_AXIS),
            "rise": best_index_by_topmean(np.clip(rise_vol3d, 0.0, 1.0), axis=VIZ_AXIS),
        }

    # shared_best (default)
    comb = np.maximum(np.maximum(ig_vol3d, nem_vol3d), rise_vol3d)
    shared = best_index_by_topmean(np.clip(comb, 0.0, 1.0), axis=VIZ_AXIS)
    return {"shared": shared, "ig": shared, "nem": shared, "rise": shared}

def render_side_by_side(example: dict, out_png: Path, case_title: str):
    base_vol3d, ig_vol3d, nem_vol3d, rise_vol3d, kinds, used_context = prepare_volumes(example)
    idxs = choose_index(base_vol3d, ig_vol3d, nem_vol3d, rise_vol3d)

    idx_ig = idxs["ig"]; idx_nem = idxs["nem"]; idx_rise = idxs["rise"]
    base2d_ig,  ig2d   = make_view(base_vol3d, ig_vol3d,   idx_ig)
    base2d_nem, nem2d  = make_view(base_vol3d, nem_vol3d,  idx_nem)
    base2d_rise, rise2d = make_view(base_vol3d, rise_vol3d, idx_rise)

    img_ig   = overlay_red_alpha01(base2d_ig,   ig2d,   alpha=IG_ALPHA,   gamma=IG_GAMMA,   heat_eps=IG_EPS)
    img_nem  = overlay_red_alpha01(base2d_nem,  nem2d,  alpha=NEM_ALPHA,  gamma=NEM_GAMMA,  heat_eps=NEM_EPS)
    img_rise = overlay_red_alpha01(base2d_rise, rise2d, alpha=RISE_ALPHA, gamma=RISE_GAMMA, heat_eps=RISE_EPS)

    meta = example["meta"]
    gi = meta.get("gi", "NA")
    uid = meta.get("uid", meta.get("seriesuid", "NA"))
    prob = meta.get("pred_prob", meta.get("prob_pos", "NA"))

    fig, axs = plt.subplots(1, 3, figsize=(17, 6))

    axs[0].imshow(img_ig);   axs[0].axis("off")
    axs[1].imshow(img_nem);  axs[1].axis("off")
    axs[2].imshow(img_rise); axs[2].axis("off")

    axs[0].set_title("IG")
    axs[1].set_title(f"NEM(remove)\n{kinds['nem']}")
    axs[2].set_title(f"RISE\n{kinds['rise']}")

    idx_str = "mip" if idxs["shared"] is None else str(idxs["shared"])
    fig.suptitle(
        f"{case_title}\n"
        f"gi={gi} | uid={uid} | pred_prob={prob}\n"
        f"axis={VIZ_AXIS} mode={VIZ_MODE} idx={idx_str} | context={'yes' if used_context else 'no'} x{VIZ_CONTEXT_FACTOR}\n"
        f"topk(IG/NEM/RISE)={IG_TOPK}/{NEM_TOPK}/{RISE_TOPK}",
        y=0.98
    )

    plt.tight_layout()
    fig.savefig(str(out_png), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")

def render_montage(examples: list[tuple[str, dict]], out_png: Path):
    n = len(examples)
    fig, axs = plt.subplots(n, 3, figsize=(17, 4.8 * n))
    if n == 1:
        axs = np.array([axs])

    for r, (row_title, ex) in enumerate(examples):
        base_vol3d, ig_vol3d, nem_vol3d, rise_vol3d, kinds, used_context = prepare_volumes(ex)
        idxs = choose_index(base_vol3d, ig_vol3d, nem_vol3d, rise_vol3d)

        idx = idxs["shared"]
        base2d_ig,  ig2d    = make_view(base_vol3d, ig_vol3d,   idxs["ig"])
        base2d_nem, nem2d   = make_view(base_vol3d, nem_vol3d,  idxs["nem"])
        base2d_rise, rise2d = make_view(base_vol3d, rise_vol3d, idxs["rise"])

        img_ig   = overlay_red_alpha01(base2d_ig,   ig2d,   alpha=IG_ALPHA,   gamma=IG_GAMMA,   heat_eps=IG_EPS)
        img_nem  = overlay_red_alpha01(base2d_nem,  nem2d,  alpha=NEM_ALPHA,  gamma=NEM_GAMMA,  heat_eps=NEM_EPS)
        img_rise = overlay_red_alpha01(base2d_rise, rise2d, alpha=RISE_ALPHA, gamma=RISE_GAMMA, heat_eps=RISE_EPS)

        meta = ex["meta"]
        gi = meta.get("gi", "NA")
        uid = meta.get("uid", meta.get("seriesuid", "NA"))
        prob = meta.get("pred_prob", meta.get("prob_pos", "NA"))
        idx_str = "mip" if idx is None else str(idx)

        axs[r, 0].imshow(img_ig);   axs[r, 0].axis("off")
        axs[r, 1].imshow(img_nem);  axs[r, 1].axis("off")
        axs[r, 2].imshow(img_rise); axs[r, 2].axis("off")

        axs[r, 0].set_title(f"{row_title}\nIG | gi={gi} uid={uid}\nidx={idx_str} prob={prob}")
        axs[r, 1].set_title(f"NEM(remove) ({kinds['nem']})")
        axs[r, 2].set_title(f"RISE ({kinds['rise']})")

    fig.suptitle(
        f"Patch examples: IG vs NEM(remove) vs RISE (axis={VIZ_AXIS}, mode={VIZ_MODE})",
        y=0.995
    )
    plt.tight_layout()
    fig.savefig(str(out_png), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")

def pick_npz_from_folder(folder_name: str) -> Path:
    folder = TOP_DIR / folder_name
    p = find_first_npz_in_folder(folder)
    if p is not None:
        return p
    # fallback: if folder empty
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        if "gi" in df.columns:
            gi = int(df.iloc[0]["gi"])
            hits = sorted(TOP_DIR.rglob(f"*gi{gi}.npz"))
            if hits:
                return hits[0]
    raise RuntimeError(f"No npz files found in {folder} (and CSV fallback failed).")

cases = [
    ("IG best localisation", "ig_best_pos"),
    ("IG worst localisation", "ig_worst_pos"),
    ("NEM best localisation", "nem_best_pos"),
    ("NEM worst localisation", "nem_worst_pos"),
    ("RISE best localisation", "rise_best_pos"),
    ("RISE better vs NEM", "rise_better_vs_nem_pos"),
    ("NEM better vs RISE", "nem_better_vs_rise_pos"),
]

loaded = []
for title, folder_name in cases:
    npz_path = pick_npz_from_folder(folder_name)
    ex = load_npz(npz_path)

    gi = ex["meta"].get("gi", "NA")
    out_png = OUT_DIR / f"{folder_name}_gi{gi}.png"
    render_side_by_side(ex, out_png, case_title=title)
    loaded.append((title, ex))

render_montage(loaded, OUT_DIR / "montage_extremes.png")
print("[done] Patch-level extremes visualization (IG/NEM/RISE) complete.")
