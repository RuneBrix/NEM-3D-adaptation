import os
import sys
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import imageio.v2 as imageio

from exp_config import CHOSEN_DATASETS, CHOSEN_MODELS, TEMPERATURE
from attrs.intgrad3d import itg3d_atr
from attrs.rise3d import rs3d_atr
from attrs.nem_utils.method_nemt3d import NEMT3DMethod

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v in ("", "None", None):
        return default
    return int(v)

def topk_ratio_map(v01: np.ndarray, ratio: float, mode: str = "volume") -> np.ndarray:
    v = np.asarray(v01, dtype=np.float32)
    if ratio <= 0.0:
        return v

    if mode == "volume":
        flat = v.reshape(-1)
        k = max(1, int(ratio * flat.size))
        if k >= flat.size:
            return v
        idx = np.argpartition(flat, -k)[-k:]  # exact k indices
        out = np.zeros_like(flat, dtype=np.float32)
        out[idx] = flat[idx]
        return out.reshape(v.shape)

    if mode == "slice":
        D = v.shape[0]
        out = np.zeros_like(v, dtype=np.float32)
        for d in range(D):
            flat = v[d].reshape(-1)
            k = max(1, int(ratio * flat.size))
            if k >= flat.size:
                out[d] = v[d]
                continue
            idx = np.argpartition(flat, -k)[-k:]
            tmp = np.zeros_like(flat, dtype=np.float32)
            tmp[idx] = flat[idx]
            out[d] = tmp.reshape(v[d].shape)
        return out

    raise ValueError("topk_ratio_map mode must be 'volume' or 'slice'")

def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if v in ("", "None", None):
        return default
    return float(v)

def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if v in ("", "None", None):
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    if v in ("", "None", None):
        return default
    return str(v)

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

def norm01(a: np.ndarray, name="attr"):
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        print(f"[warn] norm01({name}): non-finite values -> nan_to_num")
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(a.min())
    hi = float(a.max())
    if (hi - lo) < 1e-12:
        print(f"[warn] norm01({name}): constant map (min=max={lo:.6f}) -> zeros")
        return np.zeros_like(a, dtype=np.float32)
    return (a - lo) / (hi - lo + 1e-9)

def clip_percentiles(a: np.ndarray, p_lo: float, p_hi: float):
    a = np.asarray(a, dtype=np.float32)
    lo = np.nanpercentile(a, p_lo)
    hi = np.nanpercentile(a, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return a
    return np.clip(a, lo, hi)

def prep_attr(a: np.ndarray, how: str, name="attr", clip_lo=None, clip_hi=None):
    a = np.asarray(a, dtype=np.float32)
    if how == "abs":
        a = np.abs(a)
    elif how == "relu":
        a = np.maximum(a, 0.0)
    elif how == "raw":
        pass
    else:
        raise ValueError(f"Unknown prep mode: {how}")

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_lo is not None and clip_hi is not None:
        a = clip_percentiles(a, clip_lo, clip_hi)

    return norm01(a, name=name)

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return 0.0
    return float((a @ b) / den)

def topk_ratio_map(v01: np.ndarray, ratio: float, mode: str = "volume") -> np.ndarray:
    """
    Keep only top ratio of values, set the rest to 0 (values kept unchanged).
    mode:
      - "volume": threshold from whole (D,H,W)
      - "slice":  threshold per depth slice
    """
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
        D = v.shape[0]
        out = np.zeros_like(v, dtype=np.float32)
        for d in range(D):
            flat = v[d].reshape(-1)
            k = max(1, int(ratio * flat.size))
            if k >= flat.size:
                out[d] = v[d]
                continue
            thr = np.partition(flat, -k)[-k]
            m = v[d] >= thr
            out[d][m] = v[d][m]
        return out

    raise ValueError("topk_ratio_map mode must be 'volume' or 'slice'")

def topk_com(v01_3d: np.ndarray, ratio: float) -> tuple[float, float, float]:
    v = np.asarray(v01_3d, np.float32)
    flat = v.reshape(-1)
    k = max(1, int(ratio * flat.size))
    thr = np.partition(flat, -k)[-k]
    coords = np.argwhere(v >= thr)
    if coords.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    com = coords.mean(axis=0)  # (d,h,w)
    return (float(com[0]), float(com[1]), float(com[2]))

def mip2d(vol3d: np.ndarray, axis_tag: str) -> np.ndarray:
    """
    vol3d: (D,H,W)
    axis_tag: "ax" | "cor" | "sag"
    Returns a 2D image with a consistent orientation for display.
    """
    axis_tag = axis_tag.lower()
    if axis_tag in ("ax", "axial"):
        # project along D -> (H,W)
        img = np.max(vol3d, axis=0)
        return img

    if axis_tag in ("cor", "coronal"):
        # project along H -> (D,W)
        img = np.max(vol3d, axis=1)
        return img

    if axis_tag in ("sag", "sagittal"):
        # project along W -> (D,H)
        img = np.max(vol3d, axis=2)
        img = img.T  # (H,D)
        return img

    raise ValueError(f"Unknown axis_tag={axis_tag}")

def _pct(a, ps=(0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 100)):
    a = np.asarray(a, np.float32).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {p: float("nan") for p in ps}
    vals = np.percentile(a, ps)
    return {float(p): float(v) for p, v in zip(ps, vals)}

def print_stats(name: str, a: np.ndarray, extra=False):
    a = np.asarray(a, np.float32)
    finite = np.isfinite(a)
    frac_finite = float(finite.mean()) if a.size else 0.0
    a2 = a[finite] if a.size else a
    if a2.size == 0:
        print(f"[dbg] {name}: empty / non-finite only")
        return

    mn = float(a2.min()); mx = float(a2.max())
    mean = float(a2.mean()); std = float(a2.std())
    zfrac = float((np.abs(a2) < 1e-8).mean())
    ofrac = float((np.abs(a2 - 1.0) < 1e-3).mean())
    print(f"[dbg] {name}: shape={tuple(a.shape)} finite={frac_finite*100:.2f}% "
          f"min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f} "
          f"~0={zfrac*100:.2f}% ~1={ofrac*100:.2f}%")
    if extra:
        p = _pct(a2)
        ps = ", ".join([f"p{p0:g}={p[p0]:.4f}" for p0 in sorted(p.keys()) if p0 in (0.1, 1, 5, 50, 95, 99, 99.5, 99.9)])
        print(f"[dbg] {name} percentiles: {ps}")

def print_hist(name: str, a: np.ndarray, bins=12, rng=None):
    a = np.asarray(a, np.float32).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"[dbg] {name} hist: empty")
        return
    if rng is None:
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        rng = (lo, hi)
    h, edges = np.histogram(a, bins=bins, range=rng)
    print(f"[dbg] {name} hist range={rng} bins={bins}")
    for i in range(bins):
        print(f"  [{edges[i]:.3f}, {edges[i+1]:.3f}): {int(h[i])}")

def rounded_unique_count(a: np.ndarray, decimals=3, max_show=20):
    a = np.asarray(a, np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0, []
    r = np.round(a, decimals=decimals)
    u = np.unique(r)
    u_sorted = np.sort(u)
    sample = u_sorted[:max_show].tolist()
    return int(u.size), sample

def mask_bbox(mask01: np.ndarray, thr=0.5):
    m = np.asarray(mask01, np.float32)
    coords = np.argwhere(m >= thr)
    if coords.size == 0:
        return None
    d0, h0, w0 = coords.min(axis=0)
    d1, h1, w1 = coords.max(axis=0)
    return (int(d0), int(d1), int(h0), int(h1), int(w0), int(w1))

def border_center_means(m: np.ndarray, border=10):
    """Compare mean mask in border band vs center region (per-slice averaged)."""
    m = np.asarray(m, np.float32)
    D, H, W = m.shape
    b = max(1, min(border, H//2 - 1, W//2 - 1))
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True
    center_mask = ~border_mask
    border_mean = float(m[:, border_mask].mean())
    center_mean = float(m[:, center_mask].mean())
    return border_mean, center_mean, float(border_mean - center_mean)

def dice_iou(a_bin: np.ndarray, b_bin: np.ndarray):
    a = a_bin.astype(bool).reshape(-1)
    b = b_bin.astype(bool).reshape(-1)
    inter = float(np.logical_and(a, b).sum())
    ua = float(a.sum()); ub = float(b.sum())
    union = float(np.logical_or(a, b).sum())
    dice = (2.0 * inter) / (ua + ub + 1e-9)
    iou = inter / (union + 1e-9)
    return dice, iou, ua, ub, inter, union

def tta_logits_3d(model, x):
    logits = []
    logits.append(model(x))
    logits.append(model(torch.flip(x, dims=[2])))
    logits.append(model(torch.flip(x, dims=[3])))
    logits.append(model(torch.flip(x, dims=[4])))
    return torch.stack([z.view(-1) for z in logits], dim=0).mean(dim=0)

@torch.no_grad()
def prob_pos_from_logits(logits_1d: torch.Tensor, use_temp: bool, temperature: float) -> torch.Tensor:
    logits_1d = logits_1d.view(-1)
    if use_temp:
        return torch.sigmoid(logits_1d / float(temperature))
    return torch.sigmoid(logits_1d)

@torch.no_grad()
def model_logit_prob(model, x5d, use_tta: bool, use_temp: bool, temperature: float):
    logits = tta_logits_3d(model, x5d) if use_tta else model(x5d).view(-1)
    prob = float(prob_pos_from_logits(logits, use_temp, temperature)[0].item())
    return float(logits[0].item()), prob

def overlay_rgb(gray2d: np.ndarray,
                heat01_2d: np.ndarray,
                vmin: float,
                vmax: float,
                cmap_name: str = "magma",
                alpha: float = 0.55,
                gamma: float = 0.85,
                heat_eps: float = 0.02) -> np.ndarray:
    g = np.asarray(gray2d, dtype=np.float32)
    h = np.asarray(heat01_2d, dtype=np.float32)

    g01 = (g - vmin) / (vmax - vmin + 1e-9)
    g01 = np.clip(g01, 0.0, 1.0)
    gray_rgb = np.repeat(g01[..., None], 3, axis=-1)

    cmap = cm.get_cmap(cmap_name)
    heat_rgb = cmap(np.clip(h, 0.0, 1.0))[..., :3].astype(np.float32)

    h_eff = np.clip(h, 0.0, 1.0)
    h_eff[h_eff < heat_eps] = 0.0
    a = alpha * (h_eff ** gamma)
    a = a[..., None]

    out = (1.0 - a) * gray_rgb + a * heat_rgb
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

def resolve_val_indices(val_loader):
    val_dataset = val_loader.dataset
    if hasattr(val_dataset, "indices"):
        full_dataset = val_dataset.dataset
        val_indices = list(val_dataset.indices)
        print(f"[viz] val subset size={len(val_indices)}, full dataset size={len(full_dataset)}")
    else:
        full_dataset = val_dataset
        val_indices = list(range(len(full_dataset)))
        print(f"[viz] val dataset size={len(full_dataset)} (no subset wrapper).")
    return full_dataset, val_indices

def get_label(full_dataset, gi, fallback_y=None):
    if hasattr(full_dataset, "rows"):
        row = full_dataset.rows[gi]
        return int(row.get("label", 0))
    if fallback_y is None:
        _, y = full_dataset[gi]
        return int(y.item()) if torch.is_tensor(y) else int(y)
    return int(fallback_y.item()) if torch.is_tensor(fallback_y) else int(fallback_y)

@torch.no_grad()
def select_top_samples(
    full_dataset,
    val_indices,
    model,
    device,
    pos_label: int,
    decision_thr: float,
    use_tta: bool,
    use_temp: bool,
    temperature: float,
    min_prob: float,
    require_correct: bool,
    scan_max_val: int,
    max_samples: int,
):
    pos_candidates = []
    for gi in val_indices:
        if hasattr(full_dataset, "rows"):
            lbl = int(full_dataset.rows[gi].get("label", 0))
        else:
            _, y = full_dataset[gi]
            lbl = int(y.item()) if torch.is_tensor(y) else int(y)
        if lbl == pos_label:
            pos_candidates.append(gi)

    print(f"[viz] Found {len(pos_candidates)} label={pos_label} candidates in validation subset.")
    if len(pos_candidates) == 0:
        return []

    if scan_max_val > 0 and len(pos_candidates) > scan_max_val:
        print(f"[viz] Limiting scan to first {scan_max_val} positives (from {len(pos_candidates)}).")
        pos_candidates = pos_candidates[:scan_max_val]

    kept = []
    skipped_low_prob = 0
    skipped_incorrect = 0

    for gi in pos_candidates:
        X_patch, y = full_dataset[gi]
        label = get_label(full_dataset, gi, y)
        X_sample = X_patch.unsqueeze(0).to(device)

        logits = tta_logits_3d(model, X_sample) if use_tta else model(X_sample).view(-1)
        prob_pos = float(prob_pos_from_logits(logits, use_temp, temperature)[0].item())
        pred_label = int(prob_pos >= decision_thr)

        if prob_pos < min_prob:
            skipped_low_prob += 1
            continue
        if require_correct and (pred_label != label):
            skipped_incorrect += 1
            continue

        kept.append(dict(gi=gi, label=label, pred_label=pred_label, prob_pos=prob_pos, logit=float(logits[0].item())))

    print(f"[viz] After filters: kept={len(kept)} (skipped_low_prob={skipped_low_prob}, skipped_incorrect={skipped_incorrect})")
    if not kept:
        return []

    kept = sorted(kept, key=lambda d: d["prob_pos"], reverse=True)[:max_samples]
    print("[viz] Selected top samples:")
    for k in kept:
        print(f"  gi={k['gi']} | label={k['label']} | pred={k['pred_label']} | prob_pos={k['prob_pos']:.3f} | logit={k['logit']:.3f}")
    return kept

def compute_ig_safe(IG_explainer: itg3d_atr,
                    X_sample_5d: torch.Tensor,
                    internal_batch: int,
                    max_retries: int = 5):
    """
    Try to compute IG using Captum internal batching.
    Falls back by reducing internal_batch and/or n_steps if CUDA OOM happens.
    """
    import inspect

    n_steps0 = int(getattr(IG_explainer, "n_steps", 32))
    tries = []

    # Try internal batch sizes down to 1
    ibs = []
    if internal_batch >= 1:
        ibs.append(int(internal_batch))
    if 1 not in ibs:
        ibs.append(1)

    # Try reducing steps as well if needed
    step_candidates = [n_steps0]
    for f in [2, 4, 8]:
        s = max(8, n_steps0 // f)
        if s not in step_candidates:
            step_candidates.append(s)

    for s in step_candidates:
        for ib in ibs:
            tries.append((s, ib))

    last_err = None

    for attempt_idx, (n_steps, ib) in enumerate(tries[:max_retries], start=1):
        try:
            # update explainer steps if possible
            if hasattr(IG_explainer, "n_steps"):
                IG_explainer.n_steps = int(n_steps)

            sig = None
            try:
                sig = inspect.signature(IG_explainer.gen_attr)
            except Exception:
                sig = None

            torch.cuda.empty_cache()

            t0 = time.perf_counter()

            if sig is not None and ("internal_batch_size" in sig.parameters):
                ig = IG_explainer.gen_attr(X_sample_5d, internal_batch_size=int(ib))
            elif hasattr(IG_explainer, "_ig") and hasattr(IG_explainer._ig, "attribute"):
                baseline = None
                # common attribute names to probe
                for k in ["baseline", "baselines", "_baseline", "_baselines"]:
                    if hasattr(IG_explainer, k):
                        baseline = getattr(IG_explainer, k)
                        break
                if baseline is None:
                    baseline = torch.zeros_like(X_sample_5d)

                target = 0
                ig_t = IG_explainer._ig.attribute(
                    inputs=X_sample_5d,
                    baselines=baseline,
                    target=target,
                    n_steps=int(n_steps),
                    internal_batch_size=int(ib),
                )
                ig = ig_t.detach().cpu().numpy()
            else:
                ig = IG_explainer.gen_attr(X_sample_5d)

            dt = time.perf_counter() - t0
            print(f"[viz][IG] success attempt={attempt_idx} n_steps={n_steps} internal_batch={ib} time={dt:.2f}s")
            return ig, dict(ig_n_steps=int(n_steps), ig_internal_batch=int(ib), ig_time=float(dt))

        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            print(f"[viz][IG][OOM] attempt={attempt_idx} n_steps={n_steps} internal_batch={ib} -> {repr(e)}")
            torch.cuda.empty_cache()
        except Exception as e:
            last_err = e
            print(f"[viz][IG][ERR] attempt={attempt_idx} n_steps={n_steps} internal_batch={ib} -> {repr(e)}")
            traceback.print_exc()
            break

    raise RuntimeError(f"IG failed after retries. Last error: {repr(last_err)}")

def compute_attributions(
    X_sample_5d: torch.Tensor,
    IG_explainer: itg3d_atr,
    RISE_explainer: rs3d_atr,
    nem_explainer: NEMT3DMethod,
    ig_internal_batch: int,
):
    # IG (OOM-safe)
    t0 = time.perf_counter()
    ig_np, ig_meta = compute_ig_safe(IG_explainer, X_sample_5d, internal_batch=ig_internal_batch)
    t_ig = time.perf_counter() - t0

    # RISE
    t0 = time.perf_counter()
    rise = RISE_explainer.gen_attr(X_sample_5d, None)
    t_rise = time.perf_counter() - t0

    # NEM keep-mask + masked volume (if returned)
    t0 = time.perf_counter()
    out = nem_explainer.gen_mask(X_sample_5d)
    t_nem = time.perf_counter() - t0

    # Support both return signatures:
    #   (x_masked, scores, mask) or (scores, x_masked, mask) etc
    x_masked = None
    mask_tensor = None
    if isinstance(out, (tuple, list)) and len(out) >= 3:
        candidates = list(out)
        tensor_cands = [c for c in candidates if torch.is_tensor(c)]
        if len(tensor_cands) >= 1:
            mask_tensor = tensor_cands[-1]
            for c in tensor_cands:
                if c is mask_tensor:
                    continue
                if c.shape == X_sample_5d.shape:
                    x_masked = c
                    break
    if mask_tensor is None:
        raise RuntimeError("Could not parse NEM output: mask_tensor is None")

    ig3d = to_3d_volume(ig_np)
    rise3d = to_3d_volume(rise)
    nem_keep3d = to_3d_volume(mask_tensor.detach().cpu().numpy())  # keep in [0,1]
    x_masked3d = to_3d_volume(x_masked.detach().cpu().numpy()) if x_masked is not None else None

    debug = {
        "t_ig": float(t_ig),
        "t_rise": float(t_rise),
        "t_nem": float(t_nem),
        "ig_meta": ig_meta,
        "ig_stats": (float(np.min(ig3d)), float(np.max(ig3d)), float(np.mean(ig3d))),
        "rise_stats": (float(np.min(rise3d)), float(np.max(rise3d)), float(np.mean(rise3d))),
        "nem_keep_stats": (float(np.min(nem_keep3d)), float(np.max(nem_keep3d)), float(np.mean(nem_keep3d))),
        "x_masked_present": x_masked3d is not None,
    }
    return ig3d, rise3d, nem_keep3d, x_masked3d, debug

def slice_score_topmean(a01_3d: np.ndarray, top_frac: float = 0.02) -> np.ndarray:
    D = a01_3d.shape[0]
    scores = np.zeros(D, dtype=np.float32)
    for d in range(D):
        flat = a01_3d[d].reshape(-1)
        k = max(1, int(top_frac * flat.size))
        topk = np.partition(flat, -k)[-k:]
        scores[d] = float(np.mean(topk))
    return scores

def choose_slices(
    ig01: np.ndarray,
    rise01: np.ndarray,
    nem01: np.ndarray,
    num_slices: int,
    slice_by: str = "rise",
    top_frac: float = 0.02,
):
    s_ig = slice_score_topmean(ig01, top_frac=top_frac)
    s_r = slice_score_topmean(rise01, top_frac=top_frac)
    s_n = slice_score_topmean(nem01, top_frac=top_frac)

    if slice_by == "ig":
        s = s_ig
    elif slice_by == "rise":
        s = s_r
    elif slice_by == "nem":
        s = s_n
    elif slice_by == "max":
        s = np.maximum(np.maximum(s_ig, s_r), s_n)
    elif slice_by == "combined":
        s = (s_ig + s_r + s_n) / 3.0
    else:
        raise ValueError("VIZ_SLICE_BY must be one of: ig|rise|nem|max|combined")

    order = np.argsort(s)[::-1]
    picks = sorted([int(x) for x in order[:num_slices]])
    return picks, dict(ig=s_ig, rise=s_r, nem=s_n, used=s)

def save_slice_comparison_figure(
    out_path_png: Path,
    out_path_pdf: Path,
    x3d: np.ndarray,
    ig01: np.ndarray,
    rise01: np.ndarray,
    nem01: np.ndarray,
    slice_ids,
    cmap_name: str,
    alpha: float,
    gamma: float,
    heat_eps: float,
    title_prefix: str = "",
):
    vmin, vmax = robust_minmax(x3d, 1.0, 99.0)
    cols = ["Input", "IG", "RISE", "NEM"]
    rows = len(slice_ids)

    fig, axes = plt.subplots(rows, len(cols), figsize=(3.2 * len(cols), 3.0 * rows), dpi=250)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, d in enumerate(slice_ids):
        sl = x3d[d]
        axes[r, 0].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 0].set_title(f"{cols[0]} (d={d})")

        for c, (name, a01) in enumerate([("IG", ig01), ("RISE", rise01), ("NEM", nem01)], start=1):
            axes[r, c].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)

            a = np.clip(a01[d], 0.0, 1.0)
            a[a < heat_eps] = 0.0
            alpha_map = alpha * (a ** gamma)

            axes[r, c].imshow(a01[d], cmap=cmap_name, vmin=0.0, vmax=1.0, alpha=alpha_map)
            axes[r, c].set_title(f"{name} (d={d})")
            axes[r, c].axis("off")

        axes[r, 0].axis("off")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path_png, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)

def save_gifs(
    out_dir: Path,
    x3d: np.ndarray,
    ig01: np.ndarray,
    rise01: np.ndarray,
    nem01: np.ndarray,
    cmap_name: str,
    alpha: float,
    gamma: float,
    heat_eps: float,
    gif_max_frames: int,
    gif_fps: int,
    make_panel_gif: bool,
):
    D = x3d.shape[0]
    vmin, vmax = robust_minmax(x3d, 1.0, 99.0)

    step = max(1, int(math.ceil(D / max(1, gif_max_frames))))
    depth_ids = list(range(0, D, step))
    if depth_ids[-1] != (D - 1):
        depth_ids.append(D - 1)

    print(f"[viz] GIF depth frames: D={D}, step={step}, frames={len(depth_ids)}")

    def frames_for(attr01: np.ndarray, label: str):
        frames = []
        for d in depth_ids:
            rgb = overlay_rgb(
                x3d[d], attr01[d],
                vmin=vmin, vmax=vmax,
                cmap_name=cmap_name,
                alpha=alpha, gamma=gamma, heat_eps=heat_eps
            )
            frames.append(rgb)
        out_path = out_dir / f"{label}.gif"
        imageio.mimsave(out_path, frames, fps=gif_fps)
        return out_path

    ig_gif = frames_for(ig01, "ig")
    rise_gif = frames_for(rise01, "rise")
    nem_gif = frames_for(nem01, "nem")

    panel_gif = None
    if make_panel_gif:
        frames = []
        zero = np.zeros_like(ig01[0], dtype=np.float32)
        for d in depth_ids:
            inp = overlay_rgb(x3d[d], zero, vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=0.0, gamma=gamma, heat_eps=heat_eps)
            igp = overlay_rgb(x3d[d], ig01[d], vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, gamma=gamma, heat_eps=heat_eps)
            rip = overlay_rgb(x3d[d], rise01[d], vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, gamma=gamma, heat_eps=heat_eps)
            nep = overlay_rgb(x3d[d], nem01[d], vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, gamma=gamma, heat_eps=heat_eps)
            panel = np.concatenate([inp, igp, rip, nep], axis=1)
            frames.append(panel)
        panel_gif = out_dir / "panel_input_ig_rise_nem.gif"
        imageio.mimsave(panel_gif, frames, fps=gif_fps)

    return ig_gif, rise_gif, nem_gif, panel_gif

def save_debug_nem_slices(out_path: Path,
                         x3d: np.ndarray,
                         mask_keep01: np.ndarray,
                         nem_kind01: np.ndarray,
                         rise01: np.ndarray,
                         slice_ids,
                         title: str):
    vmin, vmax = robust_minmax(x3d, 1.0, 99.0)
    cols = ["Input", "NEM keep mask", "NEM kind (viz)", "RISE"]
    rows = len(slice_ids)

    fig, axes = plt.subplots(rows, len(cols), figsize=(3.2 * len(cols), 3.0 * rows), dpi=200)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, d in enumerate(slice_ids):
        sl = x3d[d]

        axes[r, 0].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 0].set_title(f"Input d={d}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 1].imshow(mask_keep01[d], cmap="viridis", vmin=0, vmax=1, alpha=0.75)
        axes[r, 1].set_title("Mask keep (overlay)")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 2].imshow(nem_kind01[d], cmap="magma", vmin=0, vmax=1, alpha=0.75)
        axes[r, 2].set_title("NEM viz kind (overlay)")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 3].imshow(rise01[d], cmap="magma", vmin=0, vmax=1, alpha=0.75)
        axes[r, 3].set_title("RISE (overlay)")
        axes[r, 3].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_debug_profiles(out_path: Path,
                        mask_keep01: np.ndarray,
                        nem_kind01: np.ndarray,
                        rise01: np.ndarray,
                        ig01: np.ndarray,
                        thr: float = 0.5):
    D = mask_keep01.shape[0]
    xs = np.arange(D)

    keep_mean = mask_keep01.reshape(D, -1).mean(axis=1)
    keep_cov = (mask_keep01.reshape(D, -1) >= thr).mean(axis=1)

    nem_mean = nem_kind01.reshape(D, -1).mean(axis=1)
    nem_top = slice_score_topmean(nem_kind01, top_frac=0.02)

    rise_mean = rise01.reshape(D, -1).mean(axis=1)
    rise_top = slice_score_topmean(rise01, top_frac=0.02)

    ig_mean = ig01.reshape(D, -1).mean(axis=1)
    ig_top = slice_score_topmean(ig01, top_frac=0.02)

    fig = plt.figure(figsize=(12, 7), dpi=200)
    ax = plt.gca()

    ax.plot(xs, keep_mean, label="mask_keep mean")
    ax.plot(xs, keep_cov, label=f"mask_keep cov>= {thr}")
    ax.plot(xs, nem_mean, label="nem_kind mean")
    ax.plot(xs, nem_top, label="nem_kind top-2% mean")
    ax.plot(xs, rise_top, label="rise top-2% mean")
    ax.plot(xs, ig_top, label="ig top-2% mean")

    ax.set_xlabel("depth slice (d)")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Per-slice profiles (mask / NEM kind / IG / RISE)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_debug_overlap(out_path: Path,
                       nem01: np.ndarray,
                       rise01: np.ndarray,
                       ratio: float,
                       mode: str = "slice"):
    nem_top = topk_ratio_map(nem01, ratio, mode=mode) > 0
    rise_top = topk_ratio_map(rise01, ratio, mode=mode) > 0

    # per-slice dice/iou
    D = nem01.shape[0]
    dice_s = np.zeros(D, np.float32)
    iou_s = np.zeros(D, np.float32)
    a_s = np.zeros(D, np.float32)
    b_s = np.zeros(D, np.float32)

    for d in range(D):
        dice, iou, ua, ub, inter, union = dice_iou(nem_top[d], rise_top[d])
        dice_s[d] = dice
        iou_s[d] = iou
        a_s[d] = ua
        b_s[d] = ub

    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = plt.gca()
    ax.plot(dice_s, label="Dice (slice)")
    ax.plot(iou_s, label="IoU (slice)")
    ax.set_xlabel("depth slice (d)")
    ax.set_ylabel("overlap")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"NEM vs RISE overlap per slice (topk ratio={ratio}, mode={mode})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    dataset_name = os.environ.get("NEM_DATASETS", "luna3d").split(",")[0]
    model_name = os.environ.get("NEM_MODELS", "phase3_3d").split(",")[0]

    DECISION_THR = env_float("NEM_DECISION_THR", 0.20)
    USE_TTA = env_bool("NEM_USE_TTA", True)
    USE_TEMP = env_bool("NEM_USE_TEMP", True)

    POS_LABEL = env_int("VIZ_POS_LABEL", 1)
    MAX_SAMPLES = env_int("VIZ_MAX_SAMPLES", 5)
    MIN_PROB = env_float("VIZ_MIN_PROB", 0.5)
    REQUIRE_CORRECT = env_bool("VIZ_REQUIRE_CORRECT", False)
    SCAN_MAX_VAL = env_int("VIZ_SCAN_MAX_VAL", 2000)

    NUM_SLICES = env_int("VIZ_NUM_SLICES", 3)
    SLICE_BY = env_str("VIZ_SLICE_BY", "rise")
    SLICE_TOP_FRAC = env_float("VIZ_SLICE_TOP_FRAC", 0.02)

    CMAP = env_str("VIZ_CMAP", "magma")
    OVERLAY_ALPHA = env_float("VIZ_OVERLAY_ALPHA", 0.55)
    OVERLAY_GAMMA = env_float("VIZ_OVERLAY_GAMMA", 0.85)
    HEAT_EPS = env_float("VIZ_HEAT_EPS", 0.02)

    GIF_MAX_FRAMES = env_int("VIZ_GIF_MAX_FRAMES", 64)
    GIF_FPS = env_int("VIZ_GIF_FPS", 8)
    MAKE_PANEL_GIF = env_bool("VIZ_MAKE_PANEL_GIF", True)

    IG_PREP = env_str("VIZ_IG_PREP", "abs")
    RISE_PREP = env_str("VIZ_RISE_PREP", "raw")
    NEM_PREP = env_str("VIZ_NEM_PREP", "raw")

    CLIP_LO = env_float("VIZ_ATTR_CLIP_LO", 0.5)
    CLIP_HI = env_float("VIZ_ATTR_CLIP_HI", 99.5)

    TOPK_IG = env_float("VIZ_TOPK_IG", 0.0)
    TOPK_RISE = env_float("VIZ_TOPK_RISE", 0.0)
    TOPK_NEM = env_float("VIZ_TOPK_NEM", 0.10)
    TOPK_MODE_IG = env_str("VIZ_TOPK_MODE_IG", "volume")
    TOPK_MODE_RISE = env_str("VIZ_TOPK_MODE_RISE", "volume")
    TOPK_MODE_NEM = env_str("VIZ_TOPK_MODE_NEM", "slice")

    NEM_ATTR_KIND = env_str("VIZ_NEM_ATTR_KIND", "delta").strip().lower()  # keep|remove|delta|keep_delta
    NEM_CONTRAST_WEIGHT = env_bool("VIZ_NEM_CONTRAST_WEIGHT", False)

    IG_INTERNAL_BATCH = env_int("VIZ_IG_INTERNAL_BATCH", 4)
    SAVE_NPZ = env_bool("VIZ_SAVE_NPZ", False)
    DEBUG_OVERLAP_RATIO = env_float("VIZ_DEBUG_OVERLAP_RATIO", 0.05)

    IG_N_STEPS = os.environ.get("IG_N_STEPS", "")
    RISE_N_MASKS = os.environ.get("RISE_N_MASKS", "")

    print(f"[info] dataset='{dataset_name}', model='{model_name}'")
    print(f"[info] USE_TTA={USE_TTA} | USE_TEMP={USE_TEMP} | TEMPERATURE={float(TEMPERATURE)} | DECISION_THR={DECISION_THR}")
    print(f"[info] Selection: POS_LABEL={POS_LABEL} | MAX_SAMPLES={MAX_SAMPLES} | MIN_PROB={MIN_PROB} | REQUIRE_CORRECT={REQUIRE_CORRECT} | SCAN_MAX_VAL={SCAN_MAX_VAL}")
    print(f"[info] Slices: NUM_SLICES={NUM_SLICES} | SLICE_BY={SLICE_BY} | TOP_FRAC={SLICE_TOP_FRAC}")
    print(f"[info] Overlay: CMAP={CMAP} | ALPHA={OVERLAY_ALPHA} | GAMMA={OVERLAY_GAMMA} | HEAT_EPS={HEAT_EPS}")
    print(f"[info] Prep: IG={IG_PREP} | RISE={RISE_PREP} | NEM={NEM_PREP}")
    print(f"[info] Attr clip percentiles: lo={CLIP_LO} hi={CLIP_HI}")
    print(f"[info] TopK: IG={TOPK_IG}({TOPK_MODE_IG}) RISE={TOPK_RISE}({TOPK_MODE_RISE}) NEM={TOPK_NEM}({TOPK_MODE_NEM})")
    print(f"[info] NEM attr kind: {NEM_ATTR_KIND} | contrast_weight={NEM_CONTRAST_WEIGHT}")
    print(f"[info] IG internal_batch={IG_INTERNAL_BATCH} | SAVE_NPZ={SAVE_NPZ} | DEBUG_OVERLAP_RATIO={DEBUG_OVERLAP_RATIO}")

    data_obj = CHOSEN_DATASETS[dataset_name]()
    _, val_loader = data_obj.get_data()
    if val_loader is None:
        raise RuntimeError("Validation loader not found (val_loader is None).")

    model = CHOSEN_MODELS[model_name]().eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[info] Device: {device} | CUDA available? {torch.cuda.is_available()}")

    use_predicted = True
    IG_explainer = itg3d_atr(model, train_data=None, use_predicted_labels=use_predicted)
    RISE_explainer = rs3d_atr(model, train_data=None, use_predicted_labels=use_predicted)

    if IG_N_STEPS not in ("", "None"):
        try:
            IG_explainer.n_steps = int(IG_N_STEPS)
            print(f"[info] IG n_steps overridden to {IG_explainer.n_steps}")
        except ValueError:
            print(f"[warn] IG_N_STEPS='{IG_N_STEPS}' invalid; keeping default.")

    if RISE_N_MASKS not in ("", "None"):
        try:
            RISE_explainer.n_masks = int(RISE_N_MASKS)
            print(f"[info] RISE n_masks overridden to {RISE_explainer.n_masks}")
        except ValueError:
            print(f"[warn] RISE_N_MASKS='{RISE_N_MASKS}' invalid; keeping default.")

    nem_explainer = NEMT3DMethod(model, train_loader=None, train_or_load=False, device=str(device))

    full_dataset, val_indices = resolve_val_indices(val_loader)

    selected = select_top_samples(
        full_dataset=full_dataset,
        val_indices=val_indices,
        model=model,
        device=device,
        pos_label=POS_LABEL,
        decision_thr=DECISION_THR,
        use_tta=USE_TTA,
        use_temp=USE_TEMP,
        temperature=float(TEMPERATURE),
        min_prob=MIN_PROB,
        require_correct=REQUIRE_CORRECT,
        scan_max_val=SCAN_MAX_VAL,
        max_samples=MAX_SAMPLES,
    )
    if not selected:
        print("[error] No samples selected.")
        sys.exit(0)

    OUT_DIR = Path("experiments") / dataset_name / model_name / "viz_attributions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[output] OUT_DIR: {OUT_DIR.resolve()}")

    for i, info in enumerate(selected):
        gi = info["gi"]
        X_patch, y = full_dataset[gi]
        label = get_label(full_dataset, gi, y)

        X_sample = X_patch.unsqueeze(0).to(device)
        x3d = to_3d_volume(X_sample.detach().cpu().numpy().astype(np.float32))

        sample_dir = OUT_DIR / f"rank{i:02d}_gi{gi}_prob{info['prob_pos']:.3f}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 80)
        print(f"[viz] Sample {i}: gi={gi} | label={label} | pred={info['pred_label']} | prob={info['prob_pos']:.3f} | logit={info['logit']:.3f}")
        print(f"[viz] Input shape: {x3d.shape} | min={x3d.min():.3f} max={x3d.max():.3f} mean={x3d.mean():.3f}")

        # model score sanity
        base_logit, base_prob = model_logit_prob(model, X_sample, USE_TTA, USE_TEMP, float(TEMPERATURE))
        print(f"[dbg] model(base): logit={base_logit:.4f} prob={base_prob:.4f} (selection said prob={info['prob_pos']:.4f})")

        ig3d, rise3d, nem_keep3d, x_masked3d, dbg = compute_attributions(
            X_sample_5d=X_sample,
            IG_explainer=IG_explainer,
            RISE_explainer=RISE_explainer,
            nem_explainer=nem_explainer,
            ig_internal_batch=IG_INTERNAL_BATCH,
        )
        print(f"[viz] Times: IG={dbg['t_ig']:.2f}s | RISE={dbg['t_rise']:.2f}s | NEM={dbg['t_nem']:.2f}s")
        print(f"[viz] IG raw stats    : {dbg['ig_stats']} | meta={dbg['ig_meta']}")
        print(f"[viz] RISE raw stats  : {dbg['rise_stats']}")
        print(f"[viz] NEM keep stats  : {dbg['nem_keep_stats']} | x_masked_present={dbg['x_masked_present']}")

        # Normalize maps for display base
        ig01 = prep_attr(ig3d, IG_PREP, name="IG", clip_lo=CLIP_LO, clip_hi=CLIP_HI)
        rise01 = prep_attr(rise3d, RISE_PREP, name="RISE", clip_lo=CLIP_LO, clip_hi=CLIP_HI)

        # NEM raw constructions
        baseline_scalar = float(x3d.mean())
        x_contrast = np.abs(x3d - baseline_scalar).astype(np.float32)

        nem_keep = np.clip(nem_keep3d.astype(np.float32), 0.0, 1.0)
        nem_remove = (1.0 - nem_keep).astype(np.float32)
        nem_delta = (nem_remove * x_contrast).astype(np.float32)
        nem_keep_delta = (nem_keep * x_contrast).astype(np.float32)

        if NEM_ATTR_KIND == "keep":
            nem_raw_for_viz = nem_keep
        elif NEM_ATTR_KIND == "remove":
            nem_raw_for_viz = nem_remove
        elif NEM_ATTR_KIND == "delta":
            nem_raw_for_viz = nem_delta
        elif NEM_ATTR_KIND == "keep_delta":
            nem_raw_for_viz = nem_keep_delta
        else:
            raise ValueError("VIZ_NEM_ATTR_KIND must be keep|remove|delta|keep_delta")

        if NEM_ATTR_KIND in ("keep", "remove"):
            nem01 = np.clip(nem_raw_for_viz.astype(np.float32), 0.0, 1.0)
        else:
            nem01 = prep_attr(nem_raw_for_viz, NEM_PREP, name=f"NEM_{NEM_ATTR_KIND}",
                            clip_lo=CLIP_LO, clip_hi=CLIP_HI)


        # Optional: contrast weight
        if NEM_CONTRAST_WEIGHT:
            contrast01 = norm01(x_contrast, name="contrast")
            nem01 = norm01(nem01 * contrast01, name="nem_contrast_weighted")
            print("[viz] Applied NEM contrast weighting (nem *= |x-baseline|).")

        print_stats("x3d", x3d, extra=True)
        print_stats("nem_keep(raw)", nem_keep, extra=True)
        print_hist("nem_keep(raw)", nem_keep, bins=12, rng=(0.0, 1.0))
        ucnt, usample = rounded_unique_count(nem_keep, decimals=3, max_show=25)
        print(f"[dbg] nem_keep(raw) unique@3dp: count={ucnt} sample={usample}")

        bb_05 = mask_bbox(nem_keep, thr=0.5)
        bb_08 = mask_bbox(nem_keep, thr=0.8)
        print(f"[dbg] nem_keep bbox thr=0.5: {bb_05}")
        print(f"[dbg] nem_keep bbox thr=0.8: {bb_08}")

        bmean, cmean, diff = border_center_means(nem_keep, border=10)
        print(f"[dbg] nem_keep border-vs-center (border=10): border_mean={bmean:.4f} center_mean={cmean:.4f} diff={diff:.4f}")

        x_keep_masked = (x3d * nem_keep + baseline_scalar * (1.0 - nem_keep)).astype(np.float32)
        if x_masked3d is not None:
            md = np.abs(x_keep_masked - x_masked3d)
            print(f"[dbg] x_masked diff vs reconstructed keep-masked: max={float(md.max()):.6f} mean={float(md.mean()):.6f}")

        # Model deltas for masking
        x_keep5d = torch.from_numpy(x_keep_masked[None, None]).to(device)
        x_remove_masked = (x3d * (1.0 - nem_keep) + baseline_scalar * nem_keep).astype(np.float32)
        x_remove5d = torch.from_numpy(x_remove_masked[None, None]).to(device)
        log_keep, prob_keep = model_logit_prob(model, x_keep5d, USE_TTA, USE_TEMP, float(TEMPERATURE))
        log_remove, prob_remove = model_logit_prob(model, x_remove5d, USE_TTA, USE_TEMP, float(TEMPERATURE))
        print(f"[dbg] model(keep-masked):   logit={log_keep:.4f} prob={prob_keep:.4f} | delta_prob={prob_keep-base_prob:+.4f}")
        print(f"[dbg] model(remove-masked): logit={log_remove:.4f} prob={prob_remove:.4f} | delta_prob={prob_remove-base_prob:+.4f}")

        # COMs
        com_ratio = max(0.01, TOPK_NEM if TOPK_NEM > 0 else 0.05)
        com_ig = topk_com(ig01, com_ratio)
        com_rise = topk_com(rise01, com_ratio)
        com_nem = topk_com(nem01, com_ratio)
        print(f"[dbg] topk COM ratio={com_ratio:.3f} | IG={com_ig} | RISE={com_rise} | NEM(kind)={com_nem}")

        # Overlap NEM vs RISE (topk binarized)
        ov_ratio = float(DEBUG_OVERLAP_RATIO)
        if ov_ratio > 0:
            nem_bin = topk_ratio_map(nem01, ov_ratio, mode=TOPK_MODE_NEM) > 0
            rise_bin = topk_ratio_map(rise01, ov_ratio, mode=TOPK_MODE_NEM) > 0
            dice, iou, ua, ub, inter, union = dice_iou(nem_bin, rise_bin)
            print(f"[dbg] overlap NEM(kind) vs RISE topk={ov_ratio} mode={TOPK_MODE_NEM}: "
                  f"Dice={dice:.4f} IoU={iou:.4f} | "
                  f"|NEM|={ua:.0f} |RISE|={ub:.0f} inter={inter:.0f} union={union:.0f}")
        
        ig01_full   = ig01.copy()
        rise01_full = rise01.copy()
        nem01_full  = nem01.copy()

        ig01 = topk_ratio_map(ig01, TOPK_IG, mode=TOPK_MODE_IG)
        rise01 = topk_ratio_map(rise01, TOPK_RISE, mode=TOPK_MODE_RISE)
        nem01_disp = topk_ratio_map(nem01, TOPK_NEM, mode=TOPK_MODE_NEM)
        
        # for display
        ig01_disp   = topk_ratio_map(ig01_full, TOPK_IG, mode=TOPK_MODE_IG)
        rise01_disp = topk_ratio_map(rise01_full, TOPK_RISE, mode=TOPK_MODE_RISE)
        nem01_disp  = topk_ratio_map(nem01_full, TOPK_NEM, mode=TOPK_MODE_NEM)

        # choose slices using full maps
        slice_ids, scores = choose_slices(
            ig01=ig01_full,
            rise01=rise01_full,
            nem01=nem01_full,
            num_slices=NUM_SLICES,
            slice_by=SLICE_BY,
            top_frac=SLICE_TOP_FRAC,
        )
        
        print(f"[viz] Chosen slices: {slice_ids} (slice_by={SLICE_BY})")
        for d in slice_ids:
            print(f"  d={d} | used={scores['used'][d]:.4f} | ig={scores['ig'][d]:.4f} | rise={scores['rise'][d]:.4f} | nem={scores['nem'][d]:.4f}")

        # Save main paper figs
        title_prefix = f"gi={gi} | label={label} | prob_pos={info['prob_pos']:.3f} | NEM={NEM_ATTR_KIND}"
        out_png = sample_dir / "slices_input_ig_rise_nem.png"
        out_pdf = sample_dir / "slices_input_ig_rise_nem.pdf"
        save_slice_comparison_figure(
            out_path_png=out_png,
            out_path_pdf=out_pdf,
            x3d=x3d,
            ig01=ig01_disp,
            rise01=rise01_disp,
            nem01=nem01_disp,
            slice_ids=slice_ids,
            cmap_name=CMAP,
            alpha=OVERLAY_ALPHA,
            gamma=OVERLAY_GAMMA,
            heat_eps=HEAT_EPS,
            title_prefix=title_prefix,
        )
        print(f"[output] Saved: {out_png.name} and {out_pdf.name}")

        # Save GIFs
        ig_gif, rise_gif, nem_gif, panel_gif = save_gifs(
            out_dir=sample_dir,
            x3d=x3d,
            ig01=ig01_disp,
            rise01=rise01_disp,
            nem01=nem01_disp,
            cmap_name=CMAP,
            alpha=OVERLAY_ALPHA,
            gamma=OVERLAY_GAMMA,
            heat_eps=HEAT_EPS,
            gif_max_frames=GIF_MAX_FRAMES,
            gif_fps=GIF_FPS,
            make_panel_gif=MAKE_PANEL_GIF,
        )
        print(f"[output] Saved GIFs: {ig_gif.name}, {rise_gif.name}, {nem_gif.name}")
        if panel_gif is not None:
            print(f"[output] Saved panel GIF: {panel_gif.name}")

        # Save debug figures
        dbg_slices = slice_ids
        dbg_path1 = sample_dir / "debug_nem_mask_slices.png"
        save_debug_nem_slices(
            out_path=dbg_path1,
            x3d=x3d,
            mask_keep01=nem_keep,       # raw keep mask in [0,1]
            nem_kind01=nem01,           # pre-topk normalized viz-kind
            rise01=rise01,
            slice_ids=dbg_slices,
            title=f"DEBUG NEM: keep mask / kind={NEM_ATTR_KIND} / RISE (gi={gi})"
        )
        print(f"[output] Saved debug: {dbg_path1.name}")

        dbg_path2 = sample_dir / "debug_nem_profiles.png"
        save_debug_profiles(
            out_path=dbg_path2,
            mask_keep01=nem_keep,
            nem_kind01=nem01_full,
            rise01=rise01_full,
            ig01=ig01_full,
            thr=0.5
        )
        print(f"[output] Saved debug: {dbg_path2.name}")

        if DEBUG_OVERLAP_RATIO > 0:
            dbg_path3 = sample_dir / "debug_overlap_profiles.png"
            save_debug_overlap(
                out_path=dbg_path3,
                nem01=nem01,
                rise01=rise01,
                ratio=float(DEBUG_OVERLAP_RATIO),
                mode=TOPK_MODE_NEM
            )
            print(f"[output] Saved debug: {dbg_path3.name}")

        if SAVE_NPZ:
            npz_path = sample_dir / "debug_arrays.npz"
            np.savez_compressed(
                npz_path,
                x3d=x3d,
                ig3d=ig3d,
                rise3d=rise3d,
                nem_keep=nem_keep,
                nem_raw_for_viz=nem_raw_for_viz,
                ig01=ig01,
                rise01=rise01,
                nem01=nem01,
                nem01_disp=nem01_disp,
                slice_ids=np.array(slice_ids, dtype=np.int32),
                meta=np.array([str(dict(gi=gi, label=label, prob=info["prob_pos"], nem_kind=NEM_ATTR_KIND))], dtype=object),
            )
            print(f"[output] Saved debug arrays: {npz_path.name}")

        # free per-sample tensors
        del x_keep5d, x_remove5d
        torch.cuda.empty_cache()

    print("\n[done] Visualization complete.")


if __name__ == "__main__":
    main()
