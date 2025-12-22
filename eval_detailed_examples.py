import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import quantus

from exp_config import CHOSEN_DATASETS, CHOSEN_MODELS, TEMPERATURE
from attrs.intgrad3d import itg3d_atr
from attrs.rise3d import rs3d_atr
from attrs.nem_utils.method_nemt3d import NEMT3DMethod

PROG_EVERY = int(os.environ.get("DETAIL_PROGRESS_EVERY", "10"))
MON_PROG_EVERY = int(os.environ.get("MON_PROGRESS_EVERY", "0"))

def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"

def progress_line(prefix: str, i: int, n: int, t0: float) -> None:
    if n <= 0:
        return
    i = max(0, min(int(i), int(n)))
    dt = max(1e-9, time.perf_counter() - t0)
    rate = i / dt
    eta = (n - i) / max(rate, 1e-9)
    pct = 100.0 * i / n
    print(
        f"[progress] {prefix}: {i}/{n} ({pct:5.1f}%) | "
        f"{rate:6.2f} it/s | elapsed={_fmt_time(dt)} | ETA={_fmt_time(eta)}",
        flush=True,
    )

def stage(msg: str) -> None:
    print(f"\n{'='*88}\n[stage] {msg}\n{'='*88}\n", flush=True)

dataset_name = os.environ.get("NEM_DATASETS", "luna3d").split(",")[0]
model_name   = os.environ.get("NEM_MODELS", "phase3_3d").split(",")[0]

DETAIL_STYLE = os.environ.get("DETAIL_STYLE", "eval_pos_only").lower()
if DETAIL_STYLE != "eval_pos_only":
    print(f"[warn] DETAIL_STYLE='{DETAIL_STYLE}' but this script is positives-only; forcing eval_pos_only.")
DETAIL_STYLE = "eval_pos_only"

N_POS        = int(os.environ.get("DETAIL_N_POS", "64"))
POS_POOL_MAX = int(os.environ.get("DETAIL_POS_POOL_MAX", "2000"))
SCAN_MAX_VAL = int(os.environ.get("DETAIL_SCAN_MAX_VAL", "5000"))
SEED         = int(os.environ.get("DETAIL_SEED", "0"))

POS_MIN_PROB        = float(os.environ.get("DETAIL_POS_MIN_PROB", "0.5"))
POS_REQUIRE_CORRECT = int(os.environ.get("DETAIL_POS_REQUIRE_CORRECT", "1"))

DECISION_THR = float(os.environ.get("NEM_DECISION_THR", "0.20"))
USE_TTA      = os.environ.get("NEM_USE_TTA", "1").lower() in ("1", "true", "yes")
USE_TEMP     = os.environ.get("NEM_USE_TEMP", "1").lower() in ("1", "true", "yes")

USE_PREDICTED = True
IG_N_STEPS = os.environ.get("IG_N_STEPS", "")
if IG_N_STEPS in ("", "None"):
    IG_N_STEPS = None
else:
    IG_N_STEPS = int(IG_N_STEPS)

RISE_N_MASKS = os.environ.get("RISE_N_MASKS", "")
if RISE_N_MASKS in ("", "None"):
    RISE_N_MASKS = None
else:
    RISE_N_MASKS = int(RISE_N_MASKS)

MON_NR_SAMPLES = int(os.environ.get("MON_NR_SAMPLES", "3"))
MON_STEPS      = int(os.environ.get("MON_STEPS", "32"))
MON_EPS        = float(os.environ.get("MON_EPS", "1e-5"))
POS_LABEL_FOR_MONO = int(os.environ.get("MON_POS_LABEL", "1"))
MON_BATCH      = int(os.environ.get("MON_BATCH", "32"))

NORMALISE_ATTR_TO_01 = True
MIN_STD_EPS = 1e-12

OUT_DIR = Path("experiments") / dataset_name / model_name / "eval_detailed_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "top_examples").mkdir(parents=True, exist_ok=True)

TOPK_WORST_NEM = int(os.environ.get("DETAIL_TOPK_WORST_NEM", "16"))
TOPK_BEST_NEM  = int(os.environ.get("DETAIL_TOPK_BEST_NEM", "16"))
TOPK_IG_BETTER = int(os.environ.get("DETAIL_TOPK_IG_BETTER", "16"))
TOPK_NEM_BETTER= int(os.environ.get("DETAIL_TOPK_NEM_BETTER", "16"))
TOPK_BEST_IG   = int(os.environ.get("DETAIL_TOPK_BEST_IG", str(TOPK_BEST_NEM)))
TOPK_WORST_IG  = int(os.environ.get("DETAIL_TOPK_WORST_IG", str(TOPK_WORST_NEM)))

TOPK_RISE_BETTER        = int(os.environ.get("DETAIL_TOPK_RISE_BETTER", "16"))      # rise_better_vs_nem
TOPK_NEM_BETTER_RISE    = int(os.environ.get("DETAIL_TOPK_NEM_BETTER_RISE", "16"))  # nem_better_vs_rise
TOPK_BEST_RISE          = int(os.environ.get("DETAIL_TOPK_BEST_RISE", "16"))
TOPK_WORST_RISE         = int(os.environ.get("DETAIL_TOPK_WORST_RISE", "16"))

print(f"[info] DETAIL_STYLE={DETAIL_STYLE}")
print(f"[info] Using dataset='{dataset_name}', model='{model_name}'")
print(f"[info] OUT_DIR = {OUT_DIR}")
print(f"[info] Filters: POS_MIN_PROB={POS_MIN_PROB} POS_REQUIRE_CORRECT={POS_REQUIRE_CORRECT}")
print(f"[info] Explain: IG_N_STEPS={IG_N_STEPS} RISE_N_MASKS={RISE_N_MASKS}")
print(f"[info] Monotonicity: NR_SAMPLES={MON_NR_SAMPLES} STEPS={MON_STEPS} MON_BATCH={MON_BATCH}")
print(f"[info] Forward: USE_TTA={USE_TTA} USE_TEMP={USE_TEMP} DECISION_THR={DECISION_THR}")
print(f"[info] Progress: DETAIL_PROGRESS_EVERY={PROG_EVERY} MON_PROGRESS_EVERY={MON_PROG_EVERY}")

data_obj = CHOSEN_DATASETS[dataset_name]()
train_loader, val_loader = data_obj.get_data()
if val_loader is None:
    raise RuntimeError("Validation data loader not found.")

model = CHOSEN_MODELS[model_name]().eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"[info] Model device: {device}")

IG_explainer   = itg3d_atr(model, train_data=None, use_predicted_labels=USE_PREDICTED)
RISE_explainer = rs3d_atr(model, train_data=None, use_predicted_labels=USE_PREDICTED)

if IG_N_STEPS is not None:
    try:
        IG_explainer.n_steps = int(IG_N_STEPS)
        print(f"[info] IG n_steps overridden to {IG_explainer.n_steps} via IG_N_STEPS")
    except Exception as e:
        print(f"[warn] Could not override IG n_steps: {e}")

if RISE_N_MASKS is not None:
    try:
        RISE_explainer.n_masks = int(RISE_N_MASKS)
        print(f"[info] RISE n_masks overridden to {RISE_explainer.n_masks} via RISE_N_MASKS")
    except Exception as e:
        print(f"[warn] Could not override RISE n_masks: {e}")

try:
    nem_explainer = NEMT3DMethod(model, train_loader=None, train_or_load=False, device=str(device))
except FileNotFoundError as e:
    raise RuntimeError(f"NEM checkpoint not found: {e}")

print("[info] Explainers initialized (IG, RISE, NEM).")

def _tta_logits_3d(model_t: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = []
    logits.append(model_t(x))
    logits.append(model_t(torch.flip(x, dims=[2])))  # flip D
    logits.append(model_t(torch.flip(x, dims=[3])))  # flip H
    logits.append(model_t(torch.flip(x, dims=[4])))  # flip W
    return torch.stack([z.view(-1) for z in logits], dim=0).mean(dim=0)  # [B]

def dhw_to_hw2d(a3d: np.ndarray) -> np.ndarray:
    if a3d.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {a3d.shape}")
    D, H, W = a3d.shape
    return a3d.reshape(D * H, W)

def to_3d_volume(vol: np.ndarray) -> np.ndarray:
    arr = np.asarray(vol)
    if arr.ndim == 5:
        arr = arr[0]  # [C,D,H,W]
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.mean(axis=0)
    if arr.ndim != 3:
        raise ValueError(f"to_3d_volume expected 3D, got {arr.shape}")
    return arr.astype(np.float32)

def _prep_map(a: np.ndarray, how: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if how == "abs":
        a = np.abs(a)
    elif how == "raw":
        pass
    elif how == "relu":
        a = np.maximum(a, 0.0)
    else:
        raise ValueError(f"Unknown prep mode: {how}")

    if NORMALISE_ATTR_TO_01:
        lo, hi = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < MIN_STD_EPS:
            return np.zeros_like(a, dtype=np.float32)
        a = (a - lo) / (hi - lo + 1e-9)
    return a

PREP = {"IG": "abs", "RISE": "raw", "NEM": "raw"}

def _extract_meta(full_dataset: Any, gi: int) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"gi": int(gi)}
    if hasattr(full_dataset, "rows"):
        row = full_dataset.rows[gi]
        if isinstance(row, dict):
            # keep small scalar-ish metadata
            for k, v in row.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    meta[k] = v
    return meta

def _get_full_dataset_and_indices(vloader) -> Tuple[Any, List[int]]:
    vds = vloader.dataset
    if hasattr(vds, "indices"):
        full = vds.dataset
        idxs = list(vds.indices)
        print(f"[info] val subset size={len(idxs)} (wrapped), full={len(full)}")
    else:
        full = vds
        idxs = list(range(len(full)))
        print(f"[info] val dataset size={len(full)} (not wrapped)")
    return full, idxs

def _get_item(full_dataset: Any, gi: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    item = full_dataset[gi]
    meta = _extract_meta(full_dataset, gi)

    if isinstance(item, (tuple, list)):
        if len(item) >= 2:
            X, y = item[0], item[1]
            if len(item) >= 3 and isinstance(item[2], dict):
                meta.update(item[2])
        else:
            raise ValueError(f"Dataset item length <2 at gi={gi}: {type(item)}")
    elif isinstance(item, dict):
        X = item["x"]
        y = item["y"]
        meta.update({k: v for k, v in item.items() if k not in ("x", "y")})
    else:
        raise ValueError(f"Unsupported dataset item type at gi={gi}: {type(item)}")

    if torch.is_tensor(y):
        y_int = int(y.item())
    else:
        y_int = int(y)

    if not torch.is_tensor(X):
        X = torch.as_tensor(X)
    # Expect [1,D,H,W] or [D,H,W]
    if X.ndim == 3:
        X = X.unsqueeze(0)
    return X, y_int, meta

def _forward_info(model_t: torch.nn.Module, X_sample_5d: torch.Tensor) -> Dict[str, Any]:
    with torch.inference_mode():
        logits = _tta_logits_3d(model_t, X_sample_5d) if USE_TTA else model_t(X_sample_5d)

    info: Dict[str, Any] = {"logits_shape": tuple(logits.shape)}

    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.view(-1)
    elif logits.ndim == 1:
        logits = logits.view(-1)

    # binary head
    if logits.ndim == 1:
        if USE_TEMP:
            prob_pos_t = torch.sigmoid(logits / float(TEMPERATURE))
        else:
            prob_pos_t = torch.sigmoid(logits)

        prob_pos = float(prob_pos_t[0].item())
        logit_pos = float(logits[0].item())
        pred_label = int(prob_pos >= DECISION_THR)
        pred_prob = prob_pos if pred_label == 1 else (1.0 - prob_pos)

        info.update({
            "logit_pos": logit_pos,
            "prob_pos": prob_pos,
            "pred_label": pred_label,
            "pred_prob": pred_prob,
        })
        return info

    # multiclass fallback
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1)
    pred_label = int(pred_idx.item())
    pred_prob = float(probs[0, pred_label].item())
    info.update({"pred_label": pred_label, "pred_prob": pred_prob})
    return info

def quantus_per_sample(metric_obj, x_batch, a_batch, y_batch, model=None) -> np.ndarray:
    last_err = None
    for kwargs in (
        dict(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch),
        dict(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, channel_first=True),
    ):
        try:
            out = metric_obj(**kwargs)
            return np.asarray(out, dtype=np.float32)
        except TypeError as e:
            last_err = e
    try:
        out = metric_obj(model, x_batch, y_batch, a_batch)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        raise last_err

def _spearmanr_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape
    N, P = a.shape
    scores = np.zeros(N, dtype=np.float64)
    for i in range(N):
        x = a[i]
        y = b[i]
        ox = np.argsort(x)
        rx = np.empty_like(ox, dtype=np.float64)
        rx[ox] = np.arange(P, dtype=np.float64)

        oy = np.argsort(y)
        ry = np.empty_like(oy, dtype=np.float64)
        ry[oy] = np.arange(P, dtype=np.float64)

        rx = rx - rx.mean()
        ry = ry - ry.mean()
        num = float(np.sum(rx * ry))
        den = float(math.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
        scores[i] = num / den if den > 0 else 0.0
    return scores.astype(np.float32)

def monotonicity_corr_gpu_core(
    model_torch: torch.nn.Module,
    x_batch_5d: np.ndarray,       # (N,1,D,H,W)
    y_target_np: np.ndarray,      # (N,)
    a_batch_5d: np.ndarray,       # (N,1,D,H,W)
    device: torch.device,
    nr_samples: int,
    steps: int,
    eps: float,
    rng: np.random.Generator,
    step_prefix: str = "",
) -> np.ndarray:
    model_torch.eval()

    x = x_batch_5d.astype(np.float32, copy=False)
    a = a_batch_5d.astype(np.float32, copy=False)

    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    a_flat = a.reshape(N, -1)
    n_features = a_flat.shape[1]

    steps = max(1, int(steps))
    features_in_step = max(1, n_features // steps)
    n_perturbations = int(math.ceil(n_features / features_in_step))

    # base preds (no grad)
    with torch.inference_mode():
        x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        logits = model_torch(x_tensor)

        if logits.ndim == 2:
            C = logits.shape[1]
            if C == 1:
                y_pred_t = logits.view(N)
            else:
                idx = torch.as_tensor(y_target_np, dtype=torch.long, device=device)
                y_pred_t = logits[torch.arange(N, device=device), idx]
        elif logits.ndim == 1:
            y_pred_t = logits
        else:
            y_pred_t = logits.view(N)

        y_pred = y_pred_t.detach().cpu().numpy().astype(np.float32)

    inv_pred = np.ones(N, dtype=np.float32)
    m = np.abs(y_pred) >= float(eps)
    inv_pred[m] = 1.0 / np.abs(y_pred[m])
    inv_pred = inv_pred ** 2

    # sort by attribution (ascending)
    a_indices = np.argsort(a_flat, axis=1)

    atts_list: List[np.ndarray] = []
    vars_list: List[np.ndarray] = []

    x_min = x_flat.min(axis=1, keepdims=True)
    x_max = x_flat.max(axis=1, keepdims=True)

    t0_steps = time.perf_counter()

    for step_idx in range(n_perturbations):
        start = step_idx * features_in_step
        end = min((step_idx + 1) * features_in_step, n_features)
        if start >= end:
            break

        a_ix = a_indices[:, start:end]  # (N, K)
        step_atts = np.take_along_axis(a_flat, a_ix, axis=1).sum(axis=1)
        atts_list.append(step_atts)

        y_pred_perturbs = []
        for _ in range(int(nr_samples)):
            baseline_vals = rng.uniform(x_min, x_max, size=a_ix.shape).astype(np.float32)
            x_pert_flat = x_flat.copy()
            np.put_along_axis(x_pert_flat, a_ix, baseline_vals, axis=1)
            x_pert = x_pert_flat.reshape(x.shape)

            with torch.inference_mode():
                x_t = torch.from_numpy(x_pert).to(device=device, dtype=torch.float32)
                logits_p = model_torch(x_t)

                if logits_p.ndim == 2:
                    C = logits_p.shape[1]
                    if C == 1:
                        y_p_t = logits_p.view(N)
                    else:
                        idx = torch.as_tensor(y_target_np, dtype=torch.long, device=device)
                        y_p_t = logits_p[torch.arange(N, device=device), idx]
                elif logits_p.ndim == 1:
                    y_p_t = logits_p
                else:
                    y_p_t = logits_p.view(N)

                y_p = y_p_t.detach().cpu().numpy().astype(np.float32)

            y_pred_perturbs.append(y_p)

        y_pred_perturbs = np.stack(y_pred_perturbs, axis=1)  # (N, nr_samples)
        step_vars = np.mean((y_pred_perturbs - y_pred[:, None]) ** 2, axis=1) * inv_pred
        vars_list.append(step_vars)

        if MON_PROG_EVERY and (((step_idx + 1) % MON_PROG_EVERY == 0) or (step_idx + 1 == n_perturbations)):
            progress_line(f"{step_prefix}monotonicity steps", step_idx + 1, n_perturbations, t0_steps)

    atts = np.stack(atts_list, axis=1)
    vars_ = np.stack(vars_list, axis=1)
    return _spearmanr_batch(atts, vars_)

def monotonicity_corr_gpu(
    model_torch: torch.nn.Module,
    x_batch_5d: np.ndarray,
    y_target_np: np.ndarray,
    a_batch_5d: np.ndarray,
    device: torch.device,
    nr_samples: int,
    steps: int,
    eps: float,
    batch_size: int,
    seed: int,
    prefix: str,
) -> np.ndarray:
    N = x_batch_5d.shape[0]
    if batch_size <= 0 or batch_size >= N:
        rng = np.random.default_rng(seed)
        return monotonicity_corr_gpu_core(
            model_torch, x_batch_5d, y_target_np, a_batch_5d, device,
            nr_samples=nr_samples, steps=steps, eps=eps,
            rng=rng, step_prefix=f"{prefix}: "
        )

    scores = np.zeros(N, dtype=np.float32)
    t0 = time.perf_counter()
    n_done = 0
    chunk_id = 0
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        chunk_id += 1
        rng = np.random.default_rng(seed + 1337 * chunk_id)

        sc = monotonicity_corr_gpu_core(
            model_torch,
            x_batch_5d[s:e],
            y_target_np[s:e],
            a_batch_5d[s:e],
            device,
            nr_samples=nr_samples,
            steps=steps,
            eps=eps,
            rng=rng,
            step_prefix=f"{prefix} (chunk {chunk_id}): ",
        )
        scores[s:e] = sc
        n_done = e
        if (chunk_id % 1 == 0) and ((n_done % (PROG_EVERY * batch_size) == 0) or (n_done == N)):
            progress_line(prefix, n_done, N, t0)

    return scores

stage("Load dataset indices + metadata scan for positives")
full_dataset, val_indices = _get_full_dataset_and_indices(val_loader)

# fast label scan to find positives
pos_idx: List[int] = []
max_scan = min(len(val_indices), SCAN_MAX_VAL) if SCAN_MAX_VAL > 0 else len(val_indices)

t0_scan = time.perf_counter()
if hasattr(full_dataset, "rows"):
    for j, gi in enumerate(val_indices[:max_scan], 1):
        row = full_dataset.rows[gi]
        try:
            lbl = int(row.get("label", 0)) if isinstance(row, dict) else 0
        except Exception:
            lbl = 0
        if lbl == 1:
            pos_idx.append(gi)
        if (j % 5000 == 0) or (j == max_scan):
            progress_line("metadata-scan", j, max_scan, t0_scan)
else:
    print("[warn] full_dataset.rows missing; label scan will load samples (slow).")
    for j, gi in enumerate(val_indices[:max_scan], 1):
        _, y_int, _ = _get_item(full_dataset, gi)
        if int(y_int) == 1:
            pos_idx.append(gi)
        if (j % 200 == 0) or (j == max_scan):
            progress_line("label-scan (loaded)", j, max_scan, t0_scan)

print(f"[info] Metadata scan window={max_scan}: pos={len(pos_idx)}", flush=True)
if len(pos_idx) == 0:
    print("[error] No positives found. Check labels in dataset rows.")
    sys.exit(1)

# shuffle and cap pool
rng = np.random.default_rng(SEED)
rng.shuffle(pos_idx)
if POS_POOL_MAX > 0:
    pos_idx = pos_idx[:min(POS_POOL_MAX, len(pos_idx))]
print(f"[info] Positive candidate pool size={len(pos_idx)} (capped by DETAIL_POS_POOL_MAX={POS_POOL_MAX})", flush=True)


stage("Prediction scan on positives (with caching)")
cand_infos: List[Dict[str, Any]] = []
skipped_low_prob = 0
skipped_incorrect = 0

t0_pred = time.perf_counter()
for j, gi in enumerate(pos_idx, 1):
    X_patch, y_int, meta = _get_item(full_dataset, gi)

    # cache X on CPU
    X_cpu = X_patch.detach().cpu().contiguous()  # [1,D,H,W]
    X_sample = X_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,1,D,H,W]

    fwd = _forward_info(model, X_sample)
    pred = int(fwd["pred_label"])
    pred_prob = float(fwd.get("pred_prob", np.nan))
    prob_pos = float(fwd.get("prob_pos", np.nan))
    logit_pos = float(fwd.get("logit_pos", np.nan))
    correct = int(pred == int(y_int))

    if (not np.isnan(prob_pos)) and (prob_pos < POS_MIN_PROB):
        skipped_low_prob += 1
    elif POS_REQUIRE_CORRECT and correct != 1:
        skipped_incorrect += 1
    else:
        meta2 = dict(meta)
        meta2.update({
            "label": int(y_int),
            "pred_label": pred,
            "pred_prob": pred_prob,
            "prob_pos": prob_pos,
            "logit_pos": logit_pos,
            "correct": correct,
        })
        cand_infos.append({
            "gi": int(gi),
            "label": int(y_int),
            "pred_label": pred,
            "pred_prob": pred_prob,
            "prob_pos": prob_pos,
            "logit_pos": logit_pos,
            "correct": correct,
            "X_cpu": X_cpu,     # cached patch
            "meta": meta2,
        })

    if (j % PROG_EVERY == 0) or (j == len(pos_idx)):
        progress_line("prediction-scan (pos)", j, len(pos_idx), t0_pred)

print(
    f"[info] Pos after filters: kept={len(cand_infos)} / {len(pos_idx)} "
    f"(skipped_low_prob={skipped_low_prob}, skipped_incorrect={skipped_incorrect})",
    flush=True,
)
if len(cand_infos) == 0:
    print("[error] No positives passed filters. Relax DETAIL_POS_MIN_PROB / DETAIL_POS_REQUIRE_CORRECT.")
    sys.exit(1)

# pick top-N by prob_pos
cand_infos = sorted(cand_infos, key=lambda d: (d["prob_pos"] if not np.isnan(d["prob_pos"]) else -1e9), reverse=True)
selected_infos = cand_infos[:min(N_POS, len(cand_infos))]
print(f"[info] Selected positives for detailed eval: {len(selected_infos)} (DETAIL_N_POS={N_POS})", flush=True)


stage("Compute attributions (IG / RISE / NEM) with progress")
metas: List[Dict[str, Any]] = []
labels: List[int] = []
pred_labels: List[int] = []
pred_probs: List[float] = []
prob_pos_list: List[float] = []
logit_pos_list: List[float] = []
correct_list: List[int] = []

timing_ig: List[float] = []
timing_rise: List[float] = []
timing_nem: List[float] = []

# stack material
x_list_5d = []
x_list_2d = []
ig_list_5d = []
rise_list_5d = []
nem_list_5d = []
ig_list_2d = []
rise_list_2d = []
nem_list_2d = []

missing_ig = 0
missing_rise = 0
missing_nem = 0
dropped = 0

def _cuda_sync():
    if device.type == "cuda":
        torch.cuda.synchronize()

t0_attr = time.perf_counter()
for k, rec in enumerate(selected_infos, 1):
    gi = int(rec["gi"])
    y_int = int(rec["label"])
    meta = dict(rec["meta"])

    X_cpu: torch.Tensor = rec["X_cpu"]  # [1,D,H,W]
    X_sample = X_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,1,D,H,W]

    # store X for later stacks
    vol5d = X_sample.detach().cpu().numpy().astype(np.float32)  # (1,1,D,H,W)
    vol3d = to_3d_volume(vol5d)
    x2d = dhw_to_hw2d(vol3d)[None, ...].astype(np.float32)  # (1,H2D,W)

    # IG
    ig_ok = True
    try:
        _cuda_sync()
        t0 = time.perf_counter()
        ig_attr = IG_explainer.gen_attr(X_sample)
        _cuda_sync()
        timing_ig.append(time.perf_counter() - t0)

        ig_attr = _prep_map(ig_attr, PREP["IG"])
        ig3d = to_3d_volume(ig_attr)
        ig2d = dhw_to_hw2d(ig3d)[None, ...].astype(np.float32)

        ig_list_5d.append(ig3d[None, None, ...])  # (1,1,D,H,W)
        ig_list_2d.append(ig2d[None, ...])        # (1,1,H2D,W)
    except Exception as e:
        ig_ok = False
        missing_ig += 1
        meta["ig_error"] = str(e)

    # RISE
    rise_ok = True
    try:
        _cuda_sync()
        t0 = time.perf_counter()
        rise_attr = RISE_explainer.gen_attr(X_sample, None)
        _cuda_sync()
        timing_rise.append(time.perf_counter() - t0)

        rise_attr = _prep_map(rise_attr, PREP["RISE"])
        rise3d = to_3d_volume(rise_attr)
        rise2d = dhw_to_hw2d(rise3d)[None, ...].astype(np.float32)

        rise_list_5d.append(rise3d[None, None, ...])
        rise_list_2d.append(rise2d[None, ...])
    except Exception as e:
        rise_ok = False
        missing_rise += 1
        meta["rise_error"] = str(e)

    # NEM
    nem_ok = True
    try:
        _cuda_sync()
        t0 = time.perf_counter()
        mask_logits, X_masked, mask_tensor = nem_explainer.gen_mask(X_sample)
        _cuda_sync()
        timing_nem.append(time.perf_counter() - t0)

        mask_np = mask_tensor.detach().cpu().numpy()
        keep3d = to_3d_volume(mask_np)
        nem3d = 1.0 - keep3d
        nem3d = _prep_map(nem3d, PREP["NEM"])
        nem3d = to_3d_volume(nem3d)
        nem2d = dhw_to_hw2d(nem3d)[None, ...].astype(np.float32)

        nem_list_5d.append(nem3d[None, None, ...])
        nem_list_2d.append(nem2d[None, ...])

        try:
            if torch.is_tensor(mask_logits):
                meta["nem_mask_logit"] = float(mask_logits.view(-1)[0].item())
        except Exception:
            pass
    except Exception as e:
        nem_ok = False
        missing_nem += 1
        meta["nem_error"] = str(e)

    # Keep only if ALL three succeeded
    if ig_ok and rise_ok and nem_ok:
        metas.append(meta)
        labels.append(y_int)
        pred_labels.append(int(meta.get("pred_label", rec["pred_label"])))
        pred_probs.append(float(meta.get("pred_prob", rec["pred_prob"])))
        prob_pos_list.append(float(meta.get("prob_pos", rec["prob_pos"])))
        logit_pos_list.append(float(meta.get("logit_pos", rec["logit_pos"])))
        correct_list.append(int(meta.get("correct", rec["correct"])))

        x_list_5d.append(vol5d)
        x_list_2d.append(x2d[None, ...])  # (1,1,H2D,W)
    else:
        dropped += 1
        # drop last appended method arrays to keep alignment
        if ig_ok:
            ig_list_5d.pop()
            ig_list_2d.pop()
        if rise_ok:
            rise_list_5d.pop()
            rise_list_2d.pop()
        if nem_ok:
            nem_list_5d.pop()
            nem_list_2d.pop()

    if (k % PROG_EVERY == 0) or (k == len(selected_infos)):
        progress_line("attributions (IG+RISE+NEM)", k, len(selected_infos), t0_attr)
        if len(timing_ig) > 0 and len(timing_rise) > 0 and len(timing_nem) > 0:
            print(
                f"[timing] avg/kept: IG={np.mean(timing_ig):.3f}s | "
                f"RISE={np.mean(timing_rise):.3f}s | NEM={np.mean(timing_nem):.3f}s",
                flush=True,
            )

print(f"[info] Kept {len(metas)} samples with IG+RISE+NEM. Dropped={dropped}.", flush=True)
print(f"[info] Missing: IG={missing_ig}, RISE={missing_rise}, NEM={missing_nem}", flush=True)

if len(metas) == 0:
    print("[error] No samples with all three explanations.")
    sys.exit(1)


stage("Stack arrays for metrics")
x_batch_5d = np.concatenate(x_list_5d, axis=0).astype(np.float32)     # (N,1,D,H,W)
x_batch_2d = np.concatenate(x_list_2d, axis=0).astype(np.float32)     # (N,1,H2D,W)

ig_batch_5d   = np.concatenate(ig_list_5d, axis=0).astype(np.float32)   # (N,1,D,H,W)
rise_batch_5d = np.concatenate(rise_list_5d, axis=0).astype(np.float32)
nem_batch_5d  = np.concatenate(nem_list_5d, axis=0).astype(np.float32)

ig_batch_2d   = np.concatenate(ig_list_2d, axis=0).astype(np.float32)   # (N,1,H2D,W)
rise_batch_2d = np.concatenate(rise_list_2d, axis=0).astype(np.float32)
nem_batch_2d  = np.concatenate(nem_list_2d, axis=0).astype(np.float32)

y_true = np.asarray(labels, dtype=int)
y_target_for_mono = np.full_like(y_true, POS_LABEL_FOR_MONO)


stage("Quantus per-sample metrics (complexity family)")
metrics = {
    "complexity":            quantus.Complexity(return_aggregate=False, disable_warnings=True),
    "sparseness":            quantus.Sparseness(return_aggregate=False, disable_warnings=True),
    "effective_complexity":  quantus.EffectiveComplexity(return_aggregate=False, disable_warnings=True),
}

q_ig: Dict[str, np.ndarray] = {}
q_rise: Dict[str, np.ndarray] = {}
q_nem: Dict[str, np.ndarray] = {}

t0_q = time.perf_counter()
for m_i, (metric_name, metric_obj) in enumerate(metrics.items(), 1):
    q_ig[metric_name] = quantus_per_sample(metric_obj, x_batch_2d, ig_batch_2d, y_batch=y_true, model=None)
    q_rise[metric_name] = quantus_per_sample(metric_obj, x_batch_2d, rise_batch_2d, y_batch=y_true, model=None)
    q_nem[metric_name] = quantus_per_sample(metric_obj, x_batch_2d, nem_batch_2d, y_batch=y_true, model=None)
    progress_line("quantus metrics", m_i, len(metrics), t0_q)


stage("Monotonicity-corr per-sample (GPU) with progress")
t0_mono = time.perf_counter()

mono_ig = monotonicity_corr_gpu(
    model_torch=model,
    x_batch_5d=x_batch_5d,
    y_target_np=y_target_for_mono,
    a_batch_5d=ig_batch_5d,
    device=device,
    nr_samples=MON_NR_SAMPLES,
    steps=MON_STEPS,
    eps=MON_EPS,
    batch_size=MON_BATCH,
    seed=SEED + 10,
    prefix="monotonicity IG",
)
progress_line("monotonicity IG", x_batch_5d.shape[0], x_batch_5d.shape[0], t0_mono)

mono_rise = monotonicity_corr_gpu(
    model_torch=model,
    x_batch_5d=x_batch_5d,
    y_target_np=y_target_for_mono,
    a_batch_5d=rise_batch_5d,
    device=device,
    nr_samples=MON_NR_SAMPLES,
    steps=MON_STEPS,
    eps=MON_EPS,
    batch_size=MON_BATCH,
    seed=SEED + 20,
    prefix="monotonicity RISE",
)
progress_line("monotonicity RISE", x_batch_5d.shape[0], x_batch_5d.shape[0], t0_mono)

mono_nem = monotonicity_corr_gpu(
    model_torch=model,
    x_batch_5d=x_batch_5d,
    y_target_np=y_target_for_mono,
    a_batch_5d=nem_batch_5d,
    device=device,
    nr_samples=MON_NR_SAMPLES,
    steps=MON_STEPS,
    eps=MON_EPS,
    batch_size=MON_BATCH,
    seed=SEED + 30,
    prefix="monotonicity NEM",
)
progress_line("monotonicity NEM", x_batch_5d.shape[0], x_batch_5d.shape[0], t0_mono)


stage("Build per-sample table + save CSV")
rows = []
for i, meta in enumerate(metas):
    r = dict(meta)

    # Quantus metrics
    for metric_name in metrics.keys():
        ig_v = float(q_ig[metric_name][i])
        rise_v = float(q_rise[metric_name][i])
        nem_v = float(q_nem[metric_name][i])

        r[f"ig_{metric_name}"] = ig_v
        r[f"rise_{metric_name}"] = rise_v
        r[f"nem_{metric_name}"] = nem_v
        r[f"delta_{metric_name}"] = float(nem_v - ig_v)
        r[f"delta_rise_vs_nem_{metric_name}"] = float(rise_v - nem_v)

    # Monotonicity
    ig_m = float(mono_ig[i])
    rise_m = float(mono_rise[i])
    nem_m = float(mono_nem[i])

    r["ig_monotonicity_corr"] = ig_m
    r["rise_monotonicity_corr"] = rise_m
    r["nem_monotonicity_corr"] = nem_m
    r["delta_monotonicity_corr"] = float(nem_m - ig_m)
    r["delta_rise_vs_nem_monotonicity_corr"] = float(rise_m - nem_m)

    rows.append(r)

df = pd.DataFrame(rows)

# Ensure main columns exist
for col in ["gi", "label", "pred_label", "pred_prob", "prob_pos", "logit_pos", "correct"]:
    if col not in df.columns:
        df[col] = np.nan

csv_path = OUT_DIR / "per_sample_scores.csv"
df.to_csv(csv_path, index=False)
print(f"[output] Saved per-sample table: {csv_path}", flush=True)

stage("Save top-k example bundles (.npz)")

def _save_npz_example(out_path: Path, idx: int):
    meta = rows[idx]
    payload = {
        "x_5d": x_batch_5d[idx:idx+1],              # (1,1,D,H,W)
        "ig_attr_5d": ig_batch_5d[idx:idx+1],
        "rise_attr_5d": rise_batch_5d[idx:idx+1],
        "nem_attr_5d": nem_batch_5d[idx:idx+1],
        "y_true": np.array([y_true[idx]], dtype=np.int64),
        "y_pred": np.array([int(df.loc[idx, "pred_label"])], dtype=np.int64),
        "meta_json": np.array([json.dumps(meta)], dtype=object),
    }
    np.savez_compressed(out_path, **payload)

def _save_topk(category: str, df_sorted: pd.DataFrame, k: int):
    subdir = OUT_DIR / "top_examples" / category
    subdir.mkdir(parents=True, exist_ok=True)

    if df_sorted is None or len(df_sorted) == 0:
        print(f"[output] Saved {category}: (no rows)", flush=True)
        return

    k = min(int(k), len(df_sorted))
    picked = df_sorted.head(k).copy()
    picked_path = subdir / f"{category}_top{k}.csv"
    picked.to_csv(picked_path, index=False)

    orig_indices = list(picked.index)
    for rank, orig_idx in enumerate(orig_indices):
        gi = df.loc[orig_idx, "gi"]
        out_path = subdir / f"{rank:03d}_gi{gi}.npz"
        _save_npz_example(out_path, int(orig_idx))

    print(f"[output] Saved {category}: csv={picked_path}, npz_dir={subdir}", flush=True)

# Ensure numeric
for c in [
    "nem_monotonicity_corr", "ig_monotonicity_corr", "rise_monotonicity_corr",
    "delta_monotonicity_corr", "delta_rise_vs_nem_monotonicity_corr"
]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

_save_topk("nem_best_pos",  df.sort_values("nem_monotonicity_corr", ascending=False), TOPK_BEST_NEM)
_save_topk("nem_worst_pos", df.sort_values("nem_monotonicity_corr", ascending=True),  TOPK_WORST_NEM)

_save_topk("ig_best_pos",   df.sort_values("ig_monotonicity_corr", ascending=False), TOPK_BEST_IG)
_save_topk("ig_worst_pos",  df.sort_values("ig_monotonicity_corr", ascending=True),  TOPK_WORST_IG)

_save_topk("ig_better_pos",  df.sort_values("delta_monotonicity_corr", ascending=True),  TOPK_IG_BETTER)
_save_topk("nem_better_pos", df.sort_values("delta_monotonicity_corr", ascending=False), TOPK_NEM_BETTER)

_save_topk("rise_best_pos",  df.sort_values("rise_monotonicity_corr", ascending=False), TOPK_BEST_RISE)
_save_topk("rise_worst_pos", df.sort_values("rise_monotonicity_corr", ascending=True),  TOPK_WORST_RISE)

_save_topk("rise_better_vs_nem_pos", df.sort_values("delta_rise_vs_nem_monotonicity_corr", ascending=False), TOPK_RISE_BETTER)
_save_topk("nem_better_vs_rise_pos", df.sort_values("delta_rise_vs_nem_monotonicity_corr", ascending=True),  TOPK_NEM_BETTER_RISE)

summary = {
    "n_kept": int(len(df)),
    "mean_ig_monotonicity": float(df["ig_monotonicity_corr"].mean()),
    "mean_rise_monotonicity": float(df["rise_monotonicity_corr"].mean()),
    "mean_nem_monotonicity": float(df["nem_monotonicity_corr"].mean()),
    "mean_delta_nem_minus_ig": float(df["delta_monotonicity_corr"].mean()),
    "mean_delta_rise_minus_nem": float(df["delta_rise_vs_nem_monotonicity_corr"].mean()),
    "missing_ig": int(missing_ig),
    "missing_rise": int(missing_rise),
    "missing_nem": int(missing_nem),
    "dropped_samples": int(dropped),
    "timing_avg_ig_s": float(np.mean(timing_ig)) if len(timing_ig) else None,
    "timing_avg_rise_s": float(np.mean(timing_rise)) if len(timing_rise) else None,
    "timing_avg_nem_s": float(np.mean(timing_nem)) if len(timing_nem) else None,
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"[output] Saved summary.json: {OUT_DIR / 'summary.json'}", flush=True)

print("[done] Detailed (positives-only) evaluation finished.", flush=True)
