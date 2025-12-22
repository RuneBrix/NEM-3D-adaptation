import os
import sys
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import quantus

from exp_config import CHOSEN_DATASETS, CHOSEN_MODELS
from attrs.intgrad3d import itg3d_atr
from attrs.rise3d import rs3d_atr
from attrs.nem_utils.method_nemt3d import NEMT3DMethod

dataset_name = os.environ.get("NEM_DATASETS", "luna3d").split(",")[0]
model_name = os.environ.get("NEM_MODELS", "phase3_3d").split(",")[0]

print(f"[info] Using dataset='{dataset_name}', model='{model_name}'")

DECISION_THR = float(os.environ.get("NEM_DECISION_THR", "0.20"))
USE_TTA = int(os.environ.get("NEM_USE_TTA", "1")) == 1
USE_TEMP = int(os.environ.get("NEM_USE_TEMP", "1")) == 1

from exp_config import TEMPERATURE

data_obj = CHOSEN_DATASETS[dataset_name]()        # instantiate dataset
train_loader, val_loader = data_obj.get_data()    # get train and validation loaders
if val_loader is None:
    raise RuntimeError("Validation data loader not found. Please ensure the dataset splits are set up correctly.")

model = CHOSEN_MODELS[model_name]().eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"[info] Model device: {device}")


use_predicted = True 

IG_explainer   = itg3d_atr(model, train_data=None, use_predicted_labels=use_predicted)
RISE_explainer = rs3d_atr(model, train_data=None, use_predicted_labels=use_predicted)

ig_n_steps_env   = os.environ.get("IG_N_STEPS", "")
rise_n_masks_env = os.environ.get("RISE_N_MASKS", "")

if ig_n_steps_env not in ("", "None"):
    try:
        IG_explainer.n_steps = int(ig_n_steps_env)
        print(f"[info] IG n_steps overridden to {IG_explainer.n_steps} via IG_N_STEPS")
    except ValueError:
        print(f"[warn] IG_N_STEPS='{ig_n_steps_env}' is not a valid int; keeping default {IG_explainer.n_steps}")

if rise_n_masks_env not in ("", "None"):
    try:
        RISE_explainer.n_masks = int(rise_n_masks_env)
        print(f"[info] RISE n_masks overridden to {RISE_explainer.n_masks} via RISE_N_MASKS")
    except ValueError:
        print(f"[warn] RISE_N_MASKS='{rise_n_masks_env}' is not a valid int; keeping default {RISE_explainer.n_masks}")

try:
    nem_explainer = NEMT3DMethod(model, train_loader=None, train_or_load=False, device=str(device))
except FileNotFoundError as e:
    raise RuntimeError(f"NEM checkpoint not found: {e}")

print("[info] Explainers initialized (Integrated Gradients, RISE, NEM).")

NORMALISE_ATTR_TO_01 = True
MIN_STD_EPS = 1e-12

def _tta_logits_3d(model, x):
    # x: [B,1,D,H,W]
    logits = []
    logits.append(model(x))
    logits.append(model(torch.flip(x, dims=[2])))  # flip D
    logits.append(model(torch.flip(x, dims=[3])))  # flip H
    logits.append(model(torch.flip(x, dims=[4])))  # flip W
    return torch.stack([z.view(-1) for z in logits], dim=0).mean(dim=0)  # [B]

def dhw_to_hw2d(a3d: np.ndarray) -> np.ndarray:
    """Flatten a 3D volume (D,H,W) into a 2D image (D*H, W) by stacking depth slices vertically."""
    if a3d.ndim != 3:
        raise ValueError(f"Expected 3D array (D,H,W), got shape {a3d.shape}")
    D, H, W = a3d.shape
    return a3d.reshape(D * H, W)

def _prep_map(a: np.ndarray, how: str) -> np.ndarray:
    """
    Prepare attribution map according to specified mode (abs/raw/relu)
    and normalize to [0,1] if enabled.
    """
    a = np.asarray(a, dtype=np.float32)
    if how == "abs":
        a = np.abs(a)
    elif how == "relu":
        a = np.maximum(a, 0.0)
    elif how == "raw":
        pass
    else:
        raise ValueError(f"Unknown prep mode: {how}")

    if NORMALISE_ATTR_TO_01:
        lo, hi = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < MIN_STD_EPS:
            return np.zeros_like(a, dtype=np.float32)
        a = (a - lo) / (hi - lo + 1e-9)
    return a

def to_3d_volume(vol: np.ndarray) -> np.ndarray:
    """
    Convert a volume with shape:
      - (B, C, D, H, W) or
      - (C, D, H, W) or
      - (D, H, W)
    into a 3D array (D, H, W).
    """
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

PREP = {
    "IntegratedGradients": "abs",
    "RISE": "raw",
    "NEM": "raw",
}

# Methods we intend to evaluate
methods = ["IntegratedGradients", "RISE", "NEM"]

# Timing + stats
timings        = {m: [] for m in methods}
missing_counts = {m: 0 for m in methods}

MON_NR_SAMPLES      = int(os.environ.get("MON_NR_SAMPLES", "10"))   # MC samples per step
MON_STEPS           = int(os.environ.get("MON_STEPS", "64"))        # perturbation steps
MON_EPS             = float(os.environ.get("MON_EPS", "1e-5"))
POS_LABEL           = int(os.environ.get("MON_POS_LABEL", "1"))     # target positive class
MON_MAX_POS_SAMPLES = int(os.environ.get("MON_MAX_POS_SAMPLES", "64"))
MON_REQUIRE_CORRECT = int(os.environ.get("MON_REQUIRE_CORRECT", "1"))
MON_MIN_PROB        = float(os.environ.get("MON_MIN_PROB", "0.5"))
MON_SCAN_MAX_VAL    = int(os.environ.get("MON_SCAN_MAX_VAL", "2000"))

def collect_monotonicity_subset(
    val_loader,
    model,
    device,
    IG_explainer,
    RISE_explainer,
    nem_explainer,
    methods,
    pos_label: int,
    max_pos_samples: int,
    min_prob: float,
    require_correct: int,
    scan_max_val: int,
):
    """
    Build a common evaluation subset used for BOTH complexity (Quantus)
    and monotonicity (faithfulness) metrics.

    1) Find all label=pos_label candidates in the validation subset.
    2) Forward pass over these positives to get prob_pos, pred_label.
    3) Filter by min_prob and (optionally) require_correct.
    4) Sort by prob_pos and keep at most max_pos_samples.
    5) For this final set, compute 3D attributions (for monotonicity)
       and 2D flattened maps (for complexity) for each XAI method.
    """

    val_dataset = val_loader.dataset
    if hasattr(val_dataset, "indices"):
        full_dataset = val_dataset.dataset
        val_indices = list(val_dataset.indices)
        print(f"[monotonicity] val subset size={len(val_indices)}, full dataset size={len(full_dataset)}")
    else:
        full_dataset = val_dataset
        val_indices = list(range(len(full_dataset)))
        print(f"[monotonicity] val dataset size={len(full_dataset)} (no subset wrapper).")

    # Sanity: count positives
    if hasattr(full_dataset, "rows"):
        total_pos = sum(
            int(full_dataset.rows[gi].get("label", 0) == pos_label)
            for gi in val_indices
        )
        print(
            f"[monotonicity] Sanity: validation subset has "
            f"{total_pos} label={pos_label} samples (before any filtering)."
        )
    pos_val_indices = []
    if hasattr(full_dataset, "rows"):
        for gi in val_indices:
            row = full_dataset.rows[gi]
            lbl = int(row.get("label", 0))
            if lbl == pos_label:
                pos_val_indices.append(gi)
    else:
        for gi in val_indices:
            _, y = full_dataset[gi]
            lbl = int(y.item()) if torch.is_tensor(y) else int(y)
            if lbl == pos_label:
                pos_val_indices.append(gi)

    print(
        f"[monotonicity] Found {len(pos_val_indices)} label={pos_label} "
        f"candidates in validation subset."
    )
    if len(pos_val_indices) == 0:
        return None, None, None, None, None, None

    # Optional: limit how many positive candidates we inspect
    if scan_max_val is not None and scan_max_val > 0 and len(pos_val_indices) > scan_max_val:
        print(
            f"[monotonicity] Limiting positive scan to first "
            f"{scan_max_val} candidates (from {len(pos_val_indices)})."
        )
        pos_val_indices = pos_val_indices[:scan_max_val]

    cand_infos = []
    skipped_low_prob = 0
    skipped_incorrect = 0

    with torch.no_grad():
        for gi in pos_val_indices:
            X_patch, y = full_dataset[gi]  # [1,D,H,W]
            label = int(y.item()) if torch.is_tensor(y) else int(y)
            X_sample = X_patch.unsqueeze(0).to(device)  # [1,1,D,H,W]

            logits = _tta_logits_3d(model, X_sample) if USE_TTA else model(X_sample).view(-1)  # [1]
            logit = float(logits[0].item())

            if USE_TEMP:
                prob_pos = float(torch.sigmoid(logits / float(TEMPERATURE))[0].item())
            else:
                prob_pos = float(torch.sigmoid(logits)[0].item())

            pred_label = int(prob_pos >= DECISION_THR)

            passes_prob = (prob_pos >= min_prob)
            passes_correct = (not require_correct) or (pred_label == label)

            if not passes_prob:
                skipped_low_prob += 1
                continue
            if not passes_correct:
                skipped_incorrect += 1
                continue

            cand_infos.append(
                dict(
                    gi=gi,
                    label=label,
                    pred_label=pred_label,
                    prob_pos=prob_pos,
                    logit=logit,
                )
            )

    print(
        f"[monotonicity] After prob/TP filter: kept {len(cand_infos)} "
        f"candidates (skipped_low_prob={skipped_low_prob}, "
        f"skipped_incorrect={skipped_incorrect})."
    )

    if len(cand_infos) == 0:
        return None, None, None, None, None, None

    cand_infos = sorted(cand_infos, key=lambda d: d["prob_pos"], reverse=True)
    cand_infos = cand_infos[:max_pos_samples]
    print(
        "[monotonicity] Using the following candidates for monotonicity and complexity:\n  " +
        "\n  ".join(
            f"gi={c['gi']} | label={c['label']} | pred={c['pred_label']} "
            f"| prob_pos={c['prob_pos']:.3f} | logit={c['logit']:.3f}"
            for c in cand_infos
        )
    )

    mon_x_list_5d = []
    mon_y_list = []

    x_list_2d = []
    y_list_complex = []

    mon_xai_3d_lists = {m: [] for m in methods}
    xai_arrays_2d_lists = {m: [] for m in methods}

    for ci_idx, ci in enumerate(cand_infos):
        gi = ci["gi"]

        X_patch, y = full_dataset[gi]
        X_sample = X_patch.unsqueeze(0).to(device)  # [1,1,D,H,W]

        vol5d = X_sample.detach().cpu().numpy().astype(np.float32)  # (1,1,D,H,W)
        vol3d = to_3d_volume(vol5d)                                  # (D,H,W)

        mon_x_list_5d.append(vol5d)
        mon_y_list.append(pos_label)

        flat2d = dhw_to_hw2d(vol3d)      # (H2D, W)
        x_list_2d.append(flat2d[None, ...])
        y_list_complex.append(pos_label)

        # Compute attributions for each method
        for m in methods:
            if m == "IntegratedGradients":
                try:
                    start_time = time.perf_counter()
                    ig_attr = IG_explainer.gen_attr(X_sample)
                    timings["IntegratedGradients"].append(time.perf_counter() - start_time)

                    ig_attr_prep = _prep_map(ig_attr, PREP["IntegratedGradients"])
                    ig3d = to_3d_volume(ig_attr_prep)

                    mon_xai_3d_lists["IntegratedGradients"].append(ig3d[None, ...])  # (1,D,H,W)
                    ig2d = dhw_to_hw2d(ig3d)
                    xai_arrays_2d_lists["IntegratedGradients"].append(ig2d[None, ...])
                except Exception as e:
                    print(f"[warning] IG attribution failed for gi={gi} (candidate {ci_idx}): {e}")
                    missing_counts["IntegratedGradients"] += 1
                    nan3d = np.full_like(vol3d, np.nan, dtype=np.float32)
                    mon_xai_3d_lists["IntegratedGradients"].append(nan3d[None, ...])
                    nan2d = dhw_to_hw2d(nan3d)
                    xai_arrays_2d_lists["IntegratedGradients"].append(nan2d[None, ...])

            elif m == "RISE":
                try:
                    start_time = time.perf_counter()
                    rise_attr = RISE_explainer.gen_attr(X_sample, None)
                    timings["RISE"].append(time.perf_counter() - start_time)

                    rise_attr_prep = _prep_map(rise_attr, PREP["RISE"])
                    rise3d = to_3d_volume(rise_attr_prep)

                    mon_xai_3d_lists["RISE"].append(rise3d[None, ...])
                    rise2d = dhw_to_hw2d(rise3d)
                    xai_arrays_2d_lists["RISE"].append(rise2d[None, ...])
                except Exception as e:
                    print(f"[warning] RISE attribution failed for gi={gi} (candidate {ci_idx}): {e}")
                    missing_counts["RISE"] += 1
                    nan3d = np.full_like(vol3d, np.nan, dtype=np.float32)
                    mon_xai_3d_lists["RISE"].append(nan3d[None, ...])
                    nan2d = dhw_to_hw2d(nan3d)
                    xai_arrays_2d_lists["RISE"].append(nan2d[None, ...])

            elif m == "NEM":
                try:
                    start_time = time.perf_counter()
                    mask_logits, X_masked, mask_tensor = nem_explainer.gen_mask(X_sample)
                    timings["NEM"].append(time.perf_counter() - start_time)

                    mask_np = mask_tensor.detach().cpu().numpy()
                    nem3d_keep = to_3d_volume(mask_np)
                    nem3d = 1.0 - nem3d_keep

                    nem_attr_prep = _prep_map(nem3d, PREP["NEM"])
                    nem3d_final = to_3d_volume(nem_attr_prep)

                    mon_xai_3d_lists["NEM"].append(nem3d_final[None, ...])
                    nem2d = dhw_to_hw2d(nem3d_final)
                    xai_arrays_2d_lists["NEM"].append(nem2d[None, ...])
                except Exception as e:
                    print(f"[warning] NEM attribution failed for gi={gi} (candidate {ci_idx}): {e}")
                    missing_counts["NEM"] += 1
                    nan3d = np.full_like(vol3d, np.nan, dtype=np.float32)
                    mon_xai_3d_lists["NEM"].append(nan3d[None, ...])
                    nem2d = dhw_to_hw2d(nan3d)
                    xai_arrays_2d_lists["NEM"].append(nem2d[None, ...])

    if len(mon_x_list_5d) == 0:
        return None, None, None, None, None, None

    mon_x_batch_5d = np.concatenate(mon_x_list_5d, axis=0).astype(np.float32)   # (N_pos,1,D,H,W)
    mon_y_batch = np.array(mon_y_list, dtype=int)

    x_batch_2d = np.stack(x_list_2d, axis=0).astype(np.float32)  # (N_pos,1,H2D,W)
    y_batch = np.array(y_list_complex, dtype=int)

    mon_xai_3d = {}
    xai_methods_complexity = {}

    for m in methods:
        maps3d_list = mon_xai_3d_lists[m]
        maps2d_list = xai_arrays_2d_lists[m]

        if len(maps3d_list) == 0 or len(maps2d_list) == 0:
            continue

        arr3d = np.concatenate(maps3d_list, axis=0).astype(np.float32)   # (N_pos,1,D,H,W)
        arr2d = np.stack(maps2d_list, axis=0).astype(np.float32)         # (N_pos,1,H2D,W)

        if np.isnan(arr3d).any() or np.isnan(arr2d).any():
            print(f"[quantus] Skipping {m}: NaNs in attributions for positives.")
            continue

        # Squeeze channel dim for 3D maps: (N_pos,D,H,W)
        if arr3d.ndim == 5 and arr3d.shape[1] == 1:
            arr3d_squeezed = arr3d[:, 0]
        else:
            arr3d_squeezed = arr3d

        mon_xai_3d[m] = arr3d_squeezed
        xai_methods_complexity[m] = arr2d

    print(
        f"[monotonicity] Final positive subset: x={mon_x_batch_5d.shape}, "
        f"y={mon_y_batch.shape}"
    )

    return mon_x_batch_5d, mon_y_batch, mon_xai_3d, x_batch_2d, y_batch, xai_methods_complexity


# Build the shared evaluation subset
mon_x_batch_5d, mon_y_batch, mon_xai_3d, x_batch_2d, y_batch, xai_methods_complexity = collect_monotonicity_subset(
    val_loader=val_loader,
    model=model,
    device=device,
    IG_explainer=IG_explainer,
    RISE_explainer=RISE_explainer,
    nem_explainer=nem_explainer,
    methods=methods,
    pos_label=POS_LABEL,
    max_pos_samples=MON_MAX_POS_SAMPLES,
    min_prob=MON_MIN_PROB,
    require_correct=MON_REQUIRE_CORRECT,
    scan_max_val=MON_SCAN_MAX_VAL,
)

if mon_x_batch_5d is None:
    print("[error] No validation positives passed the filters (probability / correctness).")
    print("        Check MON_MIN_PROB, MON_REQUIRE_CORRECT or validation set.")
    sys.exit(0)

print(f"[info] Collected {x_batch_2d.shape[0]} positive samples for complexity + monotonicity.")
print(f"[info] x_batch_2d shape: {x_batch_2d.shape}, y_batch shape: {y_batch.shape}")
print("[quantus] Methods included for Complexity evaluation:", list(xai_methods_complexity.keys()))
print("[quantus] Missing attribution counts:", missing_counts)


metrics_complexity = {
    "complexity":           quantus.Complexity(return_aggregate=True, disable_warnings=True),
    "sparseness":           quantus.Sparseness(return_aggregate=True, disable_warnings=True),
    "effective-complexity": quantus.EffectiveComplexity(return_aggregate=True, disable_warnings=True),
}

results_df = quantus.evaluate(
    metrics=metrics_complexity,
    xai_methods=xai_methods_complexity,
    model=None,              # model not needed for complexity metrics
    x_batch=x_batch_2d,
    y_batch=y_batch,
    agg_func=np.mean,
    return_as_df=True,
    verbose=True,
)

df = results_df.copy()
df.columns = [str(c).lower() for c in df.columns]

def _detect_schema(dframe: pd.DataFrame) -> str:
    cols = set(dframe.columns)
    if any(c in cols for c in ["metric", "metric_name", "name"]) and \
            any(c in cols for c in ["xai_method", "method", "xai"]) and \
            any(c in cols for c in ["score", "scores", "value"]):
        return "long"
    if dframe.index.dtype == object and len(cols) > 0:
        generic = {"fold", "seed", "run", "value", "score", "metric", "name"}
        if len(cols - generic) > 0:
            return "wide"
    return "unknown"

schema = _detect_schema(df)
if schema == "long":
    metric_col = next(c for c in ["metric", "metric_name", "name"] if c in df.columns)
    method_col = next(c for c in ["xai_method", "method", "xai"] if c in df.columns)
    score_col  = next(c for c in ["score", "scores", "value"] if c in df.columns)
    pivot_df = (
        df.groupby([metric_col, method_col])[score_col]
        .mean().reset_index()
        .pivot(index=metric_col, columns=method_col, values=score_col)
    )
elif schema == "wide":
    pivot_df = df.copy()
else:
    try:
        metric_col = next(c for c in ["metric", "metric_name", "name"] if c in df.columns)
        method_col = next(c for c in ["xai_method", "method", "xai"] if c in df.columns)
        score_col  = next(c for c in ["score", "scores", "value"] if c in df.columns)
        pivot_df = (
            df.groupby([metric_col, method_col])[score_col]
            .mean().reset_index()
            .pivot(index=metric_col, columns=method_col, values=score_col)
        )
    except Exception as e:
        print(f"[warn] Could not pivot results automatically: {e}")
        pivot_df = df

def _spearmanr_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Spearman rank correlation per row between a and b.
    a, b: (N, P)
    returns: (N,)
    """
    assert a.shape == b.shape
    N, P = a.shape
    scores = np.zeros(N, dtype=np.float64)

    for i in range(N):
        x = a[i]
        y = b[i]

        order_x = np.argsort(x)
        rank_x  = np.empty_like(order_x, dtype=np.float64)
        rank_x[order_x] = np.arange(P, dtype=np.float64)

        order_y = np.argsort(y)
        rank_y  = np.empty_like(order_y, dtype=np.float64)
        rank_y[order_y] = np.arange(P, dtype=np.float64)

        rx = rank_x - rank_x.mean()
        ry = rank_y - rank_y.mean()

        num = np.sum(rx * ry)
        den = math.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
        scores[i] = num / den if den > 0 else 0.0

    return scores.astype(np.float32)

def monotonicity_corr_gpu(
    model_torch: torch.nn.Module,
    x_batch_5d: np.ndarray,       # (N,1,D,H,W)
    y_batch_np: np.ndarray,       # (N,)
    a_batch_5d: np.ndarray,       # (N,1,D,H,W)
    device: torch.device,
    nr_samples: int = 10,
    steps: int = 64,
    eps: float = 1e-5,
) -> np.ndarray:
    model_torch.eval()

    x = x_batch_5d.astype(np.float32)
    a = a_batch_5d.astype(np.float32)

    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    a_flat = a.reshape(N, -1)
    n_features = a_flat.shape[1]

    steps = max(1, steps)
    features_in_step = max(1, n_features // steps)
    n_perturbations = math.ceil(n_features / features_in_step)

    with torch.no_grad():
        x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        logits   = model_torch(x_tensor)

        if logits.ndim == 2:
            C = logits.shape[1]
            if C == 1:
                y_pred_t = logits.view(N)
            else:
                idx = torch.as_tensor(y_batch_np, dtype=torch.long, device=device)
                y_pred_t = logits[torch.arange(N, device=device), idx]
        elif logits.ndim == 1:
            y_pred_t = logits
        else:
            y_pred_t = logits.view(N)

        y_pred = y_pred_t.detach().cpu().numpy().astype(np.float32)

    inv_pred = np.ones(N, dtype=np.float32)
    mask = np.abs(y_pred) >= eps
    inv_pred[mask] = 1.0 / np.abs(y_pred[mask])
    inv_pred = inv_pred ** 2

    a_indices = np.argsort(a_flat, axis=1)

    atts_list = []
    vars_list = []

    x_min = x_flat.min(axis=1, keepdims=True)
    x_max = x_flat.max(axis=1, keepdims=True)

    print(f"[monotonicity] n_features={n_features}, features_in_step={features_in_step}, steps={n_perturbations}")

    for step_idx in range(n_perturbations):
        start = step_idx * features_in_step
        end   = min((step_idx + 1) * features_in_step, n_features)
        if start >= end:
            break

        a_ix = a_indices[:, start:end]

        step_atts = np.take_along_axis(a_flat, a_ix, axis=1).sum(axis=1)
        atts_list.append(step_atts)

        y_pred_perturbs = []

        for _ in range(nr_samples):
            baseline_vals = np.random.uniform(x_min, x_max, size=a_ix.shape).astype(np.float32)

            x_pert_flat = x_flat.copy()
            for i in range(N):
                x_pert_flat[i, a_ix[i]] = baseline_vals[i]

            x_pert = x_pert_flat.reshape(x.shape)
            with torch.no_grad():
                x_t = torch.from_numpy(x_pert).to(device=device, dtype=torch.float32)
                logits_p = model_torch(x_t)

                if logits_p.ndim == 2:
                    C = logits_p.shape[1]
                    if C == 1:
                        y_p_t = logits_p.view(N)
                    else:
                        idx = torch.as_tensor(y_batch_np, dtype=torch.long, device=device)
                        y_p_t = logits_p[torch.arange(N, device=device), idx]
                elif logits_p.ndim == 1:
                    y_p_t = logits_p
                else:
                    y_p_t = logits_p.view(N)

                y_p = y_p_t.detach().cpu().numpy().astype(np.float32)

            y_pred_perturbs.append(y_p)

        y_pred_perturbs = np.stack(y_pred_perturbs, axis=1)
        step_vars = np.mean((y_pred_perturbs - y_pred[:, None]) ** 2, axis=1) * inv_pred
        vars_list.append(step_vars)

    atts = np.stack(atts_list, axis=1)
    vars_ = np.stack(vars_list, axis=1)

    scores = _spearmanr_batch(atts, vars_)
    return scores

monotonicity_scores = {}

if mon_x_batch_5d is None:
    print("[monotonicity] No valid positive samples found; skipping monotonicity.")
else:
    print(
        f"[monotonicity] Positive subset for monotonicity: "
        f"x={mon_x_batch_5d.shape}, y={mon_y_batch.shape}"
    )

    for name in xai_methods_complexity.keys():
        a_batch_3d = mon_xai_3d.get(name, None)
        if a_batch_3d is None:
            print(f"[monotonicity] Skipping {name}: no positive attributions collected.")
            continue

        if a_batch_3d.ndim == 4:
            a_batch_5d = a_batch_3d[:, None, ...]
        elif a_batch_3d.ndim == 5:
            a_batch_5d = a_batch_3d
        else:
            print(f"[monotonicity][warn] Unexpected shape for {name} attributions: {a_batch_3d.shape}, skipping.")
            continue

        if np.isnan(a_batch_5d).any():
            print(f"[monotonicity] {name}: NaNs in attributions for positives; skipping.")
            continue

        print(f"[monotonicity] Evaluating {name} on GPU (positives only)...")
        scores = monotonicity_corr_gpu(
            model_torch=model,
            x_batch_5d=mon_x_batch_5d,
            y_batch_np=mon_y_batch,
            a_batch_5d=a_batch_5d,
            device=device,
            nr_samples=MON_NR_SAMPLES,
            steps=MON_STEPS,
            eps=MON_EPS,
        )
        monotonicity_scores[name] = float(np.mean(scores))
        print(f"[monotonicity] {name}: mean Spearman = {monotonicity_scores[name]:.4f}")

# Append monotonicity row
mono_row = {}
for method_name, score in monotonicity_scores.items():
    col = str(method_name).lower()
    mono_row[col] = score

if len(mono_row) > 0:
    pivot_df.loc["monotonicity-corr"] = mono_row
else:
    print("[monotonicity] No methods produced valid monotonicity scores.")

print("\n=== Quantus Complexity & Faithfulness (Monotonicity) Metrics "
      "(averaged across all complexity samples / positives for monotonicity) ===")
try:
    print(pivot_df.round(6))
except Exception:
    print(pivot_df)

OUT_DIR = Path("experiments") / dataset_name / model_name / "eval_nem_vs_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUT_DIR / "quantus_complexity_metrics_raw.csv", index=False)
pd.DataFrame(pivot_df).to_csv(OUT_DIR / "quantus_complexity_metrics.csv")
print(f"[output] Saved raw metrics to {OUT_DIR / 'quantus_complexity_metrics_raw.csv'}")
print(f"[output] Saved pivoted metrics (incl. monotonicity) to {OUT_DIR / 'quantus_complexity_metrics.csv'}")

labels_series = pd.Series(y_batch, name="label")
for lab in sorted(labels_series.unique()):
    mask  = (labels_series.values == lab)
    x_sub = x_batch_2d[mask]
    y_sub = y_batch[mask]
    if x_sub.shape[0] == 0:
        continue
    xai_methods_sub = {name: arr[mask] for name, arr in xai_methods_complexity.items()}

    res_sub = quantus.evaluate(
        metrics=metrics_complexity,
        xai_methods=xai_methods_sub,
        model=None,
        x_batch=x_sub,
        y_batch=y_sub,
        agg_func=np.mean,
        return_as_df=True,
        verbose=False,
    )

    d = res_sub.copy()
    d.columns = [str(c).lower() for c in d.columns]
    if all(c in d.columns for c in ["metric", "xai_method", "score"]):
        sub_pivot = (
            d.groupby(["metric", "xai_method"])["score"]
            .mean().reset_index()
            .pivot(index="metric", columns="xai_method", values="score")
        )
    else:
        sub_pivot = d

    out_path = OUT_DIR / f"quantus_complexity_metrics_label{lab}.csv"
    pd.DataFrame(sub_pivot).to_csv(out_path)
    print(f"[output] Saved per-class metrics (label={lab}) to {out_path}")

print("\n=== Mask Generation Timing ===")
for method, times in timings.items():
    if len(times) == 0:
        print(f"{method}: No timings recorded (method may have been skipped).")
        continue
    total_time = sum(times)
    avg_time   = total_time / len(times)
    print(f"{method}: total = {total_time:.2f} s, avg per sample = {avg_time:.2f} s")
