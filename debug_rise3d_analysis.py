import os, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp_config import CHOSEN_DATASETS, CHOSEN_MODELS
from attrs.rise3d import rs3d_atr
from attrs.intgrad3d import itg3d_atr


def spearman_1d(x: np.ndarray, y: np.ndarray) -> float:
    assert x.shape == y.shape
    n = x.size
    if n < 2:
        return 0.0
    order_x = np.argsort(x)
    rank_x = np.empty_like(order_x, dtype=np.float64)
    rank_x[order_x] = np.arange(n, dtype=np.float64)

    order_y = np.argsort(y)
    rank_y = np.empty_like(order_y, dtype=np.float64)
    rank_y[order_y] = np.arange(n, dtype=np.float64)

    rx = rank_x - rank_x.mean()
    ry = rank_y - rank_y.mean()

    num = float(np.sum(rx * ry))
    den = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    if den <= 0.0:
        return 0.0
    return float(num / den)


def get_scalar_logit(model: torch.nn.Module, x_5d: torch.Tensor) -> float:
    """
    Get a single scalar logit for a 5D input [1,1,D,H,W] (binary or multi-class).
    - Binary: returns the single logit.
    - Multi-class: returns the logit of the argmax class.
    """
    with torch.no_grad():
        logits = model(x_5d)
        if logits.ndim == 1:
            y_pred_t = logits.view(-1)
            scalar = y_pred_t[0]
        elif logits.ndim == 2:
            B, C = logits.shape
            if C == 1:
                scalar = logits.view(-1)[0]
            else:
                idx = torch.argmax(logits, dim=1)
                scalar = logits[torch.arange(B, device=logits.device), idx][0]
        else:
            # Fallback: flatten and take first element
            scalar = logits.view(-1)[0]
    return float(scalar.item())


def single_monotonicity_curve(
    model: torch.nn.Module,
    x_5d_np: np.ndarray,      # (1,1,D,H,W)
    a_3d_np: np.ndarray,      # (D,H,W)
    device: torch.device,
    nr_samples: int = 5,
    steps: int = 32,
    y_orig: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a monotonicity curve for a single sample:

      - step_atts: sum of attributions in each perturbation step
      - step_vars: scaled variance in prediction in that step
      - rho: Spearman(step_atts, step_vars)
    """
    x_5d = x_5d_np.astype(np.float32)
    a_3d = a_3d_np.astype(np.float32)

    # Work on the single image volume (C=1 assumed)
    x_vol = x_5d[0, 0]  # (D,H,W)
    assert x_vol.shape == a_3d.shape, f"shape mismatch x_vol={x_vol.shape}, a_3d={a_3d.shape}"

    x_flat = x_vol.reshape(-1)  # (P,)
    a_flat = a_3d.reshape(-1)   # (P,)
    P = x_flat.shape[0]

    steps = max(1, int(steps))
    features_in_step = max(1, P // steps)
    n_steps = math.ceil(P / features_in_step)

    # Original prediction
    if y_orig is None:
        x_t = torch.from_numpy(x_5d).to(device=device, dtype=torch.float32)
        y_orig = get_scalar_logit(model, x_t)
    y0 = float(y_orig)

    # same scaling as in monotonicity_corr_gpu: 1 / |y0|^2
    eps = 1e-5
    inv_pred = 1.0 / (max(eps, abs(y0)) ** 2)

    x_min = float(x_flat.min())
    x_max = float(x_flat.max())

    order = np.argsort(a_flat)  # ascending attributions

    step_atts = []
    step_vars = []

    for step_idx in range(n_steps):
        start = step_idx * features_in_step
        end = min((step_idx + 1) * features_in_step, P)
        if start >= end:
            break

        feat_idx = order[start:end]  # indices of features perturbed in this step

        # Sum of attributions in this chunk
        step_atts.append(float(a_flat[feat_idx].sum()))

        # Collect perturbed predictions
        y_pert_list = []
        for _ in range(nr_samples):
            x_flat_pert = x_flat.copy()
            baseline_vals = np.random.uniform(x_min, x_max, size=feat_idx.shape[0]).astype(np.float32)
            x_flat_pert[feat_idx] = baseline_vals
            vol_pert = x_flat_pert.reshape(x_vol.shape)           # (D,H,W)
            x_pert_5d = vol_pert[None, None, ...].astype(np.float32)  # (1,1,D,H,W)

            x_t = torch.from_numpy(x_pert_5d).to(device=device, dtype=torch.float32)
            y_p = get_scalar_logit(model, x_t)
            y_pert_list.append(y_p)

        y_pert = np.array(y_pert_list, dtype=np.float32)  # (nr_samples,)
        var = float(np.mean((y_pert - y0) ** 2) * inv_pred)
        step_vars.append(var)

    step_atts = np.array(step_atts, dtype=np.float32)
    step_vars = np.array(step_vars, dtype=np.float32)
    rho = spearman_1d(step_atts, step_vars)
    return step_atts, step_vars, rho


def occlusion_top_bottom(
    model: torch.nn.Module,
    x_5d_np: np.ndarray,     # (1,1,D,H,W)
    a_3d_np: np.ndarray,     # (D,H,W)
    device: torch.device,
    ratio: float = 0.1,
    nr_samples: int = 5,
    y_orig: float | None = None,
) -> dict:
    """
    Simple sanity check:
      - occlude bottom `ratio` of features (lowest attribution) and measure mean logit drop.
      - occlude top `ratio` of features (highest attribution) and measure mean logit drop.

    For a good deletion-based explainer we expect:
      drop_top  >> drop_bottom
    """
    x_5d = x_5d_np.astype(np.float32)
    a_3d = a_3d_np.astype(np.float32)

    x_vol = x_5d[0, 0]
    assert x_vol.shape == a_3d.shape
    x_flat = x_vol.reshape(-1)
    a_flat = a_3d.reshape(-1)
    P = x_flat.shape[0]

    k = max(1, int(ratio * P))
    order = np.argsort(a_flat)
    low_idx = order[:k]
    high_idx = order[-k:]

    if y_orig is None:
        x_t_full = torch.from_numpy(x_5d).to(device=device, dtype=torch.float32)
        y0 = get_scalar_logit(model, x_t_full)
    else:
        y0 = float(y_orig)

    x_min = float(x_flat.min())
    x_max = float(x_flat.max())

    def _drop_for_indices(feat_idx: np.ndarray) -> tuple[float, float]:
        y_pert_list = []
        for _ in range(nr_samples):
            x_flat_pert = x_flat.copy()
            baseline_vals = np.random.uniform(x_min, x_max, size=feat_idx.shape[0]).astype(np.float32)
            x_flat_pert[feat_idx] = baseline_vals
            vol_pert = x_flat_pert.reshape(x_vol.shape)
            x_pert_5d = vol_pert[None, None, ...].astype(np.float32)

            x_t = torch.from_numpy(x_pert_5d).to(device=device, dtype=torch.float32)
            y_p = get_scalar_logit(model, x_t)
            y_pert_list.append(y_p)
        y_pert = np.array(y_pert_list, dtype=np.float32)
        mean_drop = float(y0 - y_pert.mean())
        var = float(np.mean((y_pert - y0) ** 2))
        return mean_drop, var

    drop_low, var_low = _drop_for_indices(low_idx)
    drop_high, var_high = _drop_for_indices(high_idx)

    return {
        "ratio": float(ratio),
        "k": int(k),
        "y_orig": float(y0),
        "drop_bottom": drop_low,
        "var_bottom": var_low,
        "drop_top": drop_high,
        "var_top": var_high,
    }


def random_occlusion_control(
    model: torch.nn.Module,
    x_5d_np: np.ndarray,     # (1,1,D,H,W)
    device: torch.device,
    ratio: float = 0.1,
    nr_samples: int = 5,
    y_orig: float | None = None,
) -> dict:
    """
    Baseline sanity check: occlude a random subset of features of the same size
    as the "top" set, and measure mean logit drop + variance.
    If RISE is informative, we expect:
        drop_top(RISE)  >>  drop_random  ~ drop_bottom(RISE)
    """
    x_5d = x_5d_np.astype(np.float32)
    x_vol = x_5d[0, 0]
    x_flat = x_vol.reshape(-1)
    P = x_flat.shape[0]

    k = max(1, int(ratio * P))
    all_idx = np.arange(P, dtype=int)

    if y_orig is None:
        x_t_full = torch.from_numpy(x_5d).to(device=device, dtype=torch.float32)
        y0 = get_scalar_logit(model, x_t_full)
    else:
        y0 = float(y_orig)

    x_min = float(x_flat.min())
    x_max = float(x_flat.max())

    y_pert_list = []
    for _ in range(nr_samples):
        rand_idx = np.random.choice(all_idx, size=k, replace=False)
        x_flat_pert = x_flat.copy()
        baseline_vals = np.random.uniform(x_min, x_max, size=rand_idx.shape[0]).astype(np.float32)
        x_flat_pert[rand_idx] = baseline_vals
        vol_pert = x_flat_pert.reshape(x_vol.shape)
        x_pert_5d = vol_pert[None, None, ...].astype(np.float32)

        x_t = torch.from_numpy(x_pert_5d).to(device=device, dtype=torch.float32)
        y_p = get_scalar_logit(model, x_t)
        y_pert_list.append(y_p)

    y_pert = np.array(y_pert_list, dtype=np.float32)
    mean_drop = float(y0 - y_pert.mean())
    var = float(np.mean((y_pert - y0) ** 2))

    return {
        "ratio": float(ratio),
        "k": int(k),
        "y_orig": float(y0),
        "drop_random": mean_drop,
        "var_random": var,
    }


def main():
    dataset_name = os.environ.get("NEM_DATASETS", "luna3d").split(",")[0]
    model_name = os.environ.get("NEM_MODELS", "phase3_3d").split(",")[0]

    POS_LABEL = int(os.environ.get("MON_POS_LABEL", "1"))
    N_POS_DEBUG = int(os.environ.get("DEBUG_RISE_N_POS", "8"))
    MAX_VAL_SCAN = int(os.environ.get("DEBUG_RISE_MAX_VAL_SCAN", "400"))
    MIN_PROB_DEBUG = float(os.environ.get("DEBUG_RISE_MIN_PROB", "0.4"))

    MON_STEPS = int(os.environ.get("DEBUG_RISE_STEPS", "32"))
    MON_NR_SAMPLES = int(os.environ.get("DEBUG_RISE_NR_SAMPLES", "5"))

    print(f"[info][debug_rise] dataset='{dataset_name}', model='{model_name}'")
    print(f"[info][debug_rise] N_POS_DEBUG={N_POS_DEBUG}, MAX_VAL_SCAN={MAX_VAL_SCAN}, "
          f"MON_STEPS={MON_STEPS}, MON_NR_SAMPLES={MON_NR_SAMPLES}")

    data_obj = CHOSEN_DATASETS[dataset_name]()
    train_loader, val_loader = data_obj.get_data()
    if val_loader is None:
        raise RuntimeError("Validation data loader not found.")

    model = CHOSEN_MODELS[model_name]().eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[info][debug_rise] model device: {device}")

    RISE_explainer = rs3d_atr(model, train_data=None, use_predicted_labels=True)
    rise_n_masks_env = os.environ.get("RISE_N_MASKS", "")
    if rise_n_masks_env not in ("", "None"):
        try:
            RISE_explainer.n_masks = int(rise_n_masks_env)
            print(f"[info][debug_rise] RISE n_masks overridden to {RISE_explainer.n_masks}")
        except ValueError:
            print(f"[warn][debug_rise] invalid RISE_N_MASKS='{rise_n_masks_env}', using default {RISE_explainer.n_masks}")

    IG_explainer = itg3d_atr(model, train_data=None, use_predicted_labels=True)
    ig_n_steps_env = os.environ.get("IG_N_STEPS", "")
    if ig_n_steps_env not in ("", "None"):
        try:
            IG_explainer.n_steps = int(ig_n_steps_env)
            print(f"[info][debug_rise] IG n_steps overridden to {IG_explainer.n_steps}")
        except ValueError:
            print(f"[warn][debug_rise] invalid IG_N_STEPS='{ig_n_steps_env}', using default {getattr(IG_explainer, 'n_steps', 'unknown')}")

    val_dataset = val_loader.dataset
    if hasattr(val_dataset, "indices"):
        full_dataset = val_dataset.dataset
        val_indices = list(val_dataset.indices)
        print(f"[info][debug_rise] val subset size={len(val_indices)}, full dataset size={len(full_dataset)}")
    else:
        full_dataset = val_dataset
        val_indices = list(range(len(full_dataset)))
        print(f"[info][debug_rise] val dataset size={len(full_dataset)} (no subset wrapper).")

    if not hasattr(full_dataset, "rows"):
        raise RuntimeError("Full dataset has no 'rows' attribute – this debug script expects LunaCandidates3DDataset.")

    pos_samples = []
    for k, gi in enumerate(val_indices):
        if k >= MAX_VAL_SCAN:
            break
        row = full_dataset.rows[gi]
        lbl = int(row.get("label", 0))
        if lbl != POS_LABEL:
            continue

        X_patch, y = full_dataset[gi]  # X_patch: [1,D,H,W]
        label = int(y.item()) if torch.is_tensor(y) else int(y)
        X_sample = X_patch.unsqueeze(0).to(device)  # [1,1,D,H,W]

        with torch.no_grad():
            logits = model(X_sample)
            if logits.ndim == 2 and logits.shape[1] == 1:
                z = float(logits.view(-1)[0].item())
                prob_pos = 1.0 / (1.0 + math.exp(-z))
            elif logits.ndim == 1:
                z = float(logits.view(-1)[0].item())
                prob_pos = 1.0 / (1.0 + math.exp(-z))
            else:
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1)[0].item())
                z = float(logits[0, pred_idx].item())
                prob_pos = float(probs[0, pred_idx].item())

        # only keep confidently positive ones
        if prob_pos < MIN_PROB_DEBUG:
            continue

        pos_samples.append(
            dict(
                full_idx=gi,
                uid=row.get("uid", f"idx{gi}"),
                label=label,
                X_sample=X_sample,
                logit=z,
                prob_pos=prob_pos,
            )
        )

        if len(pos_samples) >= N_POS_DEBUG:
            break

    if not pos_samples:
        raise RuntimeError("No positive validation samples collected for debug_rise.")

    print(f"[info][debug_rise] Collected {len(pos_samples)} positive validation samples for RISE debug.")

    dataset_out_dir = Path("experiments") / dataset_name / model_name / "debug_rise3d"
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info][debug_rise] Output directory: {dataset_out_dir}")

    summary_rows = []

    for idx, sample in enumerate(pos_samples):
        uid = sample["uid"]
        gi = sample["full_idx"]
        label = sample["label"]
        X_sample = sample["X_sample"]
        z = sample["logit"]
        prob_pos = sample["prob_pos"]

        print(f"\n[debug_rise][sample {idx}] uid={uid} | full_idx={gi} | label={label} | "
              f"logit={z:.4f} | prob_pos={prob_pos:.3f}")

        # Convert input to numpy 5D
        x_5d_np = X_sample.detach().cpu().numpy().astype(np.float32)  # (1,1,D,H,W)

        t0 = time.perf_counter()
        rise_attr = RISE_explainer.gen_attr(X_sample, None)  # [D,H,W] or [B,D,H,W]
        t1 = time.perf_counter()
        rise_np = np.asarray(rise_attr, dtype=np.float32)
        if rise_np.ndim == 4:
            rise_np = rise_np[0]
        if rise_np.ndim != 3:
            raise ValueError(f"Unexpected RISE shape {rise_np.shape} for sample {idx}")

        rise_min = float(rise_np.min())
        rise_max = float(rise_np.max())
        rise_mean = float(rise_np.mean())
        rise_std = float(rise_np.std())

        print(f"[debug_rise][sample {idx}] RISE attr shape={rise_np.shape}, "
              f"min={rise_min:.6g}, max={rise_max:.6g}, "
              f"mean={rise_mean:.6g}, std={rise_std:.6g}, "
              f"time={t1 - t0:.2f}s")

        thr_half = float(rise_max * 0.5)
        frac_high = float((rise_np >= thr_half).sum() / rise_np.size)
        print(f"[debug_rise][sample {idx}] frac(>0.5*max) = {frac_high * 100:.3f}%")

        q10, q50, q90 = np.quantile(rise_np, [0.1, 0.5, 0.9])
        print(f"[debug_rise][sample {idx}] RISE quantiles: q10={q10:.6g}, q50={q50:.6g}, q90={q90:.6g}")

        t0_ig = time.perf_counter()
        ig_attr = IG_explainer.gen_attr(X_sample, None)
        t1_ig = time.perf_counter()
        ig_np = np.asarray(ig_attr, dtype=np.float32)
        if ig_np.ndim == 4:
            ig_np = ig_np[0]
        if ig_np.ndim != 3:
            raise ValueError(f"Unexpected IG shape {ig_np.shape} for sample {idx}")

        ig_min = float(ig_np.min())
        ig_max = float(ig_np.max())
        ig_mean = float(ig_np.mean())
        ig_std = float(ig_np.std())

        print(f"[debug_rise][sample {idx}] IG attr shape={ig_np.shape}, "
              f"min={ig_min:.6g}, max={ig_max:.6g}, "
              f"mean={ig_mean:.6g}, std={ig_std:.6g}, "
              f"time={t1_ig - t0_ig:.2f}s")

        # Spearman correlation between RISE and IG attributions
        try:
            rise_ig_spearman = spearman_1d(rise_np.reshape(-1), ig_np.reshape(-1))
        except AssertionError:
            rise_ig_spearman = float("nan")
        print(f"[debug_rise][sample {idx}] Spearman(RISE, IG) = {rise_ig_spearman:.4f}")

        #Top vs bottom occlusion sanity check
        occ_stats = occlusion_top_bottom(
            model=model,
            x_5d_np=x_5d_np,
            a_3d_np=rise_np,
            device=device,
            ratio=0.10,
            nr_samples=MON_NR_SAMPLES,
            y_orig=z,
        )
        print(
            f"[debug_rise][sample {idx}] Occlusion ratio={occ_stats['ratio']:.2f}, k={occ_stats['k']} | "
            f"drop_top={occ_stats['drop_top']:.4f}, drop_bottom={occ_stats['drop_bottom']:.4f} | "
            f"var_top={occ_stats['var_top']:.4e}, var_bottom={occ_stats['var_bottom']:.4e}"
        )

        rand_stats = random_occlusion_control(
            model=model,
            x_5d_np=x_5d_np,
            device=device,
            ratio=occ_stats["ratio"],
            nr_samples=MON_NR_SAMPLES,
            y_orig=z,
        )
        print(
            f"[debug_rise][sample {idx}] Random occlusion: "
            f"drop_random={rand_stats['drop_random']:.4f} | "
            f"var_random={rand_stats['var_random']:.4e}"
        )

        step_atts, step_vars, rho_rise = single_monotonicity_curve(
            model=model,
            x_5d_np=x_5d_np,
            a_3d_np=rise_np,
            device=device,
            nr_samples=MON_NR_SAMPLES,
            steps=MON_STEPS,
            y_orig=z,
        )
        print(f"[debug_rise][sample {idx}] Monotonicity Spearman(RISE) = {rho_rise:.4f}")

        # For comparison: monotonicity curve for IG on the same sample
        step_atts_ig, step_vars_ig, rho_ig = single_monotonicity_curve(
            model=model,
            x_5d_np=x_5d_np,
            a_3d_np=ig_np,
            device=device,
            nr_samples=MON_NR_SAMPLES,
            steps=MON_STEPS,
            y_orig=z,
        )
        print(f"[debug_rise][sample {idx}] Monotonicity Spearman(IG) = {rho_ig:.4f}")

        # Save per-sample curves
        curve_df = pd.DataFrame(
            {
                "step": np.arange(len(step_atts), dtype=int),
                "rise_step_attribution_sum": step_atts,
                "rise_step_var": step_vars,
                "ig_step_attribution_sum": step_atts_ig,
                "ig_step_var": step_vars_ig,
            }
        )
        curve_path = dataset_out_dir / f"rise_monotonicity_curve_sample{idx}_uid_{uid}.csv"
        curve_df.to_csv(curve_path, index=False)
        print(f"[debug_rise][sample {idx}] Saved monotonicity curves to {curve_path}")

        # Add row to summary
        summary_rows.append(
            dict(
                sample_idx=idx,
                full_idx=gi,
                uid=uid,
                label=label,
                logit=z,
                prob_pos=prob_pos,
                rise_min=rise_min,
                rise_max=rise_max,
                rise_mean=rise_mean,
                rise_std=rise_std,
                rise_frac_half=float(frac_high),
                rise_q10=float(q10),
                rise_q50=float(q50),
                rise_q90=float(q90),
                ig_min=ig_min,
                ig_max=ig_max,
                ig_mean=ig_mean,
                ig_std=ig_std,
                rise_ig_spearman=float(rise_ig_spearman),
                rise_occ_ratio=float(occ_stats["ratio"]),
                rise_occ_k=int(occ_stats["k"]),
                rise_drop_top=float(occ_stats["drop_top"]),
                rise_drop_bottom=float(occ_stats["drop_bottom"]),
                rise_var_top=float(occ_stats["var_top"]),
                rise_var_bottom=float(occ_stats["var_bottom"]),
                rand_drop=float(rand_stats["drop_random"]),
                rand_var=float(rand_stats["var_random"]),
                rise_monotonicity_spearman=float(rho_rise),
                ig_monotonicity_spearman=float(rho_ig),
                rise_runtime_sec=float(t1 - t0),
                ig_runtime_sec=float(t1_ig - t0_ig),
            )
        )

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = dataset_out_dir / "rise_debug_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[debug_rise] Saved summary to {summary_path}")


if __name__ == "__main__":
    import sys, traceback

    print("[debug_rise] Starting debug_rise3d_analysis.py ...", flush=True)
    try:
        main()
        print("[debug_rise] Finished debug_rise3d_analysis.py successfully.", flush=True)
    except Exception as e:
        print(f"[debug_rise] FATAL ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
