import math
import numpy as np
import torch


def _spearmanr_1d(x: np.ndarray, y: np.ndarray) -> float:
    assert x.shape == y.shape
    P = x.size

    ox = np.argsort(x)
    rx = np.empty(P, dtype=np.float64)
    rx[ox] = np.arange(P, dtype=np.float64)

    oy = np.argsort(y)
    ry = np.empty(P, dtype=np.float64)
    ry[oy] = np.arange(P, dtype=np.float64)

    rx = rx - rx.mean()
    ry = ry - ry.mean()
    num = np.sum(rx * ry)
    den = math.sqrt(np.sum(rx**2) * np.sum(ry**2))
    return float(num / den) if den > 0 else 0.0


def monotonicity_corr_debug(
    model: torch.nn.Module,
    x_5d: np.ndarray,          # (1,1,D,H,W)
    y_np: np.ndarray,          # (1,) (for multiclass); ignored for binary logit models
    a_5d: np.ndarray,          # (1,1,D,H,W)
    device: torch.device,
    nr_samples: int = 3,
    steps: int = 32,
    eps: float = 1e-5,
    order: str = "asc",        # "asc" = low->high, "desc" = high->low
    baseline: str = "uniform", # "uniform", "mean", "zero"
    return_indices: bool = False,
):
    """
    Returns:
      corr, atts[steps], vars_[steps], (optional) indices per step
    """
    model.eval()

    x = x_5d.astype(np.float32)
    a = a_5d.astype(np.float32)

    N = 1
    x_flat = x.reshape(N, -1)
    a_flat = a.reshape(N, -1)
    n_features = a_flat.shape[1]

    steps = max(1, int(steps))
    features_in_step = max(1, n_features // steps)
    n_steps = math.ceil(n_features / features_in_step)

    # base prediction
    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        logits = model(x_t)
        y_pred_t = logits.view(-1)  # binary/logit case
        y_pred = y_pred_t.detach().cpu().numpy().astype(np.float32)  # (1,)

    inv_pred = np.ones((1,), dtype=np.float32)
    mask = np.abs(y_pred) >= eps
    inv_pred[mask] = 1.0 / np.abs(y_pred[mask])
    inv_pred = inv_pred ** 2

    # sort indices by attribution
    if order == "asc":
        a_indices = np.argsort(a_flat, axis=1)          # low -> high
    elif order == "desc":
        a_indices = np.argsort(-a_flat, axis=1)         # high -> low
    else:
        raise ValueError("order must be 'asc' or 'desc'")

    # baseline sources
    x_min = x_flat.min(axis=1, keepdims=True)
    x_max = x_flat.max(axis=1, keepdims=True)
    x_mean = x_flat.mean(axis=1, keepdims=True)

    atts_list = []
    vars_list = []
    idx_list = []

    for step_idx in range(n_steps):
        start = step_idx * features_in_step
        end = min((step_idx + 1) * features_in_step, n_features)
        if start >= end:
            break

        a_ix = a_indices[:, start:end]  # (1, K)
        step_atts = np.take_along_axis(a_flat, a_ix, axis=1).sum(axis=1)  # (1,)
        atts_list.append(step_atts[0])

        if return_indices:
            idx_list.append(a_ix[0].copy())

        y_pred_perturbs = []
        for _ in range(nr_samples):
            if baseline == "uniform":
                base_vals = np.random.uniform(x_min, x_max, size=a_ix.shape).astype(np.float32)
            elif baseline == "mean":
                base_vals = np.broadcast_to(x_mean, a_ix.shape).astype(np.float32)
            elif baseline == "zero":
                base_vals = np.zeros(a_ix.shape, dtype=np.float32)
            else:
                raise ValueError("baseline must be 'uniform', 'mean', or 'zero'")

            x_pert = x_flat.copy()
            x_pert[0, a_ix[0]] = base_vals[0]
            x_pert = x_pert.reshape(x.shape)

            with torch.no_grad():
                x_p_t = torch.from_numpy(x_pert).to(device=device, dtype=torch.float32)
                logits_p = model(x_p_t).view(-1)
                y_p = logits_p.detach().cpu().numpy().astype(np.float32)
            y_pred_perturbs.append(y_p[0])

        y_pred_perturbs = np.array(y_pred_perturbs, dtype=np.float32)  # (nr_samples,)
        step_var = np.mean((y_pred_perturbs - y_pred[0]) ** 2) * inv_pred[0]
        vars_list.append(float(step_var))

    atts = np.array(atts_list, dtype=np.float32)
    vars_ = np.array(vars_list, dtype=np.float32)
    corr = _spearmanr_1d(atts, vars_)

    if return_indices:
        return corr, atts, vars_, idx_list
    return corr, atts, vars_


def attr_summary(a_5d: np.ndarray, name: str):
    a = a_5d.astype(np.float32).reshape(-1)
    q = np.quantile(a, [0.0, 0.5, 0.9, 0.99, 0.999, 1.0])
    print(f"[{name}] min/med/p90/p99/p99.9/max = " +
          "/".join(f"{v:.6g}" for v in q))
    # “spikiness”: fraction of mass in top 1%
    a_pos = np.maximum(a, 0)
    total = a_pos.sum() + 1e-12
    thr = np.quantile(a_pos, 0.99)
    top_mass = a_pos[a_pos >= thr].sum() / total
    print(f"[{name}] top-1% mass share = {top_mass:.4f}")


def shuffle_attr(a_5d: np.ndarray, seed=0):
    rng = np.random.default_rng(seed)
    flat = a_5d.reshape(-1).copy()
    rng.shuffle(flat)
    return flat.reshape(a_5d.shape)


def constant_attr_like(a_5d: np.ndarray, value=1.0):
    return np.full_like(a_5d, float(value), dtype=np.float32)
