import os
import csv
from pathlib import Path

import torch
from tqdm import tqdm


class NEM3DAccMaskExperiment:
    """
    3D NEMT evaluation using:
      - Deletion AUC
      - Insertion AUC
      - Randomized-mask baselines
      - Center-based pointing metric for positive patches

    Assumptions:
      - method.gen_mask(x) -> (mask_logits, x_masked, amask)
        where amask ≈ [B, 1, D, H, W], higher = more important / kept.
      - method.get_output(x) -> logits [B, 1] for binary classification.
      - val_loader yields (x, y) with y in {0,1} (LUNA candidate labels).
      - Each patch is extracted around a candidate location
        (so patch center ≈ candidate; good for a coarse pointing game).
    """

    def __init__(
        self,
        method,
        model,
        train_loader,
        val_loader,
        out_dir,
        limit_val_batches=None,
        temperature=1.0,
        decision_threshold=0.5,  # kept for compatibility; not central here
        debug_first_k=0,
    ):
        self.method = method
        self.model = model
        self.val_loader = val_loader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.limit_val_batches = limit_val_batches
        self.temperature = float(temperature)
        self.decision_threshold = float(decision_threshold)
        self.debug_first_k = int(debug_first_k)

        # Number of segments on deletion/insertion curves (N+1 points)
        self.eval_steps = max(4, int(os.environ.get("NEM_EVAL_STEPS", "16")))
        # Radius in voxels for center-based pointing hit
        self.point_radius = max(1, int(os.environ.get("NEM_POINT_RADIUS", "4")))

        self._run()

    @staticmethod
    def _auc01(values):
        """
        Trapezoidal AUC over x in [0,1] for a uniform grid of y-values.
        len(values) >= 2, points at x = 0, 1/(n-1), ..., 1.
        """
        n = len(values)
        if n < 2:
            return float("nan")
        step = 1.0 / (n - 1)
        s = 0.0
        prev = float(values[0])
        for i in range(1, n):
            cur = float(values[i])
            s += 0.5 * (prev + cur) * step
            prev = cur
        return s

    @staticmethod
    def _ensure_xy(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return batch[0], batch[1]
            elif len(batch) == 1:
                return batch[0], None
        return batch, None

    @torch.no_grad()
    def _run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        T = max(self.temperature, 1e-6)
        steps = self.eval_steps

        # bookkeeping
        n_samples = 0
        mask_means = []
        collapse_zero = 0
        collapse_one = 0

        deletion_aucs = []
        insertion_aucs = []
        deletion_aucs_rand = []
        insertion_aucs_rand = []

        pos_samples = 0
        pos_point_hits = 0

        rows = []

        # progress bar length
        try:
            total_batches = len(self.val_loader)
        except TypeError:
            total_batches = None
        if self.limit_val_batches is not None and total_batches is not None:
            total_batches = min(total_batches, self.limit_val_batches)

        pbar = tqdm(
            self.val_loader,
            total=total_batches,
            desc="Eval(3D-NEMT: del/ins/point)",
            dynamic_ncols=True,
        )

        debug_printed = 0

        for b_idx, batch in enumerate(pbar, 1):
            if self.limit_val_batches is not None and b_idx > self.limit_val_batches:
                break

            x, y = self._ensure_xy(batch)
            x = x.to(device, non_blocking=True)

            if y is not None:
                y = y.to(device, non_blocking=True).view(-1).long()
            else:
                # unlabeled fallback
                y = torch.full((x.size(0),), -1, device=device, dtype=torch.long)

            bsz = x.size(0)

            #explainer mask
            mlogits, x_masked, amask = self.method.gen_mask(x)  # expect [B,1,D,H,W]
            if amask.dim() == 4:  # [B,D,H,W] -> [B,1,D,H,W]
                amask = amask.unsqueeze(1)

            #basic mask stats
            mr = amask.flatten(1).mean(dim=1)  # [B]
            mask_means.extend(mr.detach().cpu().tolist())
            collapse_zero += int((mr < 0.05).sum().item())
            collapse_one += int((mr > 0.95).sum().item())

            # per-sample evaluations
            for i in range(bsz):
                x_i = x[i : i + 1]      # [1,1,D,H,W]
                y_i = int(y[i].item())
                m_i = amask[i, 0]       # [D,H,W]

                if not torch.isfinite(m_i).all():
                    continue

                flat = m_i.flatten()
                N = flat.numel()
                if N == 0:
                    continue

                # importance order: high -> low
                order = torch.argsort(flat, descending=True)
                # random baseline: same values, shuffled
                order_rand = torch.randperm(N, device=flat.device)

                baseline = torch.zeros_like(x_i)

                #helpers to build curves
                def build_curve_del(order_idx):
                    probs = []
                    x_del = x_i.clone()
                    for s in range(steps + 1):
                        if s > 0:
                            k_prev = int((s - 1) / steps * N)
                            k_curr = int(s / steps * N)
                            if k_curr > k_prev:
                                idx = order_idx[k_prev:k_curr]
                                x_del.view(-1)[idx] = 0.0
                        logit = self.method.get_output(x_del).view(-1)[0]
                        p_pos = torch.sigmoid(logit / T).item()
                        if y_i in (0, 1):
                            p_true = p_pos if y_i == 1 else (1.0 - p_pos)
                        else:
                            p_true = p_pos
                        probs.append(p_true)
                    return probs

                def build_curve_ins(order_idx):
                    probs = []
                    x_ins = baseline.clone()
                    for s in range(steps + 1):
                        if s > 0:
                            k_prev = int((s - 1) / steps * N)
                            k_curr = int(s / steps * N)
                            if k_curr > k_prev:
                                idx = order_idx[k_prev:k_curr]
                                x_ins.view(-1)[idx] = x_i.view(-1)[idx]
                        logit = self.method.get_output(x_ins).view(-1)[0]
                        p_pos = torch.sigmoid(logit / T).item()
                        if y_i in (0, 1):
                            p_true = p_pos if y_i == 1 else (1.0 - p_pos)
                        else:
                            p_true = p_pos
                        probs.append(p_true)
                    return probs

                #explainer deletion/insertion
                probs_del = build_curve_del(order)
                probs_ins = build_curve_ins(order)
                auc_del = self._auc01(probs_del)
                auc_ins = self._auc01(probs_ins)
                deletion_aucs.append(auc_del)
                insertion_aucs.append(auc_ins)

                #random deletion/insertion
                probs_del_r = build_curve_del(order_rand)
                probs_ins_r = build_curve_ins(order_rand)
                auc_del_r = self._auc01(probs_del_r)
                auc_ins_r = self._auc01(probs_ins_r)
                deletion_aucs_rand.append(auc_del_r)
                insertion_aucs_rand.append(auc_ins_r)

                # pointing metric (positive patches only)
                if y_i == 1:
                    pos_samples += 1

                    # patch center (approx candidate location)
                    D, H, W = m_i.shape
                    cz, cy, cx = D // 2, H // 2, W // 2

                    # index of max mask value
                    max_idx = int(torch.argmax(flat).item())
                    z = max_idx // (H * W)
                    yv = (max_idx % (H * W)) // W
                    xv = max_idx % W

                    dz = z - cz
                    dy = yv - cy
                    dx = xv - cx
                    dist = (dz * dz + dy * dy + dx * dx) ** 0.5

                    if dist <= self.point_radius:
                        pos_point_hits += 1

                # optional debug
                if self.debug_first_k > 0 and debug_printed < self.debug_first_k:
                    print(
                        f"[dbg] b={b_idx} i={i} y={y_i} "
                        f"del_auc={auc_del:.4f} (rand {auc_del_r:.4f}) "
                        f"ins_auc={auc_ins:.4f} (rand {auc_ins_r:.4f}) "
                        f"mask_mean={float(m_i.mean().item()):.4f}"
                    )
                    debug_printed += 1

            n_samples += bsz

            # running means for progress bar
            mean_mask = (sum(mask_means) / max(1, len(mask_means))) if mask_means else 0.0
            mean_del = (sum(deletion_aucs) / max(1, len(deletion_aucs))) if deletion_aucs else 0.0
            mean_ins = (sum(insertion_aucs) / max(1, len(insertion_aucs))) if insertion_aucs else 0.0
            mean_del_r = (sum(deletion_aucs_rand) / max(1, len(deletion_aucs_rand))) if deletion_aucs_rand else 0.0
            mean_ins_r = (sum(insertion_aucs_rand) / max(1, len(insertion_aucs_rand))) if insertion_aucs_rand else 0.0
            point_acc = (pos_point_hits / max(1, pos_samples)) if pos_samples > 0 else 0.0

            pbar.set_postfix({
                "maskμ": f"{mean_mask:.3f}",
                "delAUC": f"{mean_del:.3f}",
                "insAUC": f"{mean_ins:.3f}",
                "delAUC_r": f"{mean_del_r:.3f}",
                "insAUC_r": f"{mean_ins_r:.3f}",
                "pt@center": f"{point_acc:.3f}",
                "~0": collapse_zero,
                "~1": collapse_one,
            })

            rows.append({
                "batch": b_idx,
                "mask_mean_batch": float(mr.mean().item()),
                "mask_min_batch": float(mr.min().item()),
                "mask_max_batch": float(mr.max().item()),
                "collapse_zero_cum": collapse_zero,
                "collapse_one_cum": collapse_one,
                "del_auc_mean_so_far": mean_del,
                "ins_auc_mean_so_far": mean_ins,
                "del_auc_rand_mean_so_far": mean_del_r,
                "ins_auc_rand_mean_so_far": mean_ins_r,
                "pointing_pos_acc_so_far": point_acc,
            })

        # final aggregates
        mean_mask = (sum(mask_means) / max(1, len(mask_means))) if mask_means else 0.0
        mean_del = (sum(deletion_aucs) / max(1, len(deletion_aucs))) if deletion_aucs else 0.0
        mean_ins = (sum(insertion_aucs) / max(1, len(insertion_aucs))) if insertion_aucs else 0.0
        mean_del_r = (sum(deletion_aucs_rand) / max(1, len(deletion_aucs_rand))) if deletion_aucs_rand else 0.0
        mean_ins_r = (sum(insertion_aucs_rand) / max(1, len(insertion_aucs_rand))) if insertion_aucs_rand else 0.0
        point_acc = (pos_point_hits / max(1, pos_samples)) if pos_samples > 0 else 0.0

        summary_txt = (
            f"Temperature:               {T}\n"
            f"Samples:                   {n_samples}\n"
            f"Mask ratio mean:           {mean_mask:.6f}\n"
            f"Collapse ~0:               {collapse_zero} / {n_samples}\n"
            f"Collapse ~1:               {collapse_one} / {n_samples}\n"
            f"Deletion AUC (mean):       {mean_del:.6f}   (lower is better)\n"
            f"Deletion AUC rand (mean):  {mean_del_r:.6f}\n"
            f"Insertion AUC (mean):      {mean_ins:.6f}   (higher is better)\n"
            f"Insertion AUC rand (mean): {mean_ins_r:.6f}\n"
            f"Pointing pos@center(r={self.point_radius}): {point_acc:.6f} "
            f"({pos_point_hits} / {max(1, pos_samples)} pos samples)\n"
            f"Eval steps:                {steps}\n"
        )

        print("\n=== Summary (3D-NEMT deletion/insertion + baselines + pointing) ===\n" + summary_txt)

        # write CSV + summary
        csv_path = self.out_dir / "eval_nem3d.csv"
        if rows:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        with open(self.out_dir / "summary.txt", "w") as f:
            f.write(summary_txt)
