from __future__ import annotations
import os
from pathlib import Path

import torch

from exp_config import CHOSEN_DATASETS, CHOSEN_MODELS
from attrs.nem_utils.method_nemt3d import NEMT3DMethod


def main():
    dataset_name = os.environ.get("NEM_DATASETS", "luna3d").split(",")[0]
    model_name   = os.environ.get("NEM_MODELS", "phase3_3d").split(",")[0]

    print(f"[info] Debugging NEM on dataset='{dataset_name}', model='{model_name}'")

    data_obj = CHOSEN_DATASETS[dataset_name]()
    train_loader, val_loader = data_obj.get_data()

    if val_loader is None:
        raise RuntimeError("[error] val_loader is None; cannot debug on validation set.")

    val_dataset = val_loader.dataset
    if hasattr(val_dataset, "indices"):
        full_dataset = val_dataset.dataset
        val_indices = list(val_dataset.indices)
    else:
        full_dataset = val_dataset
        val_indices = list(range(len(full_dataset)))

    print(f"[info] Full dataset size={len(full_dataset)}, val subset size={len(val_indices)}")

    if not hasattr(full_dataset, "rows"):
        raise RuntimeError("[error] Expected LunaCandidates3DDataset with 'rows' attribute.")

    POS_LABEL = int(os.environ.get("VIZ_POS_LABEL", "1"))

    pos_indices = [
        gi for gi in val_indices
        if int(full_dataset.rows[gi].get("label", 0)) == POS_LABEL
    ]
    print(f"[info] Found {len(pos_indices)} positive validation candidates (label={POS_LABEL}).")

    if not pos_indices:
        print("[error] No positives found; abort.")
        return

    # Take up to 5 positives for debugging
    pos_indices = pos_indices[:5]
    print("[info] Using positive indices:", pos_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    model = CHOSEN_MODELS[model_name]().eval().to(device)
    nem = NEMT3DMethod(
        model=model,
        train_loader=None,
        supervised=True,
        inverse=True,
        train_or_load=False,
        device=str(device),
    )
    print(f"[info] NEM loaded from: {nem.out_dir}")

    masks = []

    with torch.no_grad():
        for k, gi in enumerate(pos_indices):
            x, y = full_dataset[gi]          # x: [1,D,H,W], y: tensor(label)
            if isinstance(y, torch.Tensor):
                y_val = int(y.item())
            else:
                y_val = int(y)

            # shape -> [B,1,D,H,W]
            if x.dim() == 4:
                x = x.unsqueeze(0)
            elif x.dim() == 5:
                pass  # already batched
            else:
                raise RuntimeError(f"[error] Unexpected x.dim()={x.dim()} for idx={gi}")

            x = x.to(device)

            logits_orig = nem.get_output(x)   # uses classifier behind NEM
            p_orig = torch.sigmoid(logits_orig.view(-1))[0].item()

            mlogits, x_masked, mask = nem.gen_mask(x)
            logits_masked = nem.get_output(x_masked)
            p_mask = torch.sigmoid(logits_masked.view(-1))[0].item()

            keep_ratio = mask.mean().item()
            drop = p_orig - p_mask

            print(
                f"[sample {k}] idx={gi} label={y_val} | "
                f"p_orig={p_orig:.3f}, p_mask={p_mask:.3f}, drop={drop:.3f}, keep={keep_ratio:.3f}"
            )

            masks.append(mask.cpu())

    if len(masks) > 1:
        print("\n[info] Pairwise max abs diff between masks:")
        for i in range(len(masks) - 1):
            m1 = masks[i].float()
            m2 = masks[i + 1].float()
            diff = torch.abs(m1 - m2).max().item()
            mean1 = float(m1.mean())
            mean2 = float(m2.mean())
            print(
                f"  pair {i}-{i+1}: max|diff|={diff:.6f}, "
                f"mean1={mean1:.4f}, mean2={mean2:.4f}"
            )
    else:
        print("[info] Only one mask collected; no pairwise comparison possible.")


if __name__ == "__main__":
    main()
