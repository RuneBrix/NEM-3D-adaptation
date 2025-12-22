# train_nem.py
from __future__ import annotations
import os, platform, traceback, time, copy
from pathlib import Path
import torch

from exp_config import (
    CHOSEN_DATASETS, CHOSEN_MODELS, POSSIBLE_METHODS, EXP_STORAGE_PATH
)

def _dump_debug():
    print("=== TRAIN NEM DEBUG ===")
    print("Host:", platform.node())
    print("CUDA available:", torch.cuda.is_available(),
          "| torch:", torch.__version__,
          "| CUDA build:", torch.version.cuda)
    print("EXP_STORAGE_PATH:", EXP_STORAGE_PATH)
    for p in [
        "Luna16",
        "luna16_cls3d/luna16_cls3d/fold_0/model_phase3.pth",
        "attrs/nem_utils/method_nemt3d.py",
    ]:
        print("path check:", p, "->", "OK" if Path(p).exists() else "MISSING")
    print("=======================")

def main():
    only_datasets  = set([s for s in os.environ.get("NEM_DATASETS",  "").split(",") if s])
    only_models    = set([s for s in os.environ.get("NEM_MODELS",    "").split(",") if s])

    _dump_debug()

    for dname, dctor in CHOSEN_DATASETS.items():
        if only_datasets and dname not in only_datasets:
            continue

        data = dctor()
        train_loader, _ = data.get_data()

        for mname, mctor in CHOSEN_MODELS.items():
            if only_models and mname not in only_models:
                continue

            print(f"[build] classifier: {mname}")
            model = mctor().eval()
            if torch.cuda.is_available():
                model = model.cuda()

            # Freeze classifier weights
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()

            # Save base weights (optional safety)
            base_state = copy.deepcopy(model.state_dict())

            # Build + TRAIN NEM
            try:
                t0 = time.perf_counter()
                nem_ctor = POSSIBLE_METHODS["nemt3d_train"]
                method = nem_ctor(model, train_loader, use_predicted_labels=True)

                print("[debug] method.nem class:", type(method.nem))
                has_conv3d = any(isinstance(m, torch.nn.Conv3d) for m in method.nem.modules())
                print("[debug] NEM has Conv3d:", has_conv3d)
                assert has_conv3d, "You're not running the 3D NEM path!"

                dt = time.perf_counter() - t0
                print(f"[train] NEMT3D finished. Elapsed: {dt:.1f}s")
            except Exception as e:
                print("ERROR: NEM training failed:", repr(e))
                traceback.print_exc()
                continue

            # Post-train quick sanity: mask stats
            try:
                batch = next(iter(train_loader))
                x_dbg, y_dbg = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)

                device = next(model.parameters()).device
                x_dbg = x_dbg.to(device)

                nem = method.nem.to(device).eval()
                with torch.no_grad():
                    mlogits, x_masked_dbg, amask_dbg = nem.gen_mask(x_dbg)
                    print("[post-train debug] x shape:", tuple(x_dbg.shape))
                    print("[post-train debug] mask shape:", tuple(amask_dbg.shape))
                    print("[post-train debug] mask stats:",
                          "min", float(amask_dbg.min()),
                          "max", float(amask_dbg.max()),
                          "mean", float(amask_dbg.mean()),
                          "std", float(amask_dbg.std()))
            except Exception as e:
                print("[post-train debug] failed:", e)

            # restore classifier to base weights (optional)
            try:
                model.load_state_dict(base_state)
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
            except Exception:
                pass

            print("[done] train_nem.py end.")

if __name__ == "__main__":
    main()
