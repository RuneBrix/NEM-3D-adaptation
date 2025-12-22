from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from .NEM import (
    densenet3d_nem,
    unetenc3d_nem,
    load_nemt_3d,
    is_3d_model,
)

def _pick_nem3d_ctor(model) -> type:
    name = model.__class__.__name__.lower()
    return densenet3d_nem if "densenet" in name else unetenc3d_nem

def _backbone_name(model) -> str:
    n = model.__class__.__name__
    if "DenseNet" in n or "densenet" in n:
        return "DenseNet121"
    if "UNet" in n or "Unet" in n or "Encoder" in n:
        return "UNetEnc"
    return n

def _resolve_out_dir(model) -> str:
    default_root = Path("attrs") / "nem_utils" / "logs" / "nemt3d" / _backbone_name(model)
    return os.environ.get("NEM_OUT_DIR", str(default_root))

def _best_ckpt_path(out_dir: str) -> Optional[str]:
    """
    Robustly find a checkpoint:
      1) last.ckpt directly in out_dir
      2) path from best_ckpt.txt
      3) newest *.ckpt inside out_dir/checkpoints
    """
    out = Path(out_dir)
    last = out / "last.ckpt"
    if last.is_file():
        return str(last)

    best_txt = out / "best_ckpt.txt"
    if best_txt.is_file():
        try:
            cand = best_txt.read_text().strip()
            if cand and Path(cand).is_file():
                return cand
        except Exception:
            pass

    ckdir = out / "checkpoints"
    if ckdir.is_dir():
        cands = sorted(ckdir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return str(cands[0])

    return None


class NEMT3DMethod:
    """
    Thin adapter so the pipeline can treat 3D-NEMT like any other method.

    train_or_load=True  -> training run (train if needed; or resume depending on env flags)
    train_or_load=False -> load-only (evaluation run); fail fast if no checkpoint found
    """
    def __init__(
        self,
        model,
        train_loader=None,
        supervised: bool = True,
        inverse: bool = True,
        epochs: int = 1,
        batch_size: int = 1,
        train_or_load: bool = True,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frozen_network = model.to(self.device).eval()

        if not is_3d_model(self.frozen_network):
            raise ValueError("NEMT3DMethod requires a 3D model (must contain Conv3d).")

        # Where NEM logs/checkpoints live
        self.out_dir = _resolve_out_dir(self.frozen_network)

        # read knobs
        use_ckpt    = os.environ.get("NEM_USE_CKPT", "1") == "1"
        force_train = os.environ.get("NEM_FORCE_TRAIN", "0") == "1"

        # allow env to override ctor defaults
        epochs = int(os.environ.get("NEM_EPOCHS", str(epochs)))
        batch_size = int(os.environ.get("NEM_TRAIN_BATCH_SIZE", str(batch_size)))

        if train_or_load:
            # TRAINING RUN
            if force_train:
                self.nem = self._train_new(train_loader, supervised, inverse, epochs, batch_size)
            else:
                if use_ckpt:
                    ck = _best_ckpt_path(self.out_dir)
                    if ck:
                        self.nem = self._load_from_ckpt(ck)
                    else:
                        # no ckpt -> train
                        self.nem = self._train_new(train_loader, supervised, inverse, epochs, batch_size)
                else:
                    self.nem = self._train_new(train_loader, supervised, inverse, epochs, batch_size)
        else:
            # EVALUATION RUN
            ck = _best_ckpt_path(self.out_dir)
            if not ck:
                raise FileNotFoundError(
                    f"[NEMT3DMethod] load-only requested but no checkpoint found in '{self.out_dir}'. "
                    f"Looked for last.ckpt / best_ckpt.txt / checkpoints/*.ckpt"
                )
            self.nem = self._load_from_ckpt(ck)

        self.nem = self.nem.to(self.device).eval()

    # API used by the evaluation/experiments

    @torch.no_grad()
    def gen_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (mask_logits, x_masked, applied_mask) for a batch x [B,1,D,H,W].
        """
        x = x.to(self.device, non_blocking=True)
        return self.nem.gen_mask(x)

    @torch.no_grad()
    def get_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the classifier logits for x.
        Prefer nem.get_output if present; otherwise try frozen_network.get_output;
        else just forward the classifier.
        """
        x = x.to(self.device, non_blocking=True)
        if hasattr(self.nem, "get_output") and callable(self.nem.get_output):
            return self.nem.get_output(x)
        fn = getattr(self.nem.frozen_network, "get_output", None)
        if callable(fn):
            return fn(x)
        return self.nem.frozen_network(x)

    # internals
    def _load_from_ckpt(self, ckpt_path: str):
        NemCtor = _pick_nem3d_ctor(self.frozen_network)

        nem = NemCtor.load_from_checkpoint(
            ckpt_path,
            explained_model=self.frozen_network,
            epochs=1,
            batch_size=1,
            supervised=True,
            inverse=True,
            strict=False,
        )
        print(f"[NEMT3DMethod] Loaded NEM checkpoint: {ckpt_path}")
        return nem

    def _train_new(self, train_loader, supervised, inverse, epochs, batch_size):
        """
        Train (or train+load) the 3D NEM using the unified load_nemt_3d(model, train_data)
        helper from NEM.py.

        - NEM_USE_CKPT / NEM_FORCE_TRAIN / NEM_EPOCHS / NEM_TBLOG etc.
          are handled inside load_nemt_3d / train_nem_3d.
        - Checkpoints are written under:
            attrs/nem_utils/logs/nemt3d/<ModelClassName>/
          which matches what _resolve_out_dir() returns for DenseNet.
        """
        if train_loader is None:
            raise RuntimeError("NEMT3DMethod: train_loader is required to train NEM.")

        nem = load_nemt_3d(self.frozen_network, train_loader)

        print(f"[NEMT3DMethod] NEM trained/loaded via load_nemt_3d().")
        return nem

