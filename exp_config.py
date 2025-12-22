from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from attrs.nem_utils.method_nemt3d import NEMT3DMethod
from attrs.rise3d import rs3d_atr, rs3d_ranked_atr
from attrs.intgrad3d import itg3d_atr, itg3d_ranked_atr
from attrs.saliency3d import saliency3d_atr, saliency3d_forgrad_atr

from exp_utils.experiments_nem3d import NEM3DAccMaskExperiment

from exp_utils.luna_dataset import make_luna_loaders

EXP_STORAGE_PATH = os.environ.get("EXP_STORAGE_PATH", "experiments")

PATCH_SIZE_3D = (64, 96, 96)

class LUNA3DDataset:
    def __init__(
        self,
        luna_root: str = "Luna16",
        batch_size: int = 1,
        num_workers: int = 0,
        patch_size=PATCH_SIZE_3D,
        max_items=None,
        seed: int = 42,
    ):
        fold_id = int(os.environ.get("LUNA_FOLD_ID", "0"))
        self.train_loader, self.val_loader = make_luna_loaders(
            luna_root=luna_root,
            batch_size=batch_size,
            num_workers=num_workers,
            patch_size=patch_size,
            max_items=max_items,
            seed=seed,
            fold_id=fold_id,
        )

    def get_data(self):
        # Returns (train_loader, val_loader).
        return self.train_loader, self.val_loader


POSSIBLE_DATASETS: Dict[str, callable] = {
    "luna3d": lambda: LUNA3DDataset(
        luna_root=os.environ.get("LUNA_ROOT", "Luna16"),
        batch_size=int(os.environ.get("NEM_BATCH_SIZE", "1")),
        num_workers=int(os.environ.get("NEM_NUM_WORKERS", "4")),
        patch_size=PATCH_SIZE_3D,
        max_items=(
            None
            if os.environ.get("NEM_MAX_ITEMS") in (None, "", "None")
            else int(os.environ["NEM_MAX_ITEMS"])
        ),
        seed=int(os.environ.get("NEM_SEED", "42")),
    )
}

PHASE3_CKPT = "luna16_cls3d/luna16_cls3d/fold_0/model_phase3.pth"
CALIBRATION_JSON = "luna16_cls3d/luna16_cls3d/fold_0/calibration_phase3.json"


def _strip_prefixes(sd: dict, prefixes=("_orig_mod.", "module.")) -> dict:
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        new_sd[k] = v
    return new_sd


class UNetEncoderClassifier(nn.Module):
    def __init__(self, in_ch=1, widths=(32, 64, 128, 256), norm_layer=nn.BatchNorm3d):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                norm_layer(cout),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                norm_layer(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_ch, widths[0])
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = block(widths[0], widths[1])
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = block(widths[1], widths[2])
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = block(widths[2], widths[3])
        self.head = nn.Linear(widths[3], 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)
        x = self.enc4(x)
        x = F.adaptive_avg_pool3d(x, 1).flatten(1)
        return self.head(x)


def _to_cuda_safe(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    if not torch.cuda.is_available():
        return model
    use_cl3d = os.environ.get("USE_CHANNELS_LAST_3D", "1") == "1"
    mf = getattr(torch, "channels_last_3d", None)
    if use_cl3d and mf is not None:
        try:
            return model.to("cuda", memory_format=mf)
        except (RuntimeError, AttributeError) as e:
            print("[load_phase3_model] channels_last_3d failed, falling back:", e)
            return model.to("cuda")
    else:
        return model.to("cuda")


def _load_temperature(path: str, default=1.0) -> float:
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        t = float(obj.get("temperature", default))
        return max(t, 1e-6)
    except Exception:
        return float(default)


TEMPERATURE = _load_temperature(CALIBRATION_JSON, default=1.0)


def _detect_arch(sd: dict) -> str:
    has_dense = any(k.startswith("features.") for k in sd.keys())  # MONAI DenseNet
    has_unet = any(k.startswith("enc1.") for k in sd.keys())
    if has_dense and not has_unet:
        return "densenet121"
    if has_unet and not has_dense:
        return "unetenc"
    return "densenet121"  # fallback


def load_phase3_model():
    ckpt = Path(PHASE3_CKPT)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    obj = torch.load(ckpt, map_location="cpu")
    state_dict = obj["state_dict"] if (isinstance(obj, dict) and "state_dict" in obj) else obj

    if not isinstance(state_dict, dict):
        model = obj
        model = _to_cuda_safe(model)
        print(f"[load_phase3_model] Loaded model object from {ckpt}")
        return model

    state_dict = _strip_prefixes(state_dict)
    arch = _detect_arch(state_dict)

    if arch == "densenet121":
        from monai.networks.nets import DenseNet121

        model = DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.0
        )
        arch_name = "DenseNet121-3D (MONAI)"
    else:
        model = UNetEncoderClassifier(in_ch=1, widths=(32, 64, 128, 256))
        arch_name = "UNetEncoderClassifier-3D"

    try:
        result = model.load_state_dict(state_dict, strict=True)
        mk = getattr(result, "missing_keys", [])
        uk = getattr(result, "unexpected_keys", [])
        if mk or uk:
            print(
                f"[load_phase3_model] strict=True reported Missing: {mk} | Unexpected: {uk}"
            )
    except RuntimeError as e:
        print(f"[load_phase3_model] strict=True failed ({e}); retrying with strict=False.")
        result = model.load_state_dict(state_dict, strict=False)
        mk = getattr(result, "missing_keys", [])
        uk = getattr(result, "unexpected_keys", [])
        print(
            f"[load_phase3_model] Loaded with strict=False. Missing: {mk} | Unexpected: {uk}"
        )

    model = _to_cuda_safe(model)
    print(f"[load_phase3_model] Loaded {arch_name} from {ckpt}")
    return model


POSSIBLE_MODELS = {
    "phase3_3d": load_phase3_model
}

POSSIBLE_METHODS = {
    "nemt3d_train": lambda model, train_data, use_predicted_labels=True: NEMT3DMethod(
        model=model,
        train_loader=train_data,
        supervised=True,
        inverse=True,
        train_or_load=True,   # TRAIN in this run
    ),
    "nemt3d_load": lambda model, train_data, use_predicted_labels=True: NEMT3DMethod(
        model=model,
        train_loader=train_data,
        supervised=True,
        inverse=True,
        train_or_load=False,  # LOAD-ONLY for evaluation
    ),

    "rise3d": lambda model, train_data, use_predicted_labels=True: rs3d_atr(
        model=model, train_data=train_data, use_predicted_labels=use_predicted_labels
    ),
    "intgrad3d": lambda model, train_data, use_predicted_labels=True: itg3d_atr(
        model=model, train_data=train_data, use_predicted_labels=use_predicted_labels
    ),
    "saliency3d": lambda model, train_data, use_predicted_labels=True: saliency3d_atr(
        model=model, train_data=train_data, use_predicted_labels=use_predicted_labels
    ),

    # optional ranked variants
    "rise3d_ranked": lambda model, train_data, use_predicted_labels=True: rs3d_ranked_atr(
        model=model, train_data=train_data, use_predicted_labels=use_predicted_labels
    ),
    "intgrad3d_ranked": lambda model, train_data, use_predicted_labels=True: itg3d_ranked_atr(
        model=model, train_data=train_data, use_predicted_labels=use_predicted_labels
    ),
}


POSSIBLE_EXPERIMENTS = {
    "acc_maskratio": lambda method, model, train_loader, val_loader, exp_store: NEM3DAccMaskExperiment(
        method=method,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=exp_store,
        limit_val_batches=(
            None
            if os.environ.get("NEM_LIMIT_VAL") in (None, "", "None")
            else int(os.environ["NEM_LIMIT_VAL"])
        ),
        temperature=TEMPERATURE,
        decision_threshold=0.5,
    )
}

CHOSEN_DATASETS = POSSIBLE_DATASETS
CHOSEN_MODELS = POSSIBLE_MODELS
CHOSEN_METHODS = POSSIBLE_METHODS
CHOSEN_EXPERIMENTS = POSSIBLE_EXPERIMENTS
