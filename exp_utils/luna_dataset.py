import csv
import glob
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

from exp_utils.luna_monai3d import build_val_tf


def _clip_normalize_hu(arr, hu_min=-1000, hu_max=400):
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)  # [0,1]
    arr = (arr * 2.0) - 1.0                   # [-1,1]
    return arr.astype(np.float32)


def _safe_crop(center_idx, shape, crop_size):
    zc, yc, xc = center_idx
    D, H, W = shape
    dz, dy, dx = crop_size
    z0 = max(0, zc - dz // 2); z1 = min(D, z0 + dz); z0 = max(0, z1 - dz)
    y0 = max(0, yc - dy // 2); y1 = min(H, y0 + dy); y0 = max(0, y1 - dy)
    x0 = max(0, xc - dx // 2); x1 = min(W, x0 + dx); x0 = max(0, x1 - dx)
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


class LunaCandidates3DDataset(Dataset):
    """
    Loads 3D CT patches around LUNA16 candidate locations (from candidates_V2.csv).
    Each item: (tensor [1,D,H,W], int label)
    """

    def __init__(
        self,
        luna_root: str = "Luna16",
        candidates_csv: str = "candidates_V2.csv",
        patch_size: Tuple[int, int, int] = (64, 96, 96),
        max_items: Optional[int] = None,
        seed: int = 1337,
        use_monai_pipeline: bool = True,
        transform=None,
        fold_id: int = 0,
        split: str = "val",  # "train" or "val"
        label_filter: Optional[int] = None,  # None / 0 / 1
    ):
        super().__init__()
        self.root = Path(luna_root)
        self.patch = patch_size
        self.use_monai = use_monai_pipeline
        self.transform = transform
        self.fold_id = int(fold_id)
        self.split = str(split).lower()
        self.label_filter = label_filter if label_filter in (0, 1) else None
        rng = random.Random(seed)

        # Find all .mhd files recursively
        mhd_paths = glob.glob(str(self.root / "subset*" / "**" / "*.mhd"), recursive=True)
        self.uid_to_path: Dict[str, str] = {Path(p).stem: p for p in mhd_paths}
        if not self.uid_to_path:
            raise FileNotFoundError(
                "No .mhd files found under Luna16/subset*/**/*.mhd. "
                "Check that zips were extracted and .mhd/.raw exist."
            )

        # Subset id (fold split) per UID, derived from filepath
        def _subset_from_path(p: str) -> int:
            for part in Path(p).parts:
                if part.startswith("subset"):
                    try:
                        return int(part.replace("subset", ""))
                    except Exception:
                        pass
            return -1

        self.uid_to_subset: Dict[str, int] = {
            uid: _subset_from_path(path) for uid, path in self.uid_to_path.items()
        }

        #Lung mask path per UID
        self.uid_to_lung: Dict[str, Optional[str]] = {
            uid: self._find_lung_mask(uid) for uid in self.uid_to_path.keys()
        }

        #Find candidates_V2.csv
        matches_v2 = glob.glob(str(self.root / "**" / "candidates_V2.csv"), recursive=True)
        matches_v1 = glob.glob(str(self.root / "**" / "candidates.csv"), recursive=True)

        if len(matches_v2) > 0:
            csv_path = Path(matches_v2[0])
        elif len(matches_v1) > 0:
            csv_path = Path(matches_v1[0])
        else:
            candidate_try = self.root / candidates_csv
            if candidate_try.exists():
                csv_path = candidate_try
            else:
                raise FileNotFoundError(
                    "Missing candidates file. Expected candidates_V2.csv or candidates.csv somewhere under Luna16/."
                )

        #Read candidates
        rows: List[Dict[str, Any]] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            col = {name: i for i, name in enumerate(header)}

            def _get(name, default_idx):
                return col.get(name, default_idx)

            for r in reader:
                uid = r[_get("seriesuid", 0)]
                if uid not in self.uid_to_path:
                    continue
                try:
                    rows.append(
                        {
                            "uid": uid,
                            "subset": int(self.uid_to_subset.get(uid, -1)),
                            "x": float(r[_get("coordX", 1)]),
                            "y": float(r[_get("coordY", 2)]),
                            "z": float(r[_get("coordZ", 3)]),
                            "label": int(float(r[_get("class", 4)])),
                        }
                    )
                except Exception:
                    continue

        if not rows:
            raise RuntimeError("No candidate rows matched local scans.")

        # Fold split
        if self.split == "val":
            rows = [r for r in rows if r["subset"] == self.fold_id]
        elif self.split == "train":
            rows = [r for r in rows if r["subset"] != self.fold_id]
        else:
            raise ValueError(f"Unknown split='{self.split}'. Use 'train' or 'val'.")

        if not rows:
            raise RuntimeError(
                f"No rows left after split filter. split={self.split}, fold_id={self.fold_id}. "
                "Check that subset folders exist and subset parsing works."
            )

        if self.label_filter in (0, 1):
            rows = [r for r in rows if int(r["label"]) == int(self.label_filter)]

        if not rows:
            raise RuntimeError(
                f"No rows left after label_filter={self.label_filter}. "
                f"(split={self.split}, fold_id={self.fold_id})"
            )

        # Optional cap after filters
        if max_items is not None and int(max_items) > 0:
            rng.shuffle(rows)
            rows = rows[: int(max_items)]

        self.rows = rows

    def _find_lung_mask(self, uid: str) -> Optional[str]:
        patterns = [
            str(self.root / "**" / "seg-lungs-LUNA16" / "**" / f"{uid}.*"),
            str(self.root / "**" / "lung" / "**" / f"{uid}.*"),
            str(self.root / "**" / "masks" / "**" / f"{uid}.*"),
            str(self.root / "**" / f"{uid}_lungmask.*"),
        ]
        exts_ok = (".nii.gz", ".nii", ".mhd")
        for pat in patterns:
            for p in glob.glob(pat, recursive=True):
                if p.endswith(exts_ok):
                    return p
        return None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        if self.use_monai:
            sample = {
                "image": self.uid_to_path[r["uid"]],
                "center_world": np.array([r["x"], r["y"], r["z"]], dtype=np.float32),
            }
            if self.transform is not None:
                sample = self.transform(sample)

            tensor = sample["image"]  # torch.Tensor [1,D,H,W]
            label = torch.tensor(r["label"], dtype=torch.long)
            return tensor, label
        
        img = sitk.ReadImage(self.uid_to_path[r["uid"]])
        vol = sitk.GetArrayFromImage(img)  # (D,H,W)
        vol = _clip_normalize_hu(vol)

        ijk = img.TransformPhysicalPointToIndex((r["x"], r["y"], r["z"]))
        xc, yc, zc = ijk
        center_idx = (zc, yc, xc)

        sz = self.patch
        zsl, ysl, xsl = _safe_crop(center_idx, vol.shape, sz)
        patch = vol[zsl, ysl, xsl]

        need = (sz[0] - patch.shape[0], sz[1] - patch.shape[1], sz[2] - patch.shape[2])
        if any(n > 0 for n in need):
            pad = [(0, max(n, 0)) for n in need]
            patch = np.pad(patch, pad, mode="edge")

        tensor = torch.from_numpy(patch)[None, ...]
        label = torch.tensor(r["label"], dtype=torch.long)
        return tensor, label


def make_luna_loaders(
    luna_root: str = "Luna16",
    batch_size: int = 1,
    num_workers: int = 2,
    patch_size: Tuple[int, int, int] = (64, 96, 96),
    max_items: Optional[int] = None,
    seed: int = 42,
    fold_id: int = 0,
):
    val_tf = build_val_tf(patch_size=patch_size)

    train_set = LunaCandidates3DDataset(
        luna_root=luna_root,
        patch_size=patch_size,
        max_items=max_items,
        seed=seed,
        use_monai_pipeline=True,
        transform=val_tf,
        fold_id=fold_id,
        split="train",
        label_filter=1,   # <-- POS ONLY
    )

    # validation keeps both classes
    val_set = LunaCandidates3DDataset(
        luna_root=luna_root,
        patch_size=patch_size,
        max_items=max_items,
        seed=seed,
        use_monai_pipeline=True,
        transform=val_tf,
        fold_id=fold_id,
        split="val",
        label_filter=None,
    )

    train = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train, val
