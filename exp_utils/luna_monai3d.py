import numpy as np
import SimpleITK as sitk
from collections import OrderedDict

from monai.transforms import (
    Compose,
    ResizeWithPadOrCropd,
    EnsureTyped,
)
from monai.transforms.transform import MapTransform

SPACING_PATCH = (1.0, 1.0, 1.0)

class _LRU:
    def __init__(self, max_items=2):
        self.max = int(max_items)
        self.d = OrderedDict()

    def get(self, k):
        v = self.d.pop(k, None)
        if v is not None:
            self.d[k] = v
        return v

    def put(self, k, v):
        if k in self.d:
            self.d.pop(k)
        self.d[k] = v
        while len(self.d) > self.max:
            self.d.popitem(last=False)


def _make_ref(img, new_spacing):
    old_size = np.array(list(img.GetSize()), dtype=np.int64)
    old_sp = np.array(list(img.GetSpacing()), dtype=np.float64)
    new_sp = np.array(list(new_spacing), dtype=np.float64)
    new_size = np.maximum(1, np.round(old_size * (old_sp / new_sp)).astype(np.int64))
    ref = sitk.Image(
        int(new_size[0]),
        int(new_size[1]),
        int(new_size[2]),
        img.GetPixelID(),
    )
    ref.SetOrigin(img.GetOrigin())
    ref.SetDirection(img.GetDirection())
    ref.SetSpacing(tuple(new_spacing))
    return ref


class CachedLoadPreprocessd(MapTransform):
    """
    Load CT (and optional lung mask) from disk, resample to given spacing,
    clip HU window, and cache per-volume to avoid repeated IO.

    Expected keys:
      - "image": path to CT .mhd
      - optional "lung": path to lung mask (same study)
    """

    def __init__(
        self,
        keys=("image", "lung"),
        spacing=(1.0, 1.0, 1.0),
        a_min=-1000,
        a_max=400,
        max_cache=2,
        allow_missing_keys=True,
    ):
        super().__init__(keys, allow_missing_keys)
        self.spacing = tuple(float(s) for s in spacing)
        self.a_min, self.a_max = float(a_min), float(a_max)
        self.cache = _LRU(max_items=max_cache)

    def _to_lps(self, img):
        try:
            return sitk.DICOMOrient(img, "LPS")
        except Exception:
            return img

    def _resample(self, img, ref, is_mask):
        interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        return sitk.Resample(
            img,
            referenceImage=ref,
            transform=sitk.Transform(),
            interpolator=interp,
            defaultPixelValue=0,
            outputPixelType=img.GetPixelID(),
        )

    def __call__(self, data):
        d = dict(data)
        img_path = d.get("image")
        if img_path is None:
            return d

        key = (str(img_path), self.spacing)
        cached = self.cache.get(key)
        if cached is None:
            # --- load CT ---
            img = sitk.ReadImage(str(img_path))
            img = self._to_lps(img)
            ref = _make_ref(img, self.spacing)

            img1 = self._resample(img, ref, is_mask=False)
            arr = sitk.GetArrayFromImage(img1).astype(np.float32)  # (Z,Y,X)
            arr = np.clip(arr, self.a_min, self.a_max)[None, ...]  # (1,Z,Y,X)

            lung_arr = None
            if "lung" in d and d["lung"]:
                try:
                    lung = sitk.ReadImage(str(d["lung"]))
                    lung = self._to_lps(lung)
                    lung1 = self._resample(lung, ref, is_mask=True)
                    lung_arr = sitk.GetArrayFromImage(lung1).astype(np.int16)[None, ...]
                except Exception:
                    lung_arr = None

            meta = {
                "origin":    np.array(img1.GetOrigin(),    dtype=np.float32),
                "spacing":   np.array(img1.GetSpacing(),   dtype=np.float32),
                "direction": np.array(img1.GetDirection(), dtype=np.float32),
            }
            cached = (arr, lung_arr, meta)
            self.cache.put(key, cached)

        arr, lung_arr, meta = cached
        d["image"] = arr.copy()
        if lung_arr is not None:
            d["lung"] = lung_arr.copy()
        d["image_meta_dict"] = meta
        return d


class CropAroundWorldCentroidd(MapTransform):
    """
    Crop a fixed-size ROI (roi_size) around a world-space center (x,y,z)
    given under `center_key`. Uses image_meta_dict (origin, spacing, direction)
    to convert world coords to index coords.
    """

    def __init__(
        self,
        keys,
        roi_size=(64, 96, 96),
        center_key="center_world",
        jitter_mm=(0, 0, 0),
        allow_missing_keys=True,
    ):
        super().__init__(keys, allow_missing_keys)
        self.roi = np.array(roi_size, dtype=int)  # (D,H,W) == (Z,Y,X)
        self.center_key = center_key
        self.jitter = np.array(jitter_mm, dtype=float)

    def __call__(self, data):
        d = dict(data)
        if self.center_key not in d:
            return d

        cw = np.array(d[self.center_key], dtype=float)  # (x,y,z)
        if np.any(self.jitter != 0):
            cw = cw + np.random.uniform(-1, 1, size=3) * self.jitter

        meta = d.get("image_meta_dict", {})
        origin = np.array(meta.get("origin", (0, 0, 0)), dtype=float)
        spacing = np.array(meta.get("spacing", (1, 1, 1)), dtype=float)
        direction = np.array(
            meta.get("direction", np.eye(3).reshape(-1)),
            dtype=float,
        ).reshape(3, 3)

        # world -> index via affine
        A = np.eye(4, dtype=float)
        A[:3, :3] = direction @ np.diag(spacing)
        A[:3, 3] = origin
        invA = np.linalg.inv(A)

        cidx = invA @ np.array([cw[0], cw[1], cw[2], 1.0], dtype=float)
        cidx = cidx[:3]  # (x,y,z) index
        spatial = np.array(d["image"].shape[-3:], dtype=int)  # (Z,Y,X)

        start = np.floor(cidx[[2, 1, 0]] - self.roi / 2).astype(int)
        end = start + self.roi

        pad_l = np.maximum(0, -start)
        pad_r = np.maximum(0, end - spatial)

        s = np.maximum(start, 0)
        e = np.minimum(end, spatial)
        slc = (slice(s[0], e[0]), slice(s[1], e[1]), slice(s[2], e[2]))

        def _crop_pad(arr, key: str):
            crop = arr[(...,) + slc]

            if (pad_l > 0).any() or (pad_r > 0).any():
                padder = (
                    (0, 0),
                    (pad_l[0], pad_r[0]),
                    (pad_l[1], pad_r[1]),
                    (pad_l[2], pad_r[2]),
                )

                if key == "image":
                    # avoid creating fake 0-HU slabs at borders
                    crop = np.pad(crop, padder, mode="edge")
                else:
                    # masks (lung) should stay zero outside
                    crop = np.pad(crop, padder, mode="constant", constant_values=0)

            return crop

        for k in self.keys:
            if k in d:
                d[k] = _crop_pad(d[k], k)
        return d


class ZNormWithinLungd:
    """
    Z-normalise image using lung mask if available; otherwise global.
    """

    def __init__(self, key="image"):
        self.key = key

    def __call__(self, data):
        x = data[self.key]
        if "lung" in data:
            try:
                mask = data["lung"] > 0
                if mask.sum() > 0:
                    m = x[mask].mean()
                    s = x[mask].std()
                    if s > 0:
                        data[self.key] = (x - m) / s
                        return data
            except Exception:
                pass
        m, s = x.mean(), x.std()
        if s > 0:
            data[self.key] = (x - m) / s
        return data


def build_val_tf(patch_size=(64, 96, 96)):
    """
    Validation/eval transform:

      - load & resample to 1mm (CT, and optional lung mask)
      - crop around center_world
      - pad/crop to patch_size
      - z-norm (within lung if available)
      - cast to torch.Tensor

    Works both when 'lung' is present AND when it's missing
    """
    return Compose(
        [
            CachedLoadPreprocessd(
                keys=("image", "lung"),
                spacing=SPACING_PATCH,
                a_min=-1000,
                a_max=400,
                max_cache=2,
                allow_missing_keys=True,
            ),
            CropAroundWorldCentroidd(
                keys=["image", "lung"],
                roi_size=patch_size,
                center_key="center_world",
                jitter_mm=(0.0, 0.0, 0.0),
                allow_missing_keys=True,
            ),
            ResizeWithPadOrCropd(
                keys=["image", "lung"],
                spatial_size=patch_size,
                allow_missing_keys=True,
            ),
            ZNormWithinLungd("image"),
            EnsureTyped(
                keys=["image", "lung"],
                track_meta=False,
                allow_missing_keys=True,
            ),
        ]
    )
