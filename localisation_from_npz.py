import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _to_3d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 5:
        a = a[0, 0]
    elif a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expected 3D after squeeze, got {a.shape}")
    return a.astype(np.float32)


def _load_meta_json(npz_obj) -> dict:
    meta_raw = npz_obj["meta_json"]
    if isinstance(meta_raw, np.ndarray):
        s = meta_raw[0]
        if hasattr(s, "item"):
            s = s.item()
        return json.loads(s)
    if hasattr(meta_raw, "item"):
        return json.loads(meta_raw.item())
    return json.loads(str(meta_raw))


def _norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(a.min()), float(a.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn + 1e-9)


def _prep_attr(method: str, vol3d: np.ndarray) -> np.ndarray:
    """
    Keep it simple and robust:
      - IG: abs then norm01
      - NEM: raw then norm01
      - RISE: raw then norm01
    """
    v = np.asarray(vol3d, np.float32)
    m = method.lower()
    if m == "ig":
        v = np.abs(v)
    # for nem/rise keep raw
    return np.clip(_norm01(v), 0.0, 1.0)


def _relevancy_mass(attr01: np.ndarray, region: np.ndarray, eps: float = 1e-9) -> float:
    a = np.clip(np.asarray(attr01, np.float32), 0.0, 1.0)
    denom = float(a.sum()) + eps
    if denom <= eps:
        return 0.0
    return float(a[region].sum() / denom)


def _topk_mask(attr01: np.ndarray, topk_frac: float) -> np.ndarray:
    """
    Returns boolean mask for the top-k fraction of voxels by attribution value.

    Uses argpartition so it's stable even if distributions are weird / many ties.
    Guarantees at least 1 voxel.
    """
    a = np.asarray(attr01, np.float32).reshape(-1)
    n = a.size
    if n == 0:
        return np.zeros((0,), dtype=bool)

    f = float(topk_frac)
    f = max(0.0, min(1.0, f))
    if f <= 0.0:
        k = 1
    else:
        k = int(round(f * n))
        k = max(1, min(n, k))

    # indices of k largest values
    idx = np.argpartition(a, n - k)[n - k:]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def _relevancy_mass_topk(attr01: np.ndarray, region: np.ndarray, topk_frac: float, eps: float = 1e-9) -> float:
    """
    "Top-k relevancy mass":
      mass inside region, but ONLY considering the top-k% voxels (and normalized by mass in top-k).
    This reduces the bias where broad/smooth maps get punished vs spiky maps.
    """
    a = np.clip(np.asarray(attr01, np.float32), 0.0, 1.0)
    topk = _topk_mask(a, topk_frac).reshape(a.shape)

    denom = float(a[topk].sum()) + eps
    if denom <= eps:
        return 0.0
    return float(a[topk & region].sum() / denom)


def _pointing_game_hit(attr01: np.ndarray, region: np.ndarray) -> int:
    a = np.asarray(attr01, np.float32)
    idx = np.unravel_index(int(np.argmax(a)), a.shape)
    return int(bool(region[idx]))


def _closest_annotation(ann_df: pd.DataFrame, uid: str, cand_xyz: np.ndarray):
    rows = ann_df[ann_df["seriesuid"] == uid]
    if len(rows) == 0:
        return None
    coords = rows[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float32)
    dists = np.linalg.norm(coords - cand_xyz[None, :], axis=1)
    j = int(np.argmin(dists))
    best = rows.iloc[j].to_dict()
    best["dist_mm"] = float(dists[j])
    return best


def _sphere_mask_zyx(shape_zyx, center_zyx, radius_vox: float) -> np.ndarray:
    D, H, W = shape_zyx
    cz, cy, cx = center_zyx
    r2 = float(radius_vox) ** 2
    zz, yy, xx = np.ogrid[:D, :H, :W]
    dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return dist2 <= r2


def _world_xyz_to_patch_zyx(patch_shape_zyx, cand_xyz_world, ann_xyz_world, spacing_mm: float = 1.0):
    """
    Proxy mapping for appendix:
    - assume patch centered at candidate world coord
    - patch center index is (D-1)/2, (H-1)/2, (W-1)/2
    - offset world->vox uses spacing_mm
    """
    D, H, W = patch_shape_zyx
    patch_center_zyx = np.array([(D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0], dtype=np.float32)

    offset_xyz = (ann_xyz_world - cand_xyz_world).astype(np.float32)
    offset_zyx = np.array([offset_xyz[2], offset_xyz[1], offset_xyz[0]], dtype=np.float32) / float(spacing_mm)

    ann_center_zyx = patch_center_zyx + offset_zyx
    return ann_center_zyx


def _circle_for_slice(ax, center_yx, radius_px, color="cyan", lw=1.5):
    if radius_px <= 0:
        return
    cx, cy = center_yx
    circ = Circle((cx, cy), radius_px, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(circ)


def _draw_overlay(ax, base2d, heat2d, title: str, alpha: float):
    ax.imshow(base2d, cmap="gray", origin="lower")
    ax.imshow(heat2d, cmap="Reds", origin="lower", alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")


def _npz_get_attr(d, keys):
    for k in keys:
        if k in d.files:
            return d[k]
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-root", type=str, required=True, help="Root folder containing .npz bundles (will recurse).")
    ap.add_argument("--annotations-csv", type=str, required=True, help="Path to Luna16/annotations.csv")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for PNGs + CSVs.")
    ap.add_argument("--spacing-mm", type=float, default=1.0, help="Patch spacing in mm (default 1.0).")
    ap.add_argument("--max-match-dist-mm", type=float, default=10.0, help="Max distance (mm) for matching candidate->annotation.")
    ap.add_argument("--only-label", type=int, default=1, help="Only evaluate label=1 by default (positives). Use -1 for all.")
    ap.add_argument("--dedupe-gi", type=int, default=1, help="If 1, dedupe repeated gi across folders.")
    ap.add_argument("--topk-frac", type=float, default=0.01, help="Top-k fraction for top-k relevancy mass (e.g. 0.01 = 1%).")
    ap.add_argument("--viz-max", type=int, default=24, help="Max number of PNGs to save.")
    ap.add_argument("--viz-alpha", type=float, default=0.45, help="Overlay alpha.")
    ap.add_argument(
        "--viz-sort-by",
        type=str,
        default="rise_topk_relevancy_mass",
        help="Which column to sort by for picking best/worst viz cases.",
    )
    args = ap.parse_args()

    npz_root = Path(args.npz_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_df = pd.read_csv(args.annotations_csv)
    expected = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
    missing = expected - set(ann_df.columns)
    if missing:
        raise RuntimeError(f"annotations.csv missing columns: {missing}. Found: {list(ann_df.columns)}")

    npz_files = sorted(npz_root.rglob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found under: {npz_root}")

    rows = []
    seen_gi = set()

    for p in npz_files:
        d = np.load(str(p), allow_pickle=True)
        if "meta_json" not in d.files or "x_5d" not in d.files:
            continue

        meta = _load_meta_json(d)
        gi = meta.get("gi", None)
        label = meta.get("label", None)

        if args.dedupe_gi and gi is not None:
            if gi in seen_gi:
                continue
            seen_gi.add(gi)

        if args.only_label != -1 and label is not None and int(label) != int(args.only_label):
            continue

        x3d = _to_3d(d["x_5d"])

        # IG/NEM/RISE might not exist in every folder. Make it robust:
        ig_raw = _npz_get_attr(d, ["ig_attr_5d", "ig_5d", "attr_ig_5d"])
        nem_raw = _npz_get_attr(d, ["nem_attr_5d", "nem_5d", "attr_nem_5d"])
        rise_raw = _npz_get_attr(d, ["rise_attr_5d", "rise_5d", "attr_rise_5d"])

        # If any are missing, keep NaNs for that method (but still allow other methods)
        ig01 = _prep_attr("ig", _to_3d(ig_raw)) if ig_raw is not None else None
        nem01 = _prep_attr("nem", _to_3d(nem_raw)) if nem_raw is not None else None
        rise01 = _prep_attr("rise", _to_3d(rise_raw)) if rise_raw is not None else None

        uid = meta.get("uid", meta.get("seriesuid", meta.get("series_uid", None)))
        if uid is None:
            continue

        if not all(k in meta for k in ("x", "y", "z")):
            continue
        cand_xyz = np.array([float(meta["x"]), float(meta["y"]), float(meta["z"])], dtype=np.float32)

        ann = _closest_annotation(ann_df, str(uid), cand_xyz)
        if ann is None:
            continue
        if ann["dist_mm"] > float(args.max_match_dist_mm):
            continue

        ann_xyz = np.array([float(ann["coordX"]), float(ann["coordY"]), float(ann["coordZ"])], dtype=np.float32)
        diam_mm = float(ann["diameter_mm"])
        r_mm = diam_mm / 2.0
        r_vox = r_mm / float(args.spacing_mm)

        ann_center_zyx = _world_xyz_to_patch_zyx(x3d.shape, cand_xyz, ann_xyz, spacing_mm=args.spacing_mm)
        cz, cy, cx = ann_center_zyx
        cz = float(np.clip(cz, 0.0, x3d.shape[0] - 1))
        cy = float(np.clip(cy, 0.0, x3d.shape[1] - 1))
        cx = float(np.clip(cx, 0.0, x3d.shape[2] - 1))
        ann_center_zyx = (cz, cy, cx)

        region = _sphere_mask_zyx(x3d.shape, ann_center_zyx, r_vox)

        def _metrics_for(a01):
            if a01 is None:
                return (np.nan, np.nan, np.nan, np.nan)
            mass = _relevancy_mass(a01, region)
            mass_topk = _relevancy_mass_topk(a01, region, topk_frac=float(args.topk_frac))
            hit = _pointing_game_hit(a01, region)
            return (mass, mass_topk, hit)

        ig_mass, ig_mass_topk, ig_hit = _metrics_for(ig01)
        nem_mass, nem_mass_topk, nem_hit = _metrics_for(nem01)
        rise_mass, rise_mass_topk, rise_hit = _metrics_for(rise01)

        rows.append({
            "npz_path": str(p),
            "gi": gi,
            "label": label,
            "uid": str(uid),
            "cand_x": cand_xyz[0], "cand_y": cand_xyz[1], "cand_z": cand_xyz[2],
            "ann_x": ann_xyz[0], "ann_y": ann_xyz[1], "ann_z": ann_xyz[2],
            "ann_diameter_mm": diam_mm,
            "ann_dist_mm": float(ann["dist_mm"]),
            "topk_frac": float(args.topk_frac),

            "ig_relevancy_mass": ig_mass,
            "ig_topk_relevancy_mass": ig_mass_topk,
            "ig_pointing_hit": ig_hit,

            "nem_relevancy_mass": nem_mass,
            "nem_topk_relevancy_mass": nem_mass_topk,
            "nem_pointing_hit": nem_hit,

            "rise_relevancy_mass": rise_mass,
            "rise_topk_relevancy_mass": rise_mass_topk,
            "rise_pointing_hit": rise_hit,

            "ig_monotonicity_corr": meta.get("ig_monotonicity_corr", None),
            "nem_monotonicity_corr": meta.get("nem_monotonicity_corr", None),
            "rise_monotonicity_corr": meta.get("rise_monotonicity_corr", None),
            "delta_monotonicity_corr": meta.get("delta_monotonicity_corr", None),
        })

    if not rows:
        raise RuntimeError("No matched samples found. Try increasing --max-match-dist-mm or set --only-label=-1.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "localisation_per_case.csv", index=False)

    def _mean_std(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        s = s[np.isfinite(s)]
        if len(s) == 0:
            return (np.nan, np.nan)
        return (float(s.mean()), float(s.std(ddof=0)))

    def _hit_rate(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        s = s[np.isfinite(s)]
        if len(s) == 0:
            return np.nan
        return float(s.mean())

    # aggregate summary
    ig_m, ig_s = _mean_std(df["ig_relevancy_mass"])
    igk_m, igk_s = _mean_std(df["ig_topk_relevancy_mass"])
    nem_m, nem_s = _mean_std(df["nem_relevancy_mass"])
    nemk_m, nemk_s = _mean_std(df["nem_topk_relevancy_mass"])
    rise_m, rise_s = _mean_std(df["rise_relevancy_mass"])
    risek_m, risek_s = _mean_std(df["rise_topk_relevancy_mass"])

    summary = {
        "n": int(len(df)),
        "topk_frac": float(args.topk_frac),

        "ig_relevancy_mass_mean": ig_m,
        "ig_relevancy_mass_std": ig_s,
        "ig_topk_relevancy_mass_mean": igk_m,
        "ig_topk_relevancy_mass_std": igk_s,
        "ig_pointing_hit_rate": _hit_rate(df["ig_pointing_hit"]),

        "nem_relevancy_mass_mean": nem_m,
        "nem_relevancy_mass_std": nem_s,
        "nem_topk_relevancy_mass_mean": nemk_m,
        "nem_topk_relevancy_mass_std": nemk_s,
        "nem_pointing_hit_rate": _hit_rate(df["nem_pointing_hit"]),

        "rise_relevancy_mass_mean": rise_m,
        "rise_relevancy_mass_std": rise_s,
        "rise_topk_relevancy_mass_mean": risek_m,
        "rise_topk_relevancy_mass_std": risek_s,
        "rise_pointing_hit_rate": _hit_rate(df["rise_pointing_hit"]),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "localisation_summary.csv", index=False)

    sort_col = args.viz_sort_by
    if sort_col not in df.columns:
        sort_col = "rise_topk_relevancy_mass" if "rise_topk_relevancy_mass" in df.columns else "nem_topk_relevancy_mass"

    df_sort = df.copy()
    df_sort[sort_col] = pd.to_numeric(df_sort[sort_col], errors="coerce")

    # best/worst by chosen col
    half = max(1, args.viz_max // 2)
    df_viz = pd.concat([
        df_sort.sort_values(sort_col, ascending=False).head(half),
        df_sort.sort_values(sort_col, ascending=True).head(half),
    ]).drop_duplicates(subset=["gi", "uid"], keep="first").head(args.viz_max)

    for _, r in df_viz.iterrows():
        p = Path(r["npz_path"])
        d = np.load(str(p), allow_pickle=True)
        meta = _load_meta_json(d)

        x3d = _to_3d(d["x_5d"])
        base2d = _norm01(x3d[int(np.clip(int(round((x3d.shape[0] - 1) / 2.0)), 0, x3d.shape[0] - 1))])

        # reconstruct region + choose z slice at annotation center
        cand_xyz = np.array([float(meta["x"]), float(meta["y"]), float(meta["z"])], dtype=np.float32)
        ann_xyz = np.array([float(r["ann_x"]), float(r["ann_y"]), float(r["ann_z"])], dtype=np.float32)
        diam_mm = float(r["ann_diameter_mm"])
        r_vox = (diam_mm / 2.0) / float(args.spacing_mm)

        ann_center_zyx = _world_xyz_to_patch_zyx(x3d.shape, cand_xyz, ann_xyz, spacing_mm=args.spacing_mm)
        cz, cy, cx = ann_center_zyx
        cz = float(np.clip(cz, 0.0, x3d.shape[0] - 1))
        cy = float(np.clip(cy, 0.0, x3d.shape[1] - 1))
        cx = float(np.clip(cx, 0.0, x3d.shape[2] - 1))
        z_idx = int(np.clip(int(round(cz)), 0, x3d.shape[0] - 1))

        base2d = _norm01(x3d[z_idx])

        # circle cross-section radius at this z
        dz = abs(z_idx - cz)
        r2d = math.sqrt(max(0.0, r_vox * r_vox - dz * dz)) if dz <= r_vox else 0.0

        # load + prep 2d heats (if present)
        ig_raw = _npz_get_attr(d, ["ig_attr_5d", "ig_5d", "attr_ig_5d"])
        nem_raw = _npz_get_attr(d, ["nem_attr_5d", "nem_5d", "attr_nem_5d"])
        rise_raw = _npz_get_attr(d, ["rise_attr_5d", "rise_5d", "attr_rise_5d"])

        ig2d = _prep_attr("ig", _to_3d(ig_raw))[z_idx] if ig_raw is not None else None
        nem2d = _prep_attr("nem", _to_3d(nem_raw))[z_idx] if nem_raw is not None else None
        rise2d = _prep_attr("rise", _to_3d(rise_raw))[z_idx] if rise_raw is not None else None

        # figure: CT + IG + NEM + RISE
        fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))

        axs[0].imshow(base2d, cmap="gray", origin="lower")
        axs[0].axis("off")
        axs[0].set_title("CT (axial)")

        def _panel(ax, heat2d, title):
            if heat2d is None:
                ax.imshow(base2d, cmap="gray", origin="lower")
                ax.set_title(title + "\n(missing)")
                ax.axis("off")
            else:
                _draw_overlay(ax, base2d, heat2d, title=title, alpha=float(args.viz_alpha))

        _panel(
            axs[1], ig2d,
            title=f"IG\nmass={r['ig_relevancy_mass']:.3f} | topk={r['ig_topk_relevancy_mass']:.3f} | hit={int(r['ig_pointing_hit']) if np.isfinite(r['ig_pointing_hit']) else 'NA'}"
        )
        _panel(
            axs[2], nem2d,
            title=f"NEM\nmass={r['nem_relevancy_mass']:.3f} | topk={r['nem_topk_relevancy_mass']:.3f} | hit={int(r['nem_pointing_hit']) if np.isfinite(r['nem_pointing_hit']) else 'NA'}"
        )
        _panel(
            axs[3], rise2d,
            title=f"RISE\nmass={r['rise_relevancy_mass']:.3f} | topk={r['rise_topk_relevancy_mass']:.3f} | hit={int(r['rise_pointing_hit']) if np.isfinite(r['rise_pointing_hit']) else 'NA'}"
        )

        # sphere cross-section circle on all panels
        center_yx = (cx, cy)
        for ax in axs:
            _circle_for_slice(ax, center_yx=center_yx, radius_px=r2d, color="cyan", lw=1.3)

        gi = r.get("gi", "NA")
        uid = r.get("uid", "NA")
        fig.suptitle(
            f"gi={gi} | uid={uid} | z={z_idx} | ann_dist={float(r['ann_dist_mm']):.2f}mm | diam={diam_mm:.1f}mm | topk_frac={float(args.topk_frac):.3f}",
            y=0.98
        )
        plt.tight_layout()
        out_png = out_dir / f"loc_gi{gi}_z{z_idx}.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"[ok] Wrote:\n  {out_dir / 'localisation_per_case.csv'}\n  {out_dir / 'localisation_summary.csv'}\n  PNGs in: {out_dir}")


if __name__ == "__main__":
    main()
