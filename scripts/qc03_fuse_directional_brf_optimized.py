
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperspectral_pointcloud_fusion.common import ensure_dir, load_config, save_json


CSV_ENCODING = "utf-8-sig"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse overlap-normalized directional BRF patch spectra into an optimized final point spectrum table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "project_config.yaml"))
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--spectra-name", default="directional_brf_spectra_patch5_overlap_normalized.npy")
    parser.add_argument("--output-name", default="fused_point_spectra_optimized.npy")
    parser.add_argument("--fallback-spectra-name", default="fused_point_spectra.npy")
    parser.add_argument("--quality-levels", default="high,medium")
    parser.add_argument("--max-offaxis", type=float, default=35.0)
    parser.add_argument("--min-empty-cone", type=float, default=8.0)
    parser.add_argument("--min-border", type=float, default=30.0)
    parser.add_argument("--min-surface-view-cos", type=float, default=0.35)
    parser.add_argument("--quality-weight-high", type=float, default=1.0)
    parser.add_argument("--quality-weight-medium", type=float, default=0.7)
    parser.add_argument("--quality-weight-low", type=float, default=0.25)
    parser.add_argument("--scene-gain-weight-mode", choices=["none", "stability"], default="stability")
    parser.add_argument("--scene-gain-weight-csv", default="")
    parser.add_argument("--scene-gain-weight-bands", default="31,51,73,100")
    parser.add_argument("--scene-gain-weight-scale", type=float, default=0.5)
    parser.add_argument("--scene-gain-weight-min", type=float, default=0.35)
    parser.add_argument("--scene-gain-boundary-penalty", type=float, default=0.5)
    parser.add_argument("--value-clip-mode", choices=["none", "global_quantile"], default="global_quantile")
    parser.add_argument("--value-clip-low-percentile", type=float, default=0.5)
    parser.add_argument("--value-clip-high-percentile", type=float, default=98.0)
    parser.add_argument("--value-clip-margin-fraction", type=float, default=0.0)
    parser.add_argument("--value-clip-reflectance-nonnegative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value-clip-index-physical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--band-chunk-size", type=int, default=16)
    parser.add_argument("--recompute-derived", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def parse_int_set(text: str) -> set[int]:
    return {int(part.strip()) for part in str(text).split(",") if part.strip()}


def load_observations(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    required = {
        "observation_id",
        "point_id",
        "scene_id",
        "is_clear",
        "view_weight_raw",
        "brf_quality_level",
        "offaxis_deg",
        "border_dist_px",
        "local_empty_cone_deg",
        "surface_view_cos",
    }
    existing = set(parquet_columns(path))
    missing = sorted(required - existing)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    obs = pd.read_parquet(path, columns=sorted(required))
    obs["scene_id"] = obs["scene_id"].astype(str)
    obs["observation_id"] = pd.to_numeric(obs["observation_id"], errors="raise").astype(np.int64)
    obs["point_id"] = pd.to_numeric(obs["point_id"], errors="coerce").astype("Int64")
    for col in ["view_weight_raw", "offaxis_deg", "border_dist_px", "local_empty_cone_deg", "surface_view_cos"]:
        obs[col] = pd.to_numeric(obs[col], errors="coerce")

    levels = {part.strip().lower() for part in str(args.quality_levels).split(",") if part.strip()}
    mask = obs["point_id"].notna().to_numpy()
    mask &= obs["is_clear"].fillna(False).astype(bool).to_numpy()
    if levels:
        mask &= obs["brf_quality_level"].astype(str).str.lower().isin(levels).to_numpy()
    mask &= obs["offaxis_deg"].to_numpy(dtype=np.float64) <= float(args.max_offaxis)
    mask &= obs["border_dist_px"].to_numpy(dtype=np.float64) >= float(args.min_border)
    mask &= obs["local_empty_cone_deg"].to_numpy(dtype=np.float64) >= float(args.min_empty_cone)
    mask &= obs["surface_view_cos"].to_numpy(dtype=np.float64) >= float(args.min_surface_view_cos)
    obs = obs.loc[mask].copy().reset_index(drop=True)
    if obs.empty:
        raise ValueError("No observations remain after optimized fusion filtering.")
    obs["point_id"] = obs["point_id"].astype(np.int64)
    obs["view_weight_raw"] = obs["view_weight_raw"].fillna(1.0).clip(lower=0.0)
    obs.loc[obs["view_weight_raw"] <= 0, "view_weight_raw"] = 1.0
    return obs


def default_gain_csv(input_dir: Path, spectra_name: str) -> Path:
    if str(spectra_name) == "directional_brf_spectra_patch5_overlap_normalized.npy":
        return input_dir / "directional_brf_patch5_overlap_scene_band_gains.csv"
    if str(spectra_name) == "directional_brf_spectra_overlap_normalized.npy":
        return input_dir / "directional_brf_overlap_scene_band_gains.csv"
    stem = Path(str(spectra_name)).stem
    if stem.endswith("_normalized"):
        stem = stem[: -len("_normalized")]
    return input_dir / f"{stem}_scene_band_gains.csv"


def scene_gain_weights(obs: pd.DataFrame, input_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if str(args.scene_gain_weight_mode) == "none":
        return np.ones(len(obs), dtype=np.float64), {"scene_gain_weight_mode": "none"}
    path = Path(args.scene_gain_weight_csv) if str(args.scene_gain_weight_csv).strip() else default_gain_csv(input_dir, args.spectra_name)
    if not path.exists():
        raise FileNotFoundError(path)
    gains = pd.read_csv(path)
    required = {"scene_id", "band_number", "gain"}
    missing = sorted(required - set(gains.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    bands = parse_int_set(args.scene_gain_weight_bands)
    gains["scene_id"] = gains["scene_id"].astype(str)
    gains["band_number"] = pd.to_numeric(gains["band_number"], errors="coerce")
    gains["gain"] = pd.to_numeric(gains["gain"], errors="coerce")
    gains = gains[np.isfinite(gains["gain"].to_numpy(dtype=np.float64)) & (gains["gain"].to_numpy(dtype=np.float64) > EPS)].copy()
    if bands:
        gains = gains[gains["band_number"].astype(int).isin(bands)].copy()
    if gains.empty:
        raise ValueError(f"No usable gain rows in {path}")

    log_gain = np.log(gains["gain"].to_numpy(dtype=np.float64))
    gains["abs_log_gain"] = np.abs(log_gain)
    gains["log_gain"] = log_gain
    gain_min = float(gains["gain"].min())
    gain_max = float(gains["gain"].max())
    gains["is_boundary"] = (gains["gain"] <= gain_min + 1.0e-9) | (gains["gain"] >= gain_max - 1.0e-9)
    stats = (
        gains.groupby("scene_id", sort=False)
        .agg(
            mean_abs_log_gain=("abs_log_gain", "mean"),
            std_log_gain=("log_gain", "std"),
            boundary_fraction=("is_boundary", "mean"),
            band_count=("gain", "size"),
        )
        .reset_index()
    )
    stats["std_log_gain"] = stats["std_log_gain"].fillna(0.0)
    instability = stats["mean_abs_log_gain"].to_numpy(dtype=np.float64) + stats["std_log_gain"].to_numpy(dtype=np.float64)
    weights = np.exp(-instability / max(float(args.scene_gain_weight_scale), EPS))
    penalty = min(max(float(args.scene_gain_boundary_penalty), 0.0), 1.0)
    weights *= 1.0 - penalty * stats["boundary_fraction"].to_numpy(dtype=np.float64)
    weights = np.clip(weights, float(args.scene_gain_weight_min), 1.0)
    stats["scene_gain_weight"] = weights
    lookup = dict(zip(stats["scene_id"], stats["scene_gain_weight"]))
    row_weights = obs["scene_id"].map(lookup).fillna(1.0).to_numpy(dtype=np.float64)
    return row_weights, {
        "scene_gain_weight_mode": str(args.scene_gain_weight_mode),
        "scene_gain_weight_csv": str(path),
        "scene_gain_weight_bands": sorted(bands),
        "scene_gain_weight_min_actual": float(stats["scene_gain_weight"].min()),
        "scene_gain_weight_median_actual": float(stats["scene_gain_weight"].median()),
        "scene_gain_weight_max_actual": float(stats["scene_gain_weight"].max()),
    }


def quality_view_weights(obs: pd.DataFrame, scene_weights: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    q = obs["brf_quality_level"].astype(str).str.lower().to_numpy()
    weights = np.full(len(obs), float(args.quality_weight_low), dtype=np.float64)
    weights[q == "medium"] = float(args.quality_weight_medium)
    weights[q == "high"] = float(args.quality_weight_high)
    view_w = obs["view_weight_raw"].to_numpy(dtype=np.float64)
    view_w = np.where(np.isfinite(view_w) & (view_w > 0), view_w, 1.0)
    weights *= view_w
    weights *= scene_weights
    weights = np.where(np.isfinite(weights) & (weights > EPS), weights, 1.0)
    med = float(np.nanmedian(weights)) if weights.size else 1.0
    if np.isfinite(med) and med > EPS:
        weights = weights / med
    return weights


def nearest_band(meta: pd.DataFrame, target_nm: float, fallback_band: int) -> int:
    wav = pd.to_numeric(meta["wavelength_nm"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(wav) & (wav > 0)
    if np.any(valid):
        dist = np.abs(wav - float(target_nm))
        dist[~valid] = np.inf
        idx = int(np.argmin(dist))
        if np.isfinite(dist[idx]) and dist[idx] <= 3.0:
            return int(meta.iloc[idx]["band_index_zero_based"])
    row = meta.loc[meta["band_number"].astype(int) == int(fallback_band)]
    if row.empty:
        raise ValueError(f"Cannot find band for {target_nm} nm or fallback band {fallback_band}")
    return int(row.iloc[0]["band_index_zero_based"])


def recompute_indices(fused: np.ndarray, meta: pd.DataFrame) -> None:
    band_numbers = set(meta["band_number"].astype(int))
    if not {165, 166, 167, 168}.issubset(band_numbers):
        return
    red = nearest_band(meta, 670.0, 81)
    nir850 = nearest_band(meta, 850.0, 126)
    nir802 = nearest_band(meta, 802.0, 114)
    red_edge = nearest_band(meta, 718.0, 93)
    green = nearest_band(meta, 550.0, 51)
    idx165 = int(meta.loc[meta["band_number"].astype(int) == 165, "band_index_zero_based"].iloc[0])
    idx166 = int(meta.loc[meta["band_number"].astype(int) == 166, "band_index_zero_based"].iloc[0])
    idx167 = int(meta.loc[meta["band_number"].astype(int) == 167, "band_index_zero_based"].iloc[0])
    idx168 = int(meta.loc[meta["band_number"].astype(int) == 168, "band_index_zero_based"].iloc[0])

    red_v = fused[:, red]
    nir850_v = fused[:, nir850]
    nir802_v = fused[:, nir802]
    red_edge_v = fused[:, red_edge]
    green_v = fused[:, green]
    fused[:, idx165] = np.divide(nir850_v - red_v, nir850_v + red_v, out=np.zeros_like(red_v), where=np.abs(nir850_v + red_v) > EPS)
    fused[:, idx166] = np.divide(nir802_v, red_edge_v, out=np.zeros_like(nir802_v), where=np.abs(red_edge_v) > EPS)
    msavi_arg = np.maximum((2.0 * nir802_v + 1.0) ** 2 - 8.0 * (nir802_v - red_edge_v), 0.0)
    fused[:, idx167] = (2.0 * nir802_v + 1.0 - np.sqrt(msavi_arg)) / 2.0
    fused[:, idx168] = np.divide(green_v - nir850_v, green_v + nir850_v, out=np.zeros_like(green_v), where=np.abs(green_v + nir850_v) > EPS)


def clip_value_chunk(vals: np.ndarray, band_meta: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    out = np.asarray(vals, dtype=np.float64).copy()
    if str(args.value_clip_mode) == "global_quantile":
        q_lo = min(max(float(args.value_clip_low_percentile), 0.0), 100.0)
        q_hi = min(max(float(args.value_clip_high_percentile), q_lo), 100.0)
        for local_col, (_, band) in enumerate(band_meta.iterrows()):
            col = out[:, local_col]
            finite = np.isfinite(col)
            if not np.any(finite):
                continue
            lo = float(np.nanpercentile(col[finite], q_lo))
            hi = float(np.nanpercentile(col[finite], q_hi))
            span = max(float(hi - lo), EPS)
            margin = max(float(args.value_clip_margin_fraction), 0.0) * span
            out[:, local_col] = np.clip(col, lo - margin, hi + margin)

    band_numbers = band_meta["band_number"].astype(int).to_numpy()
    if bool(args.value_clip_reflectance_nonnegative):
        reflectance_cols = np.flatnonzero(band_numbers <= 164)
        if reflectance_cols.size:
            out[:, reflectance_cols] = np.maximum(out[:, reflectance_cols], 0.0)
    if bool(args.value_clip_index_physical):
        index_cols = np.flatnonzero(np.isin(band_numbers, [165, 167, 168]))
        if index_cols.size:
            out[:, index_cols] = np.clip(out[:, index_cols], -1.0, 1.0)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = Path(args.input_dir) if str(args.input_dir).strip() else Path(cfg["outputs"]["directional_brf_dir"])
    fusion_dir = ensure_dir(cfg["outputs"]["fusion_dir"])
    point_dir = Path(cfg["outputs"]["pointcloud_dir"])

    obs_path = input_dir / "point_directional_brf_observations.parquet"
    spectra_path = input_dir / str(args.spectra_name)
    meta_path = input_dir / "directional_brf_band_metadata.csv"
    point_xyz_path = point_dir / "point_xyz.npy"
    for path in [obs_path, spectra_path, meta_path, point_xyz_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    out_path = fusion_dir / str(args.output_name)
    fallback_path = fusion_dir / str(args.fallback_spectra_name)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite")

    t0 = time.perf_counter()
    obs = load_observations(obs_path, args)
    spectra = np.load(spectra_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    if "band_number" not in meta.columns:
        meta["band_number"] = np.arange(1, spectra.shape[1] + 1, dtype=np.int32)
    if "band_index_zero_based" not in meta.columns:
        meta["band_index_zero_based"] = np.arange(spectra.shape[1], dtype=np.int32)
    if "wavelength_nm" not in meta.columns:
        meta["wavelength_nm"] = np.nan
    meta["band_number"] = pd.to_numeric(meta["band_number"], errors="raise").astype(int)
    meta["band_index_zero_based"] = pd.to_numeric(meta["band_index_zero_based"], errors="raise").astype(int)
    meta["wavelength_nm"] = pd.to_numeric(meta["wavelength_nm"], errors="coerce")
    meta = meta.sort_values("band_index_zero_based", kind="mergesort").reset_index(drop=True)
    n_points = int(np.load(point_xyz_path, mmap_mode="r").shape[0])
    if spectra.ndim != 2:
        raise ValueError(f"Expected spectra [observation, band], got {spectra.shape}")
    if spectra.shape[1] != len(meta):
        raise ValueError(f"Spectra width {spectra.shape[1]} does not match metadata rows {len(meta)}")

    obs_ids = obs["observation_id"].to_numpy(dtype=np.int64)
    point_ids = obs["point_id"].to_numpy(dtype=np.int64)
    if obs_ids.min() < 0 or obs_ids.max() >= spectra.shape[0]:
        raise ValueError("observation_id is not aligned with spectra rows")
    if point_ids.min() < 0 or point_ids.max() >= n_points:
        raise ValueError("point_id is outside point cloud row range")

    scene_weights, scene_weight_summary = scene_gain_weights(obs, input_dir, args)
    weights = quality_view_weights(obs, scene_weights, args)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_points, spectra.shape[1]))
    out[:] = np.nan
    valid_point_mask = np.zeros(n_points, dtype=bool)
    point_weight_sum = np.bincount(point_ids, weights=weights, minlength=n_points).astype(np.float64)
    valid_point_mask = point_weight_sum > EPS
    chunk_size = max(1, int(args.band_chunk_size))
    print(f"Optimized fusion observations={len(obs)}, points={n_points}, bands={spectra.shape[1]}")
    for start in range(0, spectra.shape[1], chunk_size):
        stop = min(spectra.shape[1], start + chunk_size)
        vals = np.asarray(spectra[obs_ids, start:stop], dtype=np.float64)
        vals = clip_value_chunk(vals, meta.iloc[start:stop], args)
        fused_chunk = np.full((n_points, stop - start), np.nan, dtype=np.float64)
        for local_col in range(stop - start):
            col_vals = vals[:, local_col]
            good = np.isfinite(col_vals)
            if not np.any(good):
                continue
            num = np.bincount(point_ids[good], weights=weights[good] * col_vals[good], minlength=n_points).astype(np.float64)
            den = np.bincount(point_ids[good], weights=weights[good], minlength=n_points).astype(np.float64)
            np.divide(num, den, out=fused_chunk[:, local_col], where=den > EPS)
        out[:, start:stop] = fused_chunk.astype(np.float32, copy=False)
        print(f"[{stop}/{spectra.shape[1]}] elapsed={time.perf_counter() - t0:.1f}s")

    if args.recompute_derived:
        recompute_indices(out, meta)
    fallback_used = 0
    if fallback_path.exists():
        fallback = np.load(fallback_path, mmap_mode="r")
        if fallback.shape == out.shape:
            bad_rows = ~np.all(np.isfinite(out), axis=1)
            fallback_used = int(np.count_nonzero(bad_rows))
            if fallback_used:
                out[bad_rows, :] = np.asarray(fallback[bad_rows, :], dtype=np.float32)
        else:
            print(f"Fallback spectra shape {fallback.shape} does not match optimized shape {out.shape}; skipping fallback.")
    out.flush()
    del out

    band_numbers = meta["band_number"].to_numpy(dtype=np.int32)
    np.save(fusion_dir / "fused_band_numbers_optimized.npy", band_numbers)
    np.save(fusion_dir / "optimized_fused_weight_sum.npy", point_weight_sum.astype(np.float32))
    np.save(fusion_dir / "optimized_valid_point_mask.npy", valid_point_mask)
    summary = {
        "input_dir": str(input_dir),
        "input_spectra_path": str(spectra_path),
        "output_spectra_path": str(out_path),
        "fallback_spectra_path": str(fallback_path) if fallback_path.exists() else "",
        "fallback_point_count": int(fallback_used),
        "observation_rows_after_filter": int(len(obs)),
        "point_count": int(n_points),
        "valid_point_count": int(np.count_nonzero(valid_point_mask)),
        "valid_point_ratio": float(np.mean(valid_point_mask)) if valid_point_mask.size else 0.0,
        "quality_levels": str(args.quality_levels),
        "max_offaxis": float(args.max_offaxis),
        "min_empty_cone": float(args.min_empty_cone),
        "min_border": float(args.min_border),
        "min_surface_view_cos": float(args.min_surface_view_cos),
        "quality_weight_high": float(args.quality_weight_high),
        "quality_weight_medium": float(args.quality_weight_medium),
        "quality_weight_low": float(args.quality_weight_low),
        **scene_weight_summary,
        "value_clip_mode": str(args.value_clip_mode),
        "value_clip_low_percentile": float(args.value_clip_low_percentile),
        "value_clip_high_percentile": float(args.value_clip_high_percentile),
        "value_clip_margin_fraction": float(args.value_clip_margin_fraction),
        "value_clip_reflectance_nonnegative": bool(args.value_clip_reflectance_nonnegative),
        "value_clip_index_physical": bool(args.value_clip_index_physical),
        "recompute_derived": bool(args.recompute_derived),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    save_json(summary, fusion_dir / "optimized_directional_fusion_summary.json")
    pd.DataFrame([summary]).to_csv(fusion_dir / "optimized_directional_fusion_summary.csv", index=False, encoding=CSV_ENCODING)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
