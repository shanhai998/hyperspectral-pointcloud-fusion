
from __future__ import annotations

import argparse
import json
import re
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


DEFAULT_OUTPUT_NAME = "directional_brf_spectra_overlap_normalized.npy"
DEFAULT_INPUT_NAME = "directional_brf_spectra.npy"
CSV_ENCODING = "utf-8-sig"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate overlap-point scene-band gains and write normalized directional BRF spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "project_config.yaml"))
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--input-spectra-name", default=DEFAULT_INPUT_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--coeff-prefix", default="", help="Prefix for gain/summary CSV files. Empty derives from input/output names.")
    parser.add_argument("--normalize-bands", default="1:164")
    parser.add_argument("--quality-levels", default="high,medium")
    parser.add_argument("--min-reflectance", type=float, default=1.0e-6)
    parser.add_argument("--max-reflectance", type=float, default=2.0)
    parser.add_argument("--trim-low", type=float, default=1.0)
    parser.add_argument("--trim-high", type=float, default=99.0)
    parser.add_argument("--min-point-observations", type=int, default=2)
    parser.add_argument("--min-scene-observations", type=int, default=1000)
    parser.add_argument("--min-fit-rows", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--gain-min", type=float, default=0.5)
    parser.add_argument("--gain-max", type=float, default=2.0)
    parser.add_argument("--recompute-derived", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bands-only", default="", help="Debug comma-list of band numbers. Empty uses --normalize-bands.")
    return parser.parse_args()


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def parse_band_spec(text: str, max_band: int) -> list[int]:
    text = str(text).strip()
    if not text:
        return []
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi = part.split(":", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    out = sorted(set(values))
    bad = [b for b in out if b < 1 or b > max_band]
    if bad:
        raise ValueError(f"Band numbers outside 1-{max_band}: {bad[:10]}")
    return out


def scene_numeric_suffix(scene_id: Any) -> int:
    m = re.search(r"(\d+)$", str(scene_id))
    return int(m.group(1)) if m else 10**9


def read_inputs(input_dir: Path, spectra_name: str) -> tuple[pd.DataFrame, np.memmap, pd.DataFrame]:
    obs_path = input_dir / "point_directional_brf_observations.parquet"
    spectra_path = input_dir / str(spectra_name)
    meta_path = input_dir / "directional_brf_band_metadata.csv"
    for path in [obs_path, spectra_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    existing = set(parquet_columns(obs_path))
    required = {
        "observation_id",
        "point_id",
        "scene_id",
        "is_clear",
        "view_weight_raw",
        "brf_quality_level",
    }
    missing = sorted(required - existing)
    if missing:
        raise ValueError(f"Missing required columns in {obs_path}: {missing}")
    obs = pd.read_parquet(obs_path, columns=sorted(required))
    obs["observation_id"] = pd.to_numeric(obs["observation_id"], errors="raise").astype(np.int64)
    obs["scene_id"] = obs["scene_id"].astype(str)
    obs["brf_quality_level"] = obs["brf_quality_level"].astype(str).str.lower()
    obs["is_clear"] = obs["is_clear"].fillna(False).astype(bool)
    obs["view_weight_raw"] = pd.to_numeric(obs["view_weight_raw"], errors="coerce").fillna(1.0).clip(lower=0.0)

    spectra = np.load(spectra_path, mmap_mode="r")
    if spectra.ndim != 2:
        raise ValueError(f"Expected spectra [observation, band], got shape {spectra.shape}")
    if obs["observation_id"].min() < 0 or obs["observation_id"].max() >= spectra.shape[0]:
        raise ValueError("observation_id is not aligned with spectra rows")

    meta = pd.read_csv(meta_path)
    if "band_number" not in meta.columns:
        meta["band_number"] = np.arange(1, len(meta) + 1, dtype=np.int32)
    if "band_index_zero_based" not in meta.columns:
        meta["band_index_zero_based"] = np.arange(len(meta), dtype=np.int32)
    if "wavelength_nm" not in meta.columns:
        meta["wavelength_nm"] = np.nan
    meta["band_number"] = pd.to_numeric(meta["band_number"], errors="raise").astype(int)
    meta["band_index_zero_based"] = pd.to_numeric(meta["band_index_zero_based"], errors="raise").astype(int)
    meta["wavelength_nm"] = pd.to_numeric(meta["wavelength_nm"], errors="coerce")
    return obs, spectra, meta


def quality_mask(obs: pd.DataFrame, quality_levels: str) -> np.ndarray:
    levels = {v.strip().lower() for v in str(quality_levels).split(",") if v.strip()}
    mask = obs["is_clear"].to_numpy(dtype=bool)
    if levels:
        mask &= obs["brf_quality_level"].isin(levels).to_numpy(dtype=bool)
    return mask


def factorize_scene_ids(scene_ids: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    unique = pd.Series(scene_ids.astype(str).unique(), dtype="string")
    mapping = pd.DataFrame({"scene_id": unique})
    mapping["scene_numeric_suffix"] = mapping["scene_id"].map(scene_numeric_suffix)
    mapping = mapping.sort_values(["scene_numeric_suffix", "scene_id"], kind="mergesort").reset_index(drop=True)
    mapping["scene_norm_code"] = np.arange(len(mapping), dtype=np.int32)
    lookup = dict(zip(mapping["scene_id"].astype(str), mapping["scene_norm_code"].astype(int)))
    return scene_ids.astype(str).map(lookup).to_numpy(dtype=np.int32), mapping


def weighted_bincount_mean(values: np.ndarray, labels: np.ndarray, weights: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    denom = np.bincount(labels, weights=weights, minlength=size).astype(np.float64)
    numer = np.bincount(labels, weights=weights * values, minlength=size).astype(np.float64)
    mean = np.zeros(size, dtype=np.float64)
    np.divide(numer, denom, out=mean, where=denom > 0)
    return mean, denom


def fit_scene_gains_for_band(
    values: np.ndarray,
    point_codes: np.ndarray,
    scene_codes: np.ndarray,
    base_mask: np.ndarray,
    raw_weights: np.ndarray,
    n_points: int,
    n_scenes: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_idx = np.flatnonzero(base_mask)
    if candidate_idx.size == 0:
        return np.ones(n_scenes, dtype=np.float64), {"fit_rows": 0, "status": "no_quality_rows"}

    vals = np.asarray(values[candidate_idx], dtype=np.float64)
    local = np.isfinite(vals) & (vals > float(args.min_reflectance)) & (vals < float(args.max_reflectance))
    candidate_idx = candidate_idx[local]
    vals = vals[local]
    if candidate_idx.size < int(args.min_fit_rows):
        return np.ones(n_scenes, dtype=np.float64), {"fit_rows": int(candidate_idx.size), "status": "too_few_reflectance_rows"}

    pc0 = point_codes[candidate_idx]
    point_counts = np.bincount(pc0, minlength=n_points)
    candidate_idx = candidate_idx[point_counts[pc0] >= int(args.min_point_observations)]
    vals = np.asarray(values[candidate_idx], dtype=np.float64)
    if candidate_idx.size < int(args.min_fit_rows):
        return np.ones(n_scenes, dtype=np.float64), {"fit_rows": int(candidate_idx.size), "status": "too_few_overlap_rows"}

    log_vals = np.log(np.clip(vals, float(args.min_reflectance), None))
    lo = np.nanpercentile(log_vals, float(args.trim_low))
    hi = np.nanpercentile(log_vals, float(args.trim_high))
    keep = np.isfinite(log_vals) & (log_vals >= lo) & (log_vals <= hi)
    candidate_idx = candidate_idx[keep]
    y = log_vals[keep]
    pc = point_codes[candidate_idx]
    sc = scene_codes[candidate_idx]
    w = np.asarray(raw_weights[candidate_idx], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    if candidate_idx.size < int(args.min_fit_rows):
        return np.ones(n_scenes, dtype=np.float64), {"fit_rows": int(candidate_idx.size), "status": "too_few_trimmed_rows"}

    scene_count = np.bincount(sc, minlength=n_scenes)
    scene_valid = scene_count >= int(args.min_scene_observations)
    if np.count_nonzero(scene_valid) < 2:
        return np.ones(n_scenes, dtype=np.float64), {"fit_rows": int(candidate_idx.size), "status": "too_few_valid_scenes"}

    scene_eff = np.zeros(n_scenes, dtype=np.float64)
    point_eff = np.zeros(n_points, dtype=np.float64)
    iterations = 0
    max_delta = np.inf
    for iterations in range(1, int(args.max_iter) + 1):
        point_eff_new, point_w = weighted_bincount_mean(y - scene_eff[sc], pc, w, n_points)
        point_eff[point_w > 0] = point_eff_new[point_w > 0]

        scene_new, scene_w = weighted_bincount_mean(y - point_eff[pc], sc, w, n_scenes)
        scene_new[scene_w <= 0] = 0.0
        valid = scene_valid & (scene_w > 0)
        center = np.average(scene_new[valid], weights=scene_w[valid]) if np.any(valid) else 0.0
        scene_new = scene_new - center
        scene_new[~valid] = 0.0
        max_delta = float(np.max(np.abs(scene_new - scene_eff)))
        scene_eff = scene_new
        if max_delta <= float(args.tol):
            break

    gains = np.exp(-scene_eff)
    gains[~scene_valid] = 1.0
    gains = np.clip(gains, float(args.gain_min), float(args.gain_max))
    return gains.astype(np.float64), {
        "fit_rows": int(candidate_idx.size),
        "status": "ok",
        "iterations": int(iterations),
        "max_delta": float(max_delta),
        "scene_valid_count": int(np.count_nonzero(scene_valid)),
        "gain_min": float(np.min(gains)),
        "gain_median": float(np.median(gains)),
        "gain_max": float(np.max(gains)),
    }


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


def recompute_indices(out: np.memmap, meta: pd.DataFrame) -> None:
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

    red_v = np.asarray(out[:, red], dtype=np.float32)
    nir850_v = np.asarray(out[:, nir850], dtype=np.float32)
    nir802_v = np.asarray(out[:, nir802], dtype=np.float32)
    red_edge_v = np.asarray(out[:, red_edge], dtype=np.float32)
    green_v = np.asarray(out[:, green], dtype=np.float32)

    out[:, idx165] = np.divide(nir850_v - red_v, nir850_v + red_v, out=np.zeros_like(red_v), where=np.abs(nir850_v + red_v) > EPS)
    out[:, idx166] = np.divide(nir802_v, red_edge_v, out=np.zeros_like(nir802_v), where=np.abs(red_edge_v) > EPS)
    msavi_arg = np.maximum((2.0 * nir802_v + 1.0) ** 2 - 8.0 * (nir802_v - red_edge_v), 0.0)
    out[:, idx167] = (2.0 * nir802_v + 1.0 - np.sqrt(msavi_arg)) / 2.0
    out[:, idx168] = np.divide(green_v - nir850_v, green_v + nir850_v, out=np.zeros_like(green_v), where=np.abs(green_v + nir850_v) > EPS)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = Path(args.input_dir) if args.input_dir else Path(cfg["outputs"]["directional_brf_dir"])
    out_path = input_dir / str(args.output_name)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to regenerate it")

    obs, spectra, meta = read_inputs(input_dir, spectra_name=str(args.input_spectra_name))
    normalize_bands = parse_band_spec(args.bands_only or args.normalize_bands, max_band=int(meta["band_number"].max()))
    band_lookup = dict(zip(meta["band_number"].astype(int), meta["band_index_zero_based"].astype(int)))
    missing = [b for b in normalize_bands if b not in band_lookup]
    if missing:
        raise ValueError(f"Requested bands are not in metadata: {missing[:10]}")

    point_codes, point_uniques = pd.factorize(obs["point_id"], sort=False)
    scene_codes, scene_mapping = factorize_scene_ids(obs["scene_id"])
    n_points = int(len(point_uniques))
    n_scenes = int(len(scene_mapping))
    obs_ids = obs["observation_id"].to_numpy(dtype=np.int64)
    raw_weights = obs["view_weight_raw"].to_numpy(dtype=np.float64)
    base_mask_obs = quality_mask(obs, args.quality_levels)
    base_mask_by_spectrum_row = np.zeros(spectra.shape[0], dtype=bool)
    base_mask_by_spectrum_row[obs_ids] = base_mask_obs
    point_codes_by_spectrum_row = np.full(spectra.shape[0], -1, dtype=np.int64)
    scene_codes_by_spectrum_row = np.full(spectra.shape[0], -1, dtype=np.int32)
    raw_weights_by_spectrum_row = np.ones(spectra.shape[0], dtype=np.float64)
    point_codes_by_spectrum_row[obs_ids] = point_codes
    scene_codes_by_spectrum_row[obs_ids] = scene_codes
    raw_weights_by_spectrum_row[obs_ids] = raw_weights

    print(f"Input spectra: {spectra.shape}, scenes={n_scenes}, points={n_points}")
    print(f"Quality rows used for gain fitting: {int(base_mask_obs.sum())}/{len(obs)}")
    print(f"Writing normalized spectra: {out_path}")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=spectra.dtype, shape=spectra.shape)
    coeff_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for band_col in range(spectra.shape[1]):
        band_row = meta.loc[meta["band_index_zero_based"].astype(int) == band_col]
        band_number = int(band_row.iloc[0]["band_number"]) if not band_row.empty else band_col + 1
        values = np.asarray(spectra[:, band_col], dtype=np.float32)
        if band_number in normalize_bands:
            gains, fit = fit_scene_gains_for_band(
                values=values,
                point_codes=point_codes_by_spectrum_row,
                scene_codes=scene_codes_by_spectrum_row,
                base_mask=base_mask_by_spectrum_row,
                raw_weights=raw_weights_by_spectrum_row,
                n_points=n_points,
                n_scenes=n_scenes,
                args=args,
            )
            norm_values = values * gains[scene_codes_by_spectrum_row]
            norm_values[scene_codes_by_spectrum_row < 0] = values[scene_codes_by_spectrum_row < 0]
            out[:, band_col] = norm_values.astype(spectra.dtype, copy=False)
            for scene_idx, scene in scene_mapping.iterrows():
                coeff_rows.append({
                    "band_number": band_number,
                    "band_index_zero_based": band_col,
                    "scene_norm_code": int(scene["scene_norm_code"]),
                    "scene_id": str(scene["scene_id"]),
                    "gain": float(gains[int(scene["scene_norm_code"])]),
                    **fit,
                })
        else:
            out[:, band_col] = values.astype(spectra.dtype, copy=False)
        if (band_col + 1) % 10 == 0 or band_col + 1 == spectra.shape[1]:
            print(f"[{band_col + 1}/{spectra.shape[1]}] elapsed={time.perf_counter() - t0:.1f}s")

    if args.recompute_derived:
        print("Recomputing derived bands 165-168 from normalized reflectance ...")
        recompute_indices(out, meta)
    out.flush()
    del out

    if args.coeff_prefix:
        coeff_prefix = str(args.coeff_prefix)
    elif str(args.input_spectra_name) == DEFAULT_INPUT_NAME and str(args.output_name) == DEFAULT_OUTPUT_NAME:
        coeff_prefix = "directional_brf_overlap"
    else:
        coeff_prefix = Path(str(args.output_name)).stem
    coeff_path = input_dir / f"{coeff_prefix}_scene_band_gains.csv"
    pd.DataFrame(coeff_rows).to_csv(coeff_path, index=False, encoding=CSV_ENCODING)
    scene_mapping_path = input_dir / f"{coeff_prefix}_scene_mapping.csv"
    scene_mapping.to_csv(scene_mapping_path, index=False, encoding=CSV_ENCODING)
    summary = {
        "input_dir": str(input_dir),
        "output_spectra_path": str(out_path),
        "source_spectra_path": str(input_dir / str(args.input_spectra_name)),
        "coefficients_csv": str(coeff_path),
        "scene_mapping_csv": str(scene_mapping_path),
        "coeff_prefix": str(coeff_prefix),
        "normalize_bands": normalize_bands,
        "quality_levels": args.quality_levels,
        "quality_rows": int(base_mask_obs.sum()),
        "observation_rows": int(len(obs)),
        "scene_count": n_scenes,
        "point_count": n_points,
        "recompute_derived": bool(args.recompute_derived),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    save_json(summary, input_dir / f"{coeff_prefix}_normalization_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
