
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperspectral_pointcloud_fusion.common import ensure_dir, load_config, save_json
from hyperspectral_pointcloud_fusion.envi import read_envi_bsq_memmap


DEFAULT_OUTPUT_NAME = "directional_brf_spectra_patch5.npy"
DEFAULT_SOURCE_NAME = "directional_brf_spectra.npy"
DEFAULT_INDEX_PLUS_ONE_BANDS = [165, 167, 168]
CSV_ENCODING = "utf-8-sig"
EPS = 1e-12

_CUBE_CACHE: dict[str, np.memmap] = {}
_CUBE_CACHE_LOCK = Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build footprint/patch-sampled directional BRF spectra aligned to point_directional_brf_observations.parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "project_config.yaml"))
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--source-spectra-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--patch-radius", type=int, default=2, help="Radius in pixels. 2 means a 5x5 footprint.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.15)
    parser.add_argument("--reducer", choices=["weighted_mean", "mean", "median"], default="weighted_mean")
    parser.add_argument("--offset-sampling", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--scene-offset-csv", default="", help="Optional CSV with scene_id, offset_u_px, offset_v_px applied before patch sampling.")
    parser.add_argument("--chunk-rows", type=int, default=8000)
    parser.add_argument("--max-observations", type=int, default=0, help="Debug limit. 0 means all observations.")
    parser.add_argument("--copy-source-for-unsampled", action="store_true", help="Fill unsampled rows from the source spectra. Mainly useful for debug outputs.")
    parser.add_argument("--recompute-derived", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-memmap-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def read_observations(path: Path, max_observations: int = 0) -> pd.DataFrame:
    required = {"observation_id", "scene_id", "u", "v"}
    existing = set(parquet_columns(path))
    missing = sorted(required - existing)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    obs = pd.read_parquet(path, columns=sorted(required))
    obs["observation_id"] = pd.to_numeric(obs["observation_id"], errors="raise").astype(np.int64)
    obs["scene_id"] = obs["scene_id"].astype(str)
    obs["u"] = pd.to_numeric(obs["u"], errors="coerce")
    obs["v"] = pd.to_numeric(obs["v"], errors="coerce")
    obs = obs[np.isfinite(obs["u"].to_numpy(dtype=np.float64)) & np.isfinite(obs["v"].to_numpy(dtype=np.float64))].copy()
    obs = obs.sort_values("observation_id", kind="mergesort").reset_index(drop=True)
    if max_observations and int(max_observations) > 0:
        obs = obs.iloc[: int(max_observations)].copy()
    return obs


def read_band_metadata(path: Path, spectra_width: int) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "band_number" not in meta.columns:
        meta["band_number"] = np.arange(1, len(meta) + 1, dtype=np.int32)
    if "band_index_zero_based" not in meta.columns:
        meta["band_index_zero_based"] = np.arange(len(meta), dtype=np.int32)
    if "wavelength_nm" not in meta.columns:
        meta["wavelength_nm"] = np.nan
    meta["band_number"] = pd.to_numeric(meta["band_number"], errors="raise").astype(int)
    meta["band_index_zero_based"] = pd.to_numeric(meta["band_index_zero_based"], errors="raise").astype(int)
    meta["wavelength_nm"] = pd.to_numeric(meta["wavelength_nm"], errors="coerce")
    bad = meta[(meta["band_index_zero_based"] < 0) | (meta["band_index_zero_based"] >= int(spectra_width))]
    if not bad.empty:
        raise ValueError(f"Band metadata indexes outside spectra width {spectra_width}")
    return meta.sort_values("band_index_zero_based", kind="mergesort").reset_index(drop=True)


def read_scene_offsets(path_text: str) -> dict[str, tuple[float, float]]:
    text = str(path_text).strip()
    if not text:
        return {}
    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"scene_id", "offset_u_px", "offset_v_px"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in scene offset CSV {path}: {missing}")
    offsets: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        scene_id = str(row["scene_id"])
        du = float(pd.to_numeric(row["offset_u_px"], errors="coerce"))
        dv = float(pd.to_numeric(row["offset_v_px"], errors="coerce"))
        if np.isfinite(du) and np.isfinite(dv):
            offsets[scene_id] = (du, dv)
    return offsets


def open_cube(scene_row: dict[str, Any]) -> np.memmap:
    return read_envi_bsq_memmap(
        scene_row["data_path"],
        samples=int(scene_row["samples"]),
        lines=int(scene_row["lines"]),
        bands=int(scene_row["bands"]),
        data_type=int(scene_row["data_type"]),
        byte_order=int(scene_row["byte_order"]),
        header_offset=int(scene_row["header_offset"]),
    )


def get_cube(scene_row: dict[str, Any], enable_cache: bool) -> np.memmap:
    if not enable_cache:
        return open_cube(scene_row)
    key = str(scene_row["data_path"])
    with _CUBE_CACHE_LOCK:
        cube = _CUBE_CACHE.get(key)
        if cube is None:
            cube = open_cube(scene_row)
            _CUBE_CACHE[key] = cube
        return cube


def patch_offsets(radius: int, sigma: float, reducer: str) -> tuple[list[tuple[int, int]], np.ndarray]:
    r = max(0, int(radius))
    offsets: list[tuple[int, int]] = []
    weights: list[float] = []
    sigma = float(sigma)
    if sigma <= 0:
        sigma = max(float(r) / 2.0, 1.0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            offsets.append((dx, dy))
            if reducer == "weighted_mean":
                weights.append(math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma)))
            else:
                weights.append(1.0)
    w = np.asarray(weights, dtype=np.float64)
    w /= max(float(w.sum()), EPS)
    return offsets, w


def sample_patch_selected_bands_bsq(
    cube: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    band_indices: list[int],
    scale_factor: float,
    offsets: list[tuple[int, int]],
    weights: np.ndarray,
    reducer: str,
    offset_sampling: str,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    m = len(band_indices)
    if n == 0:
        return np.empty((0, m), dtype=np.float32)

    if reducer == "median":
        samples = []
        for dx, dy in offsets:
            samples.append(sample_points_selected_bands_bsq(cube, x + float(dx), y + float(dy), band_indices, scale_factor, offset_sampling))
        stacked = np.stack(samples, axis=1)
        return np.nanmedian(stacked, axis=1).astype(np.float32, copy=False)

    acc = np.zeros((n, m), dtype=np.float64)
    denom = np.zeros((n, m), dtype=np.float64)
    for (dx, dy), weight in zip(offsets, weights):
        vals = sample_points_selected_bands_bsq(cube, x + float(dx), y + float(dy), band_indices, scale_factor, offset_sampling).astype(np.float64, copy=False)
        good = np.isfinite(vals)
        acc += np.where(good, vals * float(weight), 0.0)
        denom += np.where(good, float(weight), 0.0)
    out = np.divide(acc, denom, out=np.full_like(acc, np.nan), where=denom > EPS)
    return out.astype(np.float32, copy=False)


def sample_points_selected_bands_bsq(
    cube: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    band_indices: list[int],
    scale_factor: float,
    mode: str,
) -> np.ndarray:
    bands = np.asarray(list(band_indices), dtype=np.int64)
    _, height, width = cube.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    band_axis = bands[:, None]
    if mode == "nearest":
        xi = np.clip(np.rint(x).astype(np.int64), 0, width - 1)
        yi = np.clip(np.rint(y).astype(np.int64), 0, height - 1)
        out = np.asarray(cube[band_axis, yi[None, :], xi[None, :]], dtype=np.float64).T
    else:
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)
        x0 = np.clip(x0, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        wx = (x - x0).astype(np.float64)
        wy = (y - y0).astype(np.float64)
        p00 = np.asarray(cube[band_axis, y0[None, :], x0[None, :]], dtype=np.float64).T
        p10 = np.asarray(cube[band_axis, y0[None, :], x1[None, :]], dtype=np.float64).T
        p01 = np.asarray(cube[band_axis, y1[None, :], x0[None, :]], dtype=np.float64).T
        p11 = np.asarray(cube[band_axis, y1[None, :], x1[None, :]], dtype=np.float64).T
        val0 = (1.0 - wx)[:, None] * p00 + wx[:, None] * p10
        val1 = (1.0 - wx)[:, None] * p01 + wx[:, None] * p11
        out = (1.0 - wy)[:, None] * val0 + wy[:, None] * val1
    if scale_factor not in (0, 1, 1.0):
        out /= float(scale_factor)
    return out.astype(np.float32, copy=False)


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


def iter_slices(n: int, chunk_rows: int):
    chunk = max(1, int(chunk_rows))
    for start in range(0, int(n), chunk):
        stop = min(int(n), start + chunk)
        yield start, stop


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = Path(args.input_dir) if args.input_dir else Path(cfg["outputs"]["directional_brf_dir"])
    input_dir = ensure_dir(input_dir)
    obs_path = input_dir / "point_directional_brf_observations.parquet"
    source_spectra_path = input_dir / str(args.source_spectra_name)
    meta_path = input_dir / "directional_brf_band_metadata.csv"
    scene_db_path = Path(cfg["outputs"]["scene_db_dir"]) / "scene_database.csv"
    out_path = input_dir / str(args.output_name)

    for path in [obs_path, source_spectra_path, meta_path, scene_db_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to regenerate it")

    source = np.load(source_spectra_path, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError(f"Expected source spectra [observation, band], got {source.shape}")
    meta = read_band_metadata(meta_path, spectra_width=source.shape[1])
    band_indices = meta["band_index_zero_based"].astype(int).tolist()
    band_numbers = meta["band_number"].astype(int).tolist()
    index_plus_one = set(int(x) for x in cfg.get("processing", {}).get("index_band_numbers_stored_plus_one", DEFAULT_INDEX_PLUS_ONE_BANDS))
    index_local_indices = [i for i, b in enumerate(band_numbers) if int(b) in index_plus_one]

    obs = read_observations(obs_path, max_observations=int(args.max_observations))
    obs_ids = obs["observation_id"].to_numpy(dtype=np.int64)
    if obs_ids.size and (obs_ids.min() < 0 or obs_ids.max() >= source.shape[0]):
        raise ValueError("observation_id is not aligned with source spectra rows")

    scene_db = pd.read_csv(scene_db_path)
    scene_map = scene_db.drop_duplicates("scene_id").set_index("scene_id")
    offsets, weights = patch_offsets(int(args.patch_radius), float(args.gaussian_sigma), str(args.reducer))
    scene_offsets = read_scene_offsets(str(args.scene_offset_csv))

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=source.dtype, shape=source.shape)
    if bool(args.copy_source_for_unsampled):
        out[:] = np.asarray(source, dtype=source.dtype)

    print(f"Patch spectra output: {out_path}")
    print(f"Source shape: {source.shape}; observations to sample: {len(obs)}")
    print(f"Patch: radius={args.patch_radius}, samples={len(offsets)}, reducer={args.reducer}, offset_sampling={args.offset_sampling}")
    if scene_offsets:
        print(f"Applying scene offsets from {args.scene_offset_csv}: {len(scene_offsets)} scene(s)")
    t0 = time.perf_counter()
    sampled = 0
    scene_count = 0
    for scene_id, group in obs.groupby("scene_id", sort=False):
        scene_id = str(scene_id)
        if scene_id not in scene_map.index:
            print(f"WARNING: missing scene in scene_database.csv: {scene_id}")
            continue
        scene_row = scene_map.loc[scene_id].to_dict()
        cube = get_cube(scene_row, enable_cache=bool(args.enable_memmap_cache))
        offset_u, offset_v = scene_offsets.get(scene_id, (0.0, 0.0))
        scene_count += 1
        g = group.reset_index(drop=True)
        for start, stop in iter_slices(len(g), int(args.chunk_rows)):
            part = g.iloc[start:stop]
            spec = sample_patch_selected_bands_bsq(
                cube=cube,
                x=part["u"].to_numpy(dtype=np.float64) + float(offset_u),
                y=part["v"].to_numpy(dtype=np.float64) + float(offset_v),
                band_indices=band_indices,
                scale_factor=float(scene_row["reflectance_scale_factor"]),
                offsets=offsets,
                weights=weights,
                reducer=str(args.reducer),
                offset_sampling=str(args.offset_sampling),
            )
            if index_local_indices:
                spec[:, index_local_indices] -= 1.0
            out[part["observation_id"].to_numpy(dtype=np.int64), :] = spec
            sampled += int(len(part))
        print(f"{scene_id}: sampled={sampled}/{len(obs)} elapsed={time.perf_counter() - t0:.1f}s")

    if args.recompute_derived:
        print("Recomputing derived bands 165-168 from patch reflectance ...")
        recompute_indices(out, meta)
    out.flush()
    del out

    summary = {
        "input_dir": str(input_dir),
        "observation_table": str(obs_path),
        "source_spectra_path": str(source_spectra_path),
        "output_spectra_path": str(out_path),
        "shape": [int(source.shape[0]), int(source.shape[1])],
        "sampled_observations": int(sampled),
        "scene_count_sampled": int(scene_count),
        "patch_radius": int(args.patch_radius),
        "patch_size": int(2 * int(args.patch_radius) + 1),
        "patch_sample_count": int(len(offsets)),
        "gaussian_sigma": float(args.gaussian_sigma),
        "reducer": str(args.reducer),
        "offset_sampling": str(args.offset_sampling),
        "scene_offset_csv": str(args.scene_offset_csv),
        "scene_offset_count": int(len(scene_offsets)),
        "recompute_derived": bool(args.recompute_derived),
        "max_observations": int(args.max_observations),
        "copy_source_for_unsampled": bool(args.copy_source_for_unsampled),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    save_json(summary, input_dir / f"{Path(args.output_name).stem}_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
