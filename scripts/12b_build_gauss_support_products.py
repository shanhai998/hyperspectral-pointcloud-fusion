from __future__ import annotations


import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = DEFAULT_PROJECT_ROOT / "config" / "project_config.yaml"
SIGMA_PX = 8.0
TRUNCATE = 3.0
MEDIAN_GSD_M_PER_PX = 0.06125232
GAUSSIAN_FWHM_FACTOR = 2.3548200450309493
INDEX_BAND_NAMES = {165: "NDVI", 166: "RVI", 167: "MSAVI", 168: "NDWI"}


def gaussian_support_metadata() -> dict[str, float | str]:
    sigma_m = SIGMA_PX * MEDIAN_GSD_M_PER_PX
    return {
        "kernel": "Gaussian",
        "sigma_px": SIGMA_PX,
        "truncate": TRUNCATE,
        "edge_mode": "nearest",
        "median_gsd_m_per_px": MEDIAN_GSD_M_PER_PX,
        "median_sigma_m": sigma_m,
        "median_fwhm_m": sigma_m * GAUSSIAN_FWHM_FACTOR,
    }


def load_project_modules(project_root: Path):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from hyperspectral_pointcloud_fusion.common import load_config
    from hyperspectral_pointcloud_fusion.envi import read_envi_bsq_memmap

    fusion_path = project_root / "scripts" / "11_extract_and_fuse_spectra.py"
    spec = importlib.util.spec_from_file_location(
        "hyperspectral_pointcloud_fusion_existing_fusion", fusion_path,
    )
    fusion = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fusion)
    return load_config, read_envi_bsq_memmap, fusion


def bilinear_sample_plane(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = plane.shape
    xf = np.floor(x).astype(np.int64)
    yf = np.floor(y).astype(np.int64)
    x0 = np.clip(xf, 0, width - 1)
    y0 = np.clip(yf, 0, height - 1)
    x1 = np.clip(xf + 1, 0, width - 1)
    y1 = np.clip(yf + 1, 0, height - 1)
    wx = x - x0
    wy = y - y0
    v0 = (1.0 - wx) * plane[y0, x0] + wx * plane[y0, x1]
    v1 = (1.0 - wx) * plane[y1, x0] + wx * plane[y1, x1]
    return (1.0 - wy) * v0 + wy * v1


def extract_all_views(
    config: dict,
    staging: Path,
    read_envi_bsq_memmap,
    band_indices: list[int],
    band_numbers: list[int],
) -> dict[str, object]:
    observation_path = Path(config["outputs"]["directional_brf_dir"]) / "point_directional_brf_observations.parquet"
    scene_db = pd.read_csv(Path(config["outputs"]["scene_db_dir"]) / "scene_database.csv").set_index("scene_id")
    total_rows = int(pq.ParquetFile(observation_path).metadata.num_rows)
    band_count = len(band_indices)
    spectra_path = staging / "directional_brf_spectra.npy"
    filled_path = staging / "directional_brf_spectra_filled.npy"
    spectra = np.lib.format.open_memmap(spectra_path, mode="w+", dtype=np.float32, shape=(total_rows, band_count))
    spectra[:] = np.nan
    filled = np.lib.format.open_memmap(filled_path, mode="w+", dtype=np.uint8, shape=(total_rows,))
    filled[:] = 0
    scene_rows: list[dict[str, object]] = []
    scenes = scene_db.sort_index()
    for scene_index, (scene_id, row) in enumerate(scenes.iterrows(), 1):
        observations = pd.read_parquet(
            observation_path,
            columns=["observation_id", "scene_id", "u", "v"],
            filters=[("scene_id", "==", str(scene_id))],
        )
        ids = observations["observation_id"].to_numpy(np.int64)
        if len(ids) == 0:
            scene_rows.append({"scene_id": str(scene_id), "rows": 0})
            continue
        cube = read_envi_bsq_memmap(
            str(row["data_path"]), int(row["samples"]), int(row["lines"]), int(row["bands"]),
            int(row["data_type"]), int(row["byte_order"]), int(row["header_offset"]),
        )
        u = observations["u"].to_numpy(np.float64)
        v = observations["v"].to_numpy(np.float64)
        scale = float(row["reflectance_scale_factor"])
        plus_one_bands = set(map(int, config.get("processing", {}).get("index_band_numbers_stored_plus_one", [])))
        for column, (band_index, band_number) in enumerate(zip(band_indices, band_numbers)):
            filtered = gaussian_filter(
                np.asarray(cube[band_index], dtype=np.float32),
                sigma=SIGMA_PX,
                mode="nearest",
                truncate=TRUNCATE,
                output=np.float32,
            )
            values = bilinear_sample_plane(filtered, u, v).astype(np.float32)
            if scale not in (0.0, 1.0):
                values /= scale
            if int(band_number) in plus_one_bands:
                values -= 1.0
            spectra[ids, column] = values
        filled[ids] = 1
        scene_rows.append({
            "scene_id": str(scene_id),
            "rows": int(len(ids)),
            "minimum_observation_id": int(ids.min()),
            "maximum_observation_id": int(ids.max()),
        })
        spectra.flush()
        filled.flush()
        del cube
        print(f"source extraction {scene_index}/{len(scenes)} {scene_id}: rows={len(ids)}", flush=True)
    missing = int(np.count_nonzero(np.asarray(filled) == 0))
    finite_rows = int(np.count_nonzero(np.isfinite(np.asarray(spectra)).all(axis=1)))
    spectra.flush()
    filled.flush()
    del spectra, filled
    if missing or finite_rows != total_rows:
        raise RuntimeError(
            f"Gaussian-support extraction incomplete: missing={missing}, finite={finite_rows}/{total_rows}"
        )
    pd.DataFrame(scene_rows).to_csv(staging / "source_scene_rows.csv", index=False)
    return {
        "rows": total_rows,
        "finite_rows": finite_rows,
        "missing_rows": missing,
        "scene_count": int(len(scenes)),
        "bands": band_numbers,
        "source_band_indices_zero_based": band_indices,
        "kernel": {"type": "Gaussian", "sigma_px": SIGMA_PX, "truncate": TRUNCATE, "edge_mode": "nearest"},
    }


def rebuild_weighted(config: dict, staging: Path, fusion, band_count: int) -> dict[str, object]:
    processing = dict(config.get("processing", {}))
    mode = str(processing.get("fusion_weight_mode", "score_softmax")).strip().lower()
    temperature = float(processing.get("fusion_softmax_temperature", 30.0))
    raw_power = float(processing.get("fusion_raw_weight_power", 1.0))
    visible_path = Path(config["outputs"]["selection_dir"]) / "visible_observations.csv"
    old_spectra = np.load(Path(config["outputs"]["directional_brf_dir"]) / "directional_brf_spectra.npy", mmap_mode="r")
    new_spectra = np.load(staging / "directional_brf_spectra.npy", mmap_mode="r")
    fusion_dir = Path(config["outputs"]["fusion_dir"])
    existing_fused = np.load(fusion_dir / "fused_point_spectra.npy", mmap_mode="r")
    point_count = int(existing_fused.shape[0])
    if int(existing_fused.shape[1]) != int(band_count):
        raise RuntimeError("Configured Gaussian-support band count does not match fused_point_spectra.npy")
    old_sum = np.zeros((point_count, band_count), dtype=np.float64)
    new_sum = np.zeros((point_count, band_count), dtype=np.float64)
    weight_sum = np.zeros(point_count, dtype=np.float64)
    selected_rows = 0
    source_rows = 0
    carry = None
    usecols = [
        "point_id", "visibility_score", "view_weight_raw", "range_m", "offaxis_deg",
        "local_empty_cone_deg", "blocker_count", "is_clear", "normal_confidence",
        "surface_view_cos", "surface_verticality", "view_zenith_deg", "keep_for_fusion",
    ]

    def process(frame: pd.DataFrame | None) -> None:
        nonlocal selected_rows
        if frame is None or len(frame) == 0:
            return
        frame = fusion.add_weight_columns(frame, processing, mode, temperature, raw_power)
        if len(frame) == 0:
            return
        ids = frame["observation_id"].to_numpy(np.int64)
        point_ids = frame["point_id"].to_numpy(np.int64)
        weights = frame["view_weight_norm"].to_numpy(np.float64)
        old = np.asarray(old_spectra[ids], dtype=np.float64)
        new = np.asarray(new_spectra[ids], dtype=np.float64)
        for band in range(band_count):
            np.add.at(old_sum[:, band], point_ids, weights * old[:, band])
            np.add.at(new_sum[:, band], point_ids, weights * new[:, band])
        np.add.at(weight_sum, point_ids, weights)
        selected_rows += len(frame)

    reader = pd.read_csv(visible_path, usecols=usecols, chunksize=250000, encoding="utf-8-sig")
    for chunk_index, chunk in enumerate(reader, 1):
        chunk.insert(0, "observation_id", np.arange(source_rows, source_rows + len(chunk), dtype=np.int64))
        source_rows += len(chunk)
        ready, carry = fusion.split_complete_points(fusion.coerce_visible_chunk(chunk), carry)
        process(ready)
        print(f"fusion chunk {chunk_index}: source={source_rows}, selected={selected_rows}", flush=True)
    process(carry)
    valid = weight_sum > 0
    old_rebuilt = np.full_like(old_sum, np.nan, dtype=np.float32)
    new_fused = np.full_like(new_sum, np.nan, dtype=np.float32)
    old_rebuilt[valid] = (old_sum[valid] / weight_sum[valid, None]).astype(np.float32)
    new_fused[valid] = (new_sum[valid] / weight_sum[valid, None]).astype(np.float32)
    candidate_path = staging / "fused_point_spectra.npy"
    np.save(candidate_path, new_fused)
    existing_weights = np.load(fusion_dir / "fused_weight_sum.npy")
    old_difference = np.abs(old_rebuilt.astype(np.float64) - np.asarray(existing_fused, dtype=np.float64))
    weight_difference = np.abs(weight_sum - existing_weights.astype(np.float64))
    summary = {
        "source_rows": int(source_rows),
        "selected_rows": int(selected_rows),
        "valid_points": int(valid.sum()),
        "fusion_weight_mode": mode,
        "old_fusion_reproduction_max_abs_difference": float(np.nanmax(old_difference)),
        "old_fusion_reproduction_mean_abs_difference": float(np.nanmean(old_difference)),
        "weight_sum_max_abs_difference": float(np.nanmax(weight_difference)),
        "weight_sum_mean_abs_difference": float(np.nanmean(weight_difference)),
    }
    if source_rows != old_spectra.shape[0] or selected_rows <= 0:
        raise RuntimeError("Visible-row/observation_id alignment failed")
    if summary["old_fusion_reproduction_max_abs_difference"] > 2e-6:
        raise RuntimeError("The existing weighted-fusion pipeline could not be reproduced")
    if not np.isfinite(new_fused).all():
        raise RuntimeError("The weighted Gaussian-support candidate contains non-finite values")
    del old_spectra, new_spectra, existing_fused, existing_weights
    gc.collect()
    return summary


def update_metadata(config: dict, extraction: dict[str, object], fusion_check: dict[str, object]) -> None:
    result_root = Path(config["project"]["output_root"])
    all_view = Path(config["outputs"]["directional_brf_dir"])
    weighted = Path(config["outputs"]["fusion_dir"])
    spatial_support = gaussian_support_metadata()
    directional_path = all_view / "directional_brf_summary.json"
    directional = json.loads(directional_path.read_text(encoding="utf-8"))
    directional.update({
        "method_version": "gauss_support_selected_final",
        "spatial_support": spatial_support,
        "formula": (
            "BRF_gauss_support[p,k,lambda] = "
            f"bilinear(Gaussian_sigma{SIGMA_PX:g}px(REF_k[lambda]), u[p,k], v[p,k])"
        ),
    })
    directional_path.write_text(json.dumps(directional, ensure_ascii=False, indent=2), encoding="utf-8")
    fusion_path = weighted / "fusion_summary.json"
    fusion_meta = json.loads(fusion_path.read_text(encoding="utf-8"))
    fusion_meta.update({
        "method_version": "gauss_support_selected_final",
        "source_spatial_support": extraction["kernel"],
        "source_median_fwhm_m": spatial_support["median_fwhm_m"],
        "fusion_rebuild_audit": fusion_check,
    })
    fusion_path.write_text(json.dumps(fusion_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    band_path = all_view / "directional_brf_band_metadata.csv"
    band_meta = pd.read_csv(band_path)
    band_meta["source_support_kernel"] = "Gaussian"
    band_meta["source_support_sigma_px"] = SIGMA_PX
    band_meta["source_support_fwhm_m_median"] = spatial_support["median_fwhm_m"]
    band_meta.to_csv(band_path, index=False, encoding="utf-8-sig")
    manifest = {
        "method_version": "gauss_support_selected_final",
        "bands": extraction["bands"],
        "variables": [INDEX_BAND_NAMES.get(int(b), f"band_{int(b)}") for b in extraction["bands"]],
        "full_spectrum_fusion_used": bool(config["project"].get("all_bands", False)),
        "products": {
            "weighted_fusion": {
                "spectra": str((weighted / "fused_point_spectra.npy").relative_to(result_root)).replace("\\", "/"),
                "shape": [int(fusion_check["valid_points"]), len(extraction["bands"])],
                "weighting": "original surface-adaptive multi-view weights",
            },
            "all_views": {
                "observations": str((all_view / "point_directional_brf_observations.parquet").relative_to(result_root)).replace("\\", "/"),
                "spectra": str((all_view / "directional_brf_spectra.npy").relative_to(result_root)).replace("\\", "/"),
                "shape": [int(extraction["rows"]), len(extraction["bands"])],
                "maximum_views": 49,
            },
        },
        "source_spatial_support": directional["spatial_support"],
        "all_view_product": "retains every valid point-view observation without weighted averaging",
    }
    (result_root / "product_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the selected Gaussian-support all-view and weighted products."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--result-root", type=Path, default=None,
        help="Override project.output_root from the configuration file.",
    )
    parser.add_argument("--skip-ply", action="store_true", help="Do not refresh the derived fused PLY export.")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    load_config, read_envi_bsq_memmap, fusion = load_project_modules(project_root)
    config_path = args.config.resolve()
    config = load_config(config_path)
    result_root = (
        args.result_root.resolve()
        if args.result_root is not None
        else Path(config["project"]["output_root"]).resolve()
    )
    fusion_dir = Path(config["outputs"]["fusion_dir"])
    all_view_dir = Path(config["outputs"]["directional_brf_dir"])
    band_indices = np.load(fusion_dir / "fused_band_indices.npy").astype(int).tolist()
    band_numbers = np.load(fusion_dir / "fused_band_numbers.npy").astype(int).tolist()
    if len(band_indices) != len(band_numbers) or not band_indices:
        raise RuntimeError("Invalid configured fused-band metadata")
    staging = fusion_dir / ".gauss_support_build"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    extraction = extract_all_views(
        config, staging, read_envi_bsq_memmap, band_indices, band_numbers,
    )
    fusion_check = rebuild_weighted(config, staging, fusion, len(band_numbers))
    all_target = all_view_dir / "directional_brf_spectra.npy"
    weighted_target = fusion_dir / "fused_point_spectra.npy"
    os.replace(staging / "directional_brf_spectra.npy", all_target)
    os.replace(staging / "fused_point_spectra.npy", weighted_target)
    for stale_name in ["point_brf_summary.parquet", "point_brf_summary_preview.csv", "angular_bin_statistics.csv"]:
        stale = all_target.parent / stale_name
        if stale.exists():
            stale.unlink()
    update_metadata(config, extraction, fusion_check)
    build_summary = {"extraction": extraction, "weighted_fusion": fusion_check}
    (fusion_dir / "gauss_support_build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    shutil.rmtree(staging)
    if not args.skip_ply:
        subprocess.run(
            [sys.executable, str(project_root / "scripts" / "13_export_products.py"),
             "--config", str(config_path)],
            check=True,
        )
    print(json.dumps(build_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
