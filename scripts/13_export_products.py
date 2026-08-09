

import sys
import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hyperspectral_pointcloud_fusion.common import load_config, ensure_dir, write_table, percentile_stretch
from hyperspectral_pointcloud_fusion.plyio import write_binary_spectra_ply


EXPORT_MINUS_ONE_BANDS = []


OPTIONAL_EXTRA_FLOAT_FIELDS = [

]

INDEX_BAND_NAME_MAP = {165: 'NDVI', 166: 'RVI', 167: 'MSAVI', 168: 'NDWI'}


RGB_PREFERRED_BANDS = (73, 51, 31)


RGB_SELECTED_PRODUCT_FALLBACK_BANDS = (76, 51, 26)


def band_property_name(band_no: int, rename_index_bands: bool = True) -> str:
    b = int(band_no)
    if rename_index_bands and b in INDEX_BAND_NAME_MAP:
        return INDEX_BAND_NAME_MAP[b]
    return f'band_{b}'


def _parse_rgb_band_triplet(value, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return tuple(map(int, default))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            'RGB band configuration must contain exactly three one-based band numbers '
            'in red, green, blue order.'
        )
    triplet = tuple(int(item) for item in value)
    if len(set(triplet)) != 3 or min(triplet) <= 0:
        raise ValueError(f'Invalid RGB band triplet: {triplet}')
    return triplet


def build_rgb_visualization(
    spectra: np.ndarray,
    band_numbers: np.ndarray,
    preferred_bands: tuple[int, int, int] = RGB_PREFERRED_BANDS,
    fallback_bands: tuple[int, int, int] = RGB_SELECTED_PRODUCT_FALLBACK_BANDS,
):


    band_to_idx = {int(b): i for i, b in enumerate(np.asarray(band_numbers).tolist())}
    preferred_bands = tuple(map(int, preferred_bands))
    fallback_bands = tuple(map(int, fallback_bands))
    if all(band in band_to_idx for band in preferred_bands):
        selected_bands = preferred_bands
        selection_mode = 'v1_exact_73_51_31'
    elif all(band in band_to_idx for band in fallback_bands):
        selected_bands = fallback_bands
        selection_mode = 'selected_product_fallback_76_51_26'
    else:
        return None, []

    n = spectra.shape[0]
    rgb = np.zeros((n, 3), dtype=np.uint8)
    logs = []
    channel_specs = [
        ('red', selected_bands[0], preferred_bands[0], 0),
        ('green', selected_bands[1], preferred_bands[1], 1),
        ('blue', selected_bands[2], preferred_bands[2], 2),
    ]

    for field_name, band_no, preferred_band_no, rgb_col in channel_specs:
        arr = np.asarray(spectra[:, band_to_idx[band_no]], dtype=np.float64) * 10000.0
        valid = np.isfinite(arr)
        stretched, lo, hi = percentile_stretch(arr, valid_mask=valid, q_low=0.0005, q_high=0.9995)
        rgb[:, rgb_col] = stretched
        logs.append({
            'output_field': field_name,
            'source_band_number_one_based': int(band_no),
            'preferred_v1_band_number_one_based': int(preferred_band_no),
            'selection_mode': selection_mode,
            'input_transform': 'reflectance * 10000',
            'stretch_min': float(lo),
            'stretch_max': float(hi),
        })
    return rgb, logs


def adjust_export_spectra(spectra: np.ndarray, band_numbers: np.ndarray):
    out = np.asarray(spectra, dtype=np.float32).copy()
    band_to_idx = {int(b): i for i, b in enumerate(np.asarray(band_numbers).tolist())}
    logs = []
    for band_no in EXPORT_MINUS_ONE_BANDS:
        if band_no in band_to_idx:
            idx = band_to_idx[band_no]
            out[:, idx] = out[:, idx] - 1.0
            logs.append({
                'band_number_one_based': int(band_no),
                'export_adjustment': 'minus_1',
                'dtype_after_adjustment': 'float32',
            })
    return out, logs


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def finalize_compact_output_layout(cfg: dict, final_ply: Path) -> dict:

    run_root = Path(cfg['project']['output_root']).resolve()
    brf_dir = Path(cfg['outputs']['directional_brf_dir']).resolve()
    selection_dir = Path(cfg['outputs']['selection_dir']).resolve()
    candidate_dir = Path(cfg['outputs']['candidate_dir']).resolve()
    brf_observations = brf_dir / 'point_directional_brf_observations.parquet'
    brf_summary_path = brf_dir / 'directional_brf_summary.json'
    selection_summary_path = selection_dir / 'selection_summary.json'
    projection_product = selection_dir / 'corrected_point_projection.parquet'
    removed: list[dict[str, object]] = []

    if not all(path.exists() for path in (brf_observations, brf_summary_path, selection_summary_path)):
        return {'status': 'skipped', 'reason': 'canonical BRF/projection inputs are incomplete'}
    brf_summary = json.loads(brf_summary_path.read_text(encoding='utf-8'))
    selection_summary = json.loads(selection_summary_path.read_text(encoding='utf-8'))
    parquet_rows = int(pq.ParquetFile(brf_observations).metadata.num_rows)
    expected_rows = int(selection_summary['emitted_visible_rows'])
    if parquet_rows != expected_rows or int(brf_summary['observation_count']) != expected_rows:
        raise RuntimeError(
            'Refusing projection cleanup because BRF Parquet row count does not match selection output: '
            f'parquet={parquet_rows}, selection={expected_rows}'
        )

    if projection_product.exists() and not os.path.samefile(projection_product, brf_observations):
        projection_product.unlink()
    if not projection_product.exists():
        os.link(brf_observations, projection_product)
    if not os.path.samefile(projection_product, brf_observations):
        raise RuntimeError('Corrected projection product is not linked to the validated BRF geometry table')

    redundant_projection_intermediates = (
        selection_dir / 'visible_observations.csv',
        selection_dir / 'point_visibility_summary.csv',
        selection_dir / 'visible_candidate_count.npy',
        selection_dir / 'clear_candidate_count.npy',
        selection_dir / 'kept_view_count.npy',
        selection_dir / 'used_fallback.npy',
    )
    for path in redundant_projection_intermediates:
        if path.exists() and _within(path, run_root):
            size = int(path.stat().st_size)
            path.unlink()
            removed.append({'path': str(path), 'bytes': size, 'reason': 'redundant after validated Parquet product'})

    redundant_fusion_intermediates = (
        Path(cfg['outputs']['fusion_dir']).resolve() / 'fused_point_summary.csv',
        Path(cfg['outputs']['fusion_dir']).resolve() / 'fused_observation_table.csv',
    )
    for path in redundant_fusion_intermediates:
        if path.exists() and _within(path, run_root):
            size = int(path.stat().st_size)
            path.unlink()
            removed.append({
                'path': str(path),
                'bytes': size,
                'reason': 'duplicate table; canonical fusion arrays and BRF observations are retained',
            })


    legacy_product_names = (
        'pointcloud_fused_preview.ply',
        'pointcloud_fused_preview_allbands.ply',
        'pointcloud_fused_selected_bands.ply',
        'pointcloud_fused_selected_bands_preview.ply',
        'pointcloud_fused_allbands.ply',
        'pointcloud_fused_preview.png',
        'pointcloud_fused_preview_allbands.png',
    )
    for name in legacy_product_names:
        path = run_root / name
        if path.exists() and path.resolve() != final_ply.resolve() and _within(path, run_root):
            size = int(path.stat().st_size)
            path.unlink()
            removed.append({'path': str(path), 'bytes': size, 'reason': 'obsolete preview or legacy fused-product name'})

    cache_value = selection_summary.get('candidate_cache_dir')
    if cache_value:
        cache_dir = Path(str(cache_value)).resolve()
        if (
            cache_dir.exists()
            and _within(cache_dir, candidate_dir)
            and cache_dir.name == 'point_topk_candidates_memmap'
        ):
            size = sum(path.stat().st_size for path in cache_dir.rglob('*') if path.is_file())
            shutil.rmtree(cache_dir)
            removed.append({'path': str(cache_dir), 'bytes': int(size), 'reason': 'candidate cache no longer needed'})

    manifest = {
        'status': 'complete',
        'run_root': str(run_root),
        'band_mode': cfg['project']['band_mode'],
        'pointcloud_stem': cfg['project']['pointcloud_stem'],
        'products': {
            'fused_pointcloud': str(final_ply),
            'fusion_arrays': str(Path(cfg['outputs']['fusion_dir'])),
            'brf_reconstruction': str(brf_dir),
            'corrected_projection': str(projection_product),
            'manual_annotations': str(Path(cfg['target_calibration']['annotation_csv'])),
            'scene_correction_csv': str(Path(cfg['target_calibration']['correction_log_csv'])),
            'plot_analysis_root': str(Path(cfg['project']['analysis_root'])),
            'figure_root': str(Path(cfg['project']['figure_root'])),
        },
        'corrected_projection_rows': parquet_rows,
        'removed_redundant_intermediates': removed,
    }
    (run_root / 'output_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['outputs']['product_dir'])
    point_dir = Path(cfg['outputs']['pointcloud_dir'])
    fusion_dir = ensure_dir(cfg['outputs']['fusion_dir'])
    pointcloud_stem = str(cfg['project']['pointcloud_stem'])
    all_bands = bool(cfg['project'].get('all_bands', False))
    final_name = f'hpf_{pointcloud_stem}_allbands.ply' if all_bands else f'hpf_{pointcloud_stem}.ply'
    final_path = out_dir / final_name

    xyz = np.load(point_dir / 'point_xyz.npy', mmap_mode='r')
    product_cfg = dict(cfg.get('products', {}) or {})
    spectra_name = str(product_cfg.get('fused_spectra_name', 'fused_point_spectra.npy')).strip() or 'fused_point_spectra.npy'
    spectra_path = fusion_dir / spectra_name
    if not spectra_path.exists() and spectra_name != 'fused_point_spectra.npy':
        print(f'Configured fused_spectra_name={spectra_name} was not found; falling back to fused_point_spectra.npy.')
        spectra_path = fusion_dir / 'fused_point_spectra.npy'
    spectra_raw = np.load(spectra_path, mmap_mode='r')
    band_numbers = np.load(fusion_dir / 'fused_band_numbers.npy', mmap_mode='r')

    primary_scene_code = np.load(fusion_dir / 'primary_scene_code.npy', mmap_mode='r')
    contributing_view_count = np.load(fusion_dir / 'contributing_view_count.npy', mmap_mode='r')
    fused_weight_sum = np.load(fusion_dir / 'fused_weight_sum.npy', mmap_mode='r')
    primary_weight_ratio = np.load(fusion_dir / 'primary_weight_ratio.npy', mmap_mode='r')
    clear_view_ratio = np.load(fusion_dir / 'clear_view_ratio.npy', mmap_mode='r')
    weighted_mean_range = np.load(fusion_dir / 'weighted_mean_range_m.npy', mmap_mode='r')
    weighted_mean_offaxis = np.load(fusion_dir / 'weighted_mean_offaxis_deg.npy', mmap_mode='r')
    weighted_mean_border = np.load(fusion_dir / 'weighted_mean_border_dist_px.npy', mmap_mode='r')
    weighted_mean_empty_cone = np.load(fusion_dir / 'weighted_mean_empty_cone_deg.npy', mmap_mode='r')
    weighted_mean_view_zenith = np.load(fusion_dir / 'weighted_mean_view_zenith_deg.npy', mmap_mode='r')
    weighted_mean_surface_view_cos = np.load(fusion_dir / 'weighted_mean_surface_view_cos.npy', mmap_mode='r')
    weighted_mean_surface_verticality = np.load(fusion_dir / 'weighted_mean_surface_verticality.npy', mmap_mode='r')
    view_weight_entropy = np.load(fusion_dir / 'view_weight_entropy.npy', mmap_mode='r')
    effective_view_count = np.load(fusion_dir / 'effective_view_count.npy', mmap_mode='r')

    spectra_export, adjustment_logs = adjust_export_spectra(
        spectra=np.asarray(spectra_raw),
        band_numbers=np.asarray(band_numbers),
    )

    preferred_rgb_bands = _parse_rgb_band_triplet(
        product_cfg.get('rgb_preferred_bands_one_based'), RGB_PREFERRED_BANDS,
    )
    fallback_rgb_bands = _parse_rgb_band_triplet(
        product_cfg.get('rgb_fallback_bands_one_based'), RGB_SELECTED_PRODUCT_FALLBACK_BANDS,
    )
    rgb, rgb_logs = build_rgb_visualization(
        spectra=np.asarray(spectra_raw),
        band_numbers=np.asarray(band_numbers),
        preferred_bands=preferred_rgb_bands,
        fallback_bands=fallback_rgb_bands,
    )

    rename_index_bands = bool(product_cfg.get('rename_index_bands', True))
    minimal_extra_fields = bool(product_cfg.get('minimal_extra_fields', True))

    full_extra = {
        'primary_scene_code': np.asarray(primary_scene_code, dtype=np.int16),
        'contributing_view_count': np.asarray(contributing_view_count, dtype=np.int16),
        'fused_weight_sum': np.asarray(fused_weight_sum, dtype=np.float32),
        'primary_weight_ratio': np.asarray(primary_weight_ratio, dtype=np.float32),
        'clear_view_ratio': np.asarray(clear_view_ratio, dtype=np.float32),
        'weighted_mean_range_m': np.asarray(weighted_mean_range, dtype=np.float32),
        'weighted_mean_offaxis_deg': np.asarray(weighted_mean_offaxis, dtype=np.float32),
        'weighted_mean_border_px': np.asarray(weighted_mean_border, dtype=np.float32),
        'weighted_mean_empty_cone_deg': np.asarray(weighted_mean_empty_cone, dtype=np.float32),
        'weighted_mean_view_zenith_deg': np.asarray(weighted_mean_view_zenith, dtype=np.float32),
        'weighted_mean_surface_view_cos': np.asarray(weighted_mean_surface_view_cos, dtype=np.float32),
        'weighted_mean_surface_verticality': np.asarray(weighted_mean_surface_verticality, dtype=np.float32),
        'view_weight_entropy': np.asarray(view_weight_entropy, dtype=np.float32),
        'effective_view_count': np.asarray(effective_view_count, dtype=np.float32),
    }
    if minimal_extra_fields:
        keep = product_cfg.get('extra_fields_to_keep') or [
            'contributing_view_count', 'effective_view_count', 'primary_weight_ratio',
            'clear_view_ratio', 'weighted_mean_view_zenith_deg', 'weighted_mean_surface_view_cos'
        ]
        extra = {k: v for k, v in full_extra.items() if k in set(keep)}
    else:
        extra = full_extra
    for field_name in OPTIONAL_EXTRA_FLOAT_FIELDS:
        path = fusion_dir / f'{field_name}.npy'
        if path.exists():
            extra[field_name] = np.asarray(np.load(path, mmap_mode='r'), dtype=np.float32)

    band_names = [band_property_name(int(b), rename_index_bands=rename_index_bands) for b in np.asarray(band_numbers)]
    write_binary_spectra_ply(
        str(final_path),
        np.asarray(xyz),
        np.asarray(spectra_export, dtype=np.float32),
        band_names,
        rgb=rgb,
        extra=extra,
    )

    if rgb is not None:
        used_rgb_bands = '/'.join(
            str(int(row['source_band_number_one_based'])) for row in rgb_logs
        )
        print(
            f'RGB visualization fields generated from R/G/B bands {used_rgb_bands} '
            'with the v1 percentile stretch.'
        )
    else:
        print(
            'Neither preferred RGB bands 73/51/31 nor configured selected-product '
            'fallback RGB bands are complete; RGB visualization fields were skipped.'
        )

    mapping_rows = []
    adjusted_band_set = set(EXPORT_MINUS_ONE_BANDS)
    for i, band_no in enumerate(np.asarray(band_numbers).tolist()):
        mapping_rows.append({
            'local_index': int(i),
            'band_number_one_based': int(band_no),
            'property_name': band_property_name(int(band_no), rename_index_bands=rename_index_bands),
            'export_adjustment': 'minus_1' if int(band_no) in adjusted_band_set else 'none',
        })
    write_table(pd.DataFrame(mapping_rows), fusion_dir / 'selected_band_mapping.csv', index=False)

    if rgb_logs:
        write_table(pd.DataFrame(rgb_logs), fusion_dir / 'rgb_stretch_log.csv', index=False)
    if adjustment_logs:
        write_table(pd.DataFrame(adjustment_logs), fusion_dir / 'band_export_adjustment_log.csv', index=False)

    valid = np.isfinite(np.asarray(contributing_view_count, dtype=np.float32)) & (np.asarray(contributing_view_count, dtype=np.int16) > 0)
    quality_summary = pd.DataFrame({
        'metric': [
            'fused_spectra_source',
            'valid_point_count',
            'valid_ratio',
            'mean_contributing_view_count',
            'mean_primary_weight_ratio',
            'mean_clear_view_ratio',
            'mean_weighted_range_m',
            'mean_weighted_offaxis_deg',
            'mean_weighted_view_zenith_deg',
            'mean_weighted_surface_view_cos',
            'mean_weighted_surface_verticality',
            'mean_effective_view_count',
            'has_rgb_visualization_fields',
        ],
        'value': [
            str(spectra_path),
            int(np.count_nonzero(valid)),
            float(np.mean(valid)) if valid.size else 0.0,
            float(np.nanmean(np.asarray(contributing_view_count)[valid])) if np.any(valid) else np.nan,
            float(np.nanmean(np.asarray(primary_weight_ratio)[valid])) if np.any(valid) else np.nan,
            float(np.nanmean(np.asarray(clear_view_ratio)[valid])) if np.any(valid) else np.nan,
            float(np.nanmean(np.asarray(weighted_mean_range)[valid])) if np.any(valid) else np.nan,
            float(np.nanmean(np.asarray(weighted_mean_offaxis)[valid])) if np.any(valid) else np.nan,
            float(np.nanmean(np.asarray(weighted_mean_view_zenith)[valid])) if np.any(valid) else np.nan,
            float(np.nanmean(np.asarray(weighted_mean_surface_view_cos)[valid])) if np.any(valid) else np.nan,
            float(np.nanmean(np.asarray(weighted_mean_surface_verticality)[valid])) if np.any(valid) else np.nan,
            float(np.nanmean(np.asarray(effective_view_count)[valid])) if np.any(valid) else np.nan,
            int(rgb is not None),
        ],
    })
    write_table(quality_summary, fusion_dir / 'fusion_quality_summary.csv', index=False)

    compact_manifest = finalize_compact_output_layout(cfg, final_path)
    print(json.dumps(compact_manifest, ensure_ascii=False, indent=2))

    print(f'Output: {final_path}')


if __name__ == '__main__':
    main()
