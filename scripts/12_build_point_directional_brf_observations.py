

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import (
    append_table,
    ensure_dir,
    get_band_selection,
    is_quiet,
    load_config,
    load_json,
    read_table,
    save_json,
    write_table,
)
from hyperspectral_pointcloud_fusion.envi import (
    bilinear_sample_selected_bands_bsq,
    parse_envi_hdr,
    read_envi_bsq_memmap,
    wavelength_list_from_hdr,
)


DEFAULT_INDEX_PLUS_ONE_BANDS = [165, 167, 168]
INDEX_BAND_NAME_MAP = {165: 'NDVI', 166: 'RVI', 167: 'MSAVI', 168: 'NDWI'}
_CUBE_CACHE: dict[str, np.memmap] = {}
_CUBE_CACHE_LOCK = Lock()


def to_bool_series(s: pd.Series, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default).astype(bool)
    text = s.fillna(str(default)).astype(str).str.strip().str.lower()
    return text.isin(['true', '1', 'yes', 'y'])


def _count_csv_rows(path: Path) -> int:
    with open(path, 'rb') as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def _infer_visible_row_count(cfg: dict, vis_path: Path, allow_slow_count: bool = True) -> tuple[int, str]:
    candidates: list[tuple[int, str]] = []

    def add_candidate(val: Any, source: str) -> None:
        if val is not None and int(val) > 0:
            candidates.append((int(val), source))

    sel_summary = Path(cfg['outputs']['selection_dir']) / 'selection_summary.json'
    if sel_summary.exists():
        try:
            data = load_json(sel_summary)
            for key in ['emitted_visible_rows', 'visible_observation_rows', 'kept_visible_rows']:
                add_candidate(data.get(key), f'selection_summary.json:{key}')
        except Exception:
            pass

    fusion_summary = Path(cfg['outputs'].get('fusion_dir', '')) / 'fusion_summary.json'
    if fusion_summary.exists():
        try:
            data = load_json(fusion_summary)
            add_candidate(data.get('visible_observation_rows'), 'fusion_summary.json:visible_observation_rows')
        except Exception:
            pass

    dcfg = dict(cfg.get('directional_brf', {}) or {})
    add_candidate(dcfg.get('expected_observation_rows'), 'directional_brf.expected_observation_rows')

    if allow_slow_count:
        add_candidate(_count_csv_rows(vis_path), 'slow_csv_line_count')

    if candidates:
        count, source = max(candidates, key=lambda item: item[0])
        if len(candidates) > 1:
            detail = ', '.join(f'{name}={value}' for value, name in candidates)
            source = f'{source}; max_of({detail})'
        return count, source
    return 0, 'unknown'


def _visible_dtype_map(cols: list[str]) -> dict[str, Any]:
    dtype_map = {
        'point_id': 'Int64',
        'scene_code': 'int16',
        'scene_id': str,
        'u': 'float32',
        'v': 'float32',
        'range_m': 'float32',
        'offaxis_deg': 'float32',
        'border_dist_px': 'float32',
        'view_zenith_deg': 'float32',
        'view_azimuth_deg': 'float32',
        'local_view_cos_signed': 'float32',
        'local_view_angle_deg': 'float32',
        'solar_zenith_deg': 'float32',
        'solar_azimuth_deg': 'float32',
        'relative_azimuth_deg': 'float32',
        'local_solar_cos': 'float32',
        'local_solar_incidence_deg': 'float32',
        'surface_view_cos': 'float32',
        'surface_verticality': 'float32',
        'normal_confidence': 'float32',
        'visibility_score': 'float32',
        'view_weight_raw': 'float32',
        'local_empty_cone_deg': 'float32',
        'blocker_count': 'int16',
        'image_zbuffer_clear': bool,
        'is_clear': bool,
        'fusion_rank_within_point': 'int16',
    }
    return {k: v for k, v in dtype_map.items() if k in cols}


def _band_output_name(band_number: int, rename_index_bands: bool = True) -> str:
    b = int(band_number)
    if rename_index_bands and b in INDEX_BAND_NAME_MAP:
        return INDEX_BAND_NAME_MAP[b]
    return f'band_{b}'


def _select_directional_bands(cfg: dict, scene_row: pd.Series) -> tuple[list[int], list[int], list[float], list[str]]:
    hdr = parse_envi_hdr(scene_row['hdr_path'])
    wavelengths = wavelength_list_from_hdr(hdr)
    total_bands = int(scene_row['bands'])

    dcfg = dict(cfg.get('directional_brf', {}) or {})
    include_all_selected = bool(dcfg.get('include_all_selected_bands', True))
    bands_for_table = [int(x) for x in (dcfg.get('bands_for_table', []) or [])]
    rename_index_bands = bool(dcfg.get('rename_index_bands', True))

    if include_all_selected:
        band_indices, band_info = get_band_selection(cfg, total_bands, wavelengths)
        band_numbers = list(map(int, band_info['band_numbers_one_based']))
    else:
        band_numbers = []
        band_indices = []

    for b in bands_for_table:
        if 1 <= b <= total_bands and b not in band_numbers:
            band_numbers.append(int(b))
            band_indices.append(int(b) - 1)

    pairs = sorted(zip(band_numbers, band_indices), key=lambda x: x[0])
    band_numbers = [p[0] for p in pairs]
    band_indices = [p[1] for p in pairs]
    wavelengths_used = [float(wavelengths[i]) if wavelengths and i < len(wavelengths) else np.nan for i in band_indices]
    band_names = [_band_output_name(b, rename_index_bands=rename_index_bands) for b in band_numbers]
    return band_indices, band_numbers, wavelengths_used, band_names


def _open_cube(scene_row: dict):
    return read_envi_bsq_memmap(
        scene_row['data_path'],
        samples=int(scene_row['samples']),
        lines=int(scene_row['lines']),
        bands=int(scene_row['bands']),
        data_type=int(scene_row['data_type']),
        byte_order=int(scene_row['byte_order']),
        header_offset=int(scene_row['header_offset']),
    )


def _get_cube(scene_row: dict, enable_cache: bool = True):
    if not enable_cache:
        return _open_cube(scene_row)
    key = str(scene_row['data_path'])
    with _CUBE_CACHE_LOCK:
        cube = _CUBE_CACHE.get(key)
        if cube is None:
            cube = _open_cube(scene_row)
            _CUBE_CACHE[key] = cube
        return cube


def _sample_one_scene(
    scene_id: str,
    group: pd.DataFrame,
    scene_row: dict,
    band_indices: list[int],
    index_local_indices: list[int],
    enable_memmap_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cube = _get_cube(scene_row, enable_cache=enable_memmap_cache)
    spectra = bilinear_sample_selected_bands_bsq(
        cube,
        x=group['u'].astype(float).values,
        y=group['v'].astype(float).values,
        band_indices=band_indices,
        scale_factor=float(scene_row['reflectance_scale_factor']),
    ).astype(np.float32, copy=False)
    if index_local_indices:
        spectra[:, index_local_indices] -= 1.0
    return group.index.to_numpy(dtype=np.int64), spectra


def _import_pyarrow_or_raise():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except Exception as e:
        raise RuntimeError(
            'Parquet output requires pyarrow. Install it with: pip install pyarrow\n'
            f'Original error: {e}'
        )


def _init_parquet_writer(path: Path, df: pd.DataFrame, pa=None, pq=None):
    if pa is None or pq is None:
        pa, pq = _import_pyarrow_or_raise()
    table = pa.Table.from_pandas(df, preserve_index=False)
    writer = pq.ParquetWriter(str(path), table.schema)
    writer.write_table(table)
    return writer, pa, pq


def _classify_observation_quality(df: pd.DataFrame, qcfg: dict) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.asarray([], dtype=object)
    is_clear = to_bool_series(df.get('is_clear', pd.Series([False] * n)), default=False).to_numpy()
    selection_method = df.get('selection_method', pd.Series([''] * n)).astype(str).str.lower().to_numpy()
    is_fallback = np.char.find(selection_method.astype(str), 'fallback') >= 0

    offaxis = pd.to_numeric(df.get('offaxis_deg', np.nan), errors='coerce').to_numpy(dtype=np.float64)
    empty_cone = pd.to_numeric(df.get('local_empty_cone_deg', np.nan), errors='coerce').to_numpy(dtype=np.float64)
    border = pd.to_numeric(df.get('border_dist_px', np.nan), errors='coerce').to_numpy(dtype=np.float64)
    vz = pd.to_numeric(df.get('view_zenith_deg', np.nan), errors='coerce').to_numpy(dtype=np.float64)
    blocker = pd.to_numeric(df.get('blocker_count', np.nan), errors='coerce').to_numpy(dtype=np.float64)

    high = (
        is_clear & (~is_fallback)
        & (offaxis <= float(qcfg.get('high_offaxis_max_deg', 20.0)))
        & (empty_cone >= float(qcfg.get('high_empty_cone_min_deg', 15.0)))
        & (border >= float(qcfg.get('high_border_min_px', 80.0)))
        & (vz <= float(qcfg.get('high_view_zenith_max_deg', 60.0)))
        & (blocker <= 0)
    )
    medium = (
        is_clear & (~high) & (~is_fallback)
        & (offaxis <= float(qcfg.get('medium_offaxis_max_deg', 35.0)))
        & (empty_cone >= float(qcfg.get('medium_empty_cone_min_deg', 8.0)))
        & (border >= float(qcfg.get('medium_border_min_px', 30.0)))
        & (vz <= float(qcfg.get('medium_view_zenith_max_deg', 70.0)))
        & (blocker <= 0)
    )
    out = np.full(n, 'low', dtype=object)
    out[medium] = 'medium'
    out[high] = 'high'
    return out


def _paper_observation_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        'observation_id', 'point_id', 'scene_id', 'scene_code',
        'x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z',
        'u', 'v', 'range_m', 'offaxis_deg', 'border_dist_px',
        'view_zenith_deg', 'view_azimuth_deg',
        'solar_zenith_deg', 'solar_azimuth_deg', 'relative_azimuth_deg',
        'local_view_angle_deg', 'local_solar_incidence_deg', 'is_backlit',
        'surface_view_cos', 'surface_verticality', 'normal_confidence',
        'local_empty_cone_deg', 'blocker_count', 'image_zbuffer_clear',
        'is_clear', 'selection_method', 'fusion_rank_within_point',
        'view_weight_raw', 'brf_quality_level',
    ]
    return [c for c in preferred if c in df.columns]


def _init_summary_accumulators(point_count: int, variable_names: list[str]) -> dict[str, Any]:
    n = int(point_count)
    m = len(variable_names)
    return {
        'obs_count': np.zeros(n, dtype=np.int32),
        'clear_count': np.zeros(n, dtype=np.int32),
        'high_count': np.zeros(n, dtype=np.int32),
        'medium_count': np.zeros(n, dtype=np.int32),
        'low_count': np.zeros(n, dtype=np.int32),
        'weight_sum': np.zeros(n, dtype=np.float64),
        'weight_sq_sum': np.zeros(n, dtype=np.float64),
        'weight_max': np.zeros(n, dtype=np.float64),
        'view_min': np.full(n, np.inf, dtype=np.float64),
        'view_max': np.full(n, -np.inf, dtype=np.float64),
        'view_sum': np.zeros(n, dtype=np.float64),
        'relaz_min': np.full(n, np.inf, dtype=np.float64),
        'relaz_max': np.full(n, -np.inf, dtype=np.float64),
        'inc_min': np.full(n, np.inf, dtype=np.float64),
        'inc_max': np.full(n, -np.inf, dtype=np.float64),
        'spec_count': np.zeros((n, m), dtype=np.int32),
        'spec_sum': np.zeros((n, m), dtype=np.float64),
        'spec_sumsq': np.zeros((n, m), dtype=np.float64),
        'spec_min': np.full((n, m), np.inf, dtype=np.float64),
        'spec_max': np.full((n, m), -np.inf, dtype=np.float64),
        'variable_names': variable_names,
    }


def _safe_numeric_col(df: pd.DataFrame, col: str, default=np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)


def _update_point_summary(acc: dict[str, Any], df: pd.DataFrame, spectra: np.ndarray, variable_names: list[str]) -> None:
    if len(df) == 0:
        return
    pid = pd.to_numeric(df['point_id'], errors='coerce').to_numpy(dtype=np.int64)
    valid_pid = (pid >= 0) & (pid < acc['obs_count'].shape[0])
    if not np.any(valid_pid):
        return
    pid = pid[valid_pid]
    dfx = df.loc[valid_pid].reset_index(drop=True)
    spec = np.asarray(spectra[valid_pid], dtype=np.float64)

    np.add.at(acc['obs_count'], pid, 1)
    if 'is_clear' in dfx.columns:
        is_clear = to_bool_series(dfx['is_clear'], default=False).to_numpy(dtype=bool)
        np.add.at(acc['clear_count'], pid[is_clear], 1)

    q = dfx.get('brf_quality_level', pd.Series(['low'] * len(dfx))).astype(str).str.lower().to_numpy()
    np.add.at(acc['high_count'], pid[q == 'high'], 1)
    np.add.at(acc['medium_count'], pid[q == 'medium'], 1)
    np.add.at(acc['low_count'], pid[q == 'low'], 1)

    w = _safe_numeric_col(dfx, 'view_weight_raw', 0.0)
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    np.add.at(acc['weight_sum'], pid, w)
    np.add.at(acc['weight_sq_sum'], pid, w * w)
    np.maximum.at(acc['weight_max'], pid, w)

    for col, min_key, max_key, sum_key in [
        ('view_zenith_deg', 'view_min', 'view_max', 'view_sum'),
    ]:
        vals = _safe_numeric_col(dfx, col)
        good = np.isfinite(vals)
        np.minimum.at(acc[min_key], pid[good], vals[good])
        np.maximum.at(acc[max_key], pid[good], vals[good])
        np.add.at(acc[sum_key], pid[good], vals[good])

    vals = _safe_numeric_col(dfx, 'relative_azimuth_deg')
    good = np.isfinite(vals)
    np.minimum.at(acc['relaz_min'], pid[good], vals[good])
    np.maximum.at(acc['relaz_max'], pid[good], vals[good])

    vals = _safe_numeric_col(dfx, 'local_solar_incidence_deg')
    good = np.isfinite(vals)
    np.minimum.at(acc['inc_min'], pid[good], vals[good])
    np.maximum.at(acc['inc_max'], pid[good], vals[good])


    for j in range(len(variable_names)):
        vals = spec[:, j]
        good = np.isfinite(vals)
        if not np.any(good):
            continue
        np.add.at(acc['spec_count'][:, j], pid[good], 1)
        np.add.at(acc['spec_sum'][:, j], pid[good], vals[good])
        np.add.at(acc['spec_sumsq'][:, j], pid[good], vals[good] * vals[good])
        np.minimum.at(acc['spec_min'][:, j], pid[good], vals[good])
        np.maximum.at(acc['spec_max'][:, j], pid[good], vals[good])


def _finalize_point_summary(acc: dict[str, Any], point_xyz: np.ndarray, point_quality_cfg: dict) -> pd.DataFrame:
    obs_count = acc['obs_count']
    point_id = np.flatnonzero(obs_count > 0).astype(np.int64)
    if point_id.size == 0:
        return pd.DataFrame()
    weight_sum = acc['weight_sum'][point_id]
    weight_sq = acc['weight_sq_sum'][point_id]
    primary_ratio = np.divide(acc['weight_max'][point_id], weight_sum, out=np.full_like(weight_sum, np.nan), where=weight_sum > 0)
    effective_view = np.divide(weight_sum * weight_sum, weight_sq, out=np.full_like(weight_sum, np.nan), where=weight_sq > 0)
    view_mean = np.divide(acc['view_sum'][point_id], obs_count[point_id], out=np.full(point_id.size, np.nan), where=obs_count[point_id] > 0)
    view_min = acc['view_min'][point_id]
    view_max = acc['view_max'][point_id]
    relaz_min = acc['relaz_min'][point_id]
    relaz_max = acc['relaz_max'][point_id]
    inc_min = acc['inc_min'][point_id]
    inc_max = acc['inc_max'][point_id]
    for arr in [view_min, view_max, relaz_min, relaz_max, inc_min, inc_max]:
        arr[~np.isfinite(arr)] = np.nan

    clear_count = acc['clear_count'][point_id]
    view_range = view_max - view_min
    high = (
        (clear_count >= int(point_quality_cfg.get('high_min_clear_views', 5)))
        & (view_range >= float(point_quality_cfg.get('high_min_view_zenith_range_deg', 20.0)))
        & (primary_ratio <= float(point_quality_cfg.get('high_max_primary_weight_ratio', 0.85)))
    )
    medium = (~high) & (clear_count >= int(point_quality_cfg.get('medium_min_clear_views', 3)))
    point_quality = np.full(point_id.size, 'low', dtype=object)
    point_quality[medium] = 'medium'
    point_quality[high] = 'high'

    df = pd.DataFrame({
        'point_id': point_id,
        'x': np.asarray(point_xyz[point_id, 0], dtype=np.float32),
        'y': np.asarray(point_xyz[point_id, 1], dtype=np.float32),
        'z': np.asarray(point_xyz[point_id, 2], dtype=np.float32),
        'n_observations': obs_count[point_id].astype(np.int32),
        'n_clear_observations': clear_count.astype(np.int32),
        'n_high_quality_observations': acc['high_count'][point_id].astype(np.int32),
        'n_medium_quality_observations': acc['medium_count'][point_id].astype(np.int32),
        'n_low_quality_observations': acc['low_count'][point_id].astype(np.int32),
        'effective_view_count': effective_view.astype(np.float32),
        'primary_weight_ratio': primary_ratio.astype(np.float32),
        'view_zenith_min_deg': view_min.astype(np.float32),
        'view_zenith_max_deg': view_max.astype(np.float32),
        'view_zenith_range_deg': view_range.astype(np.float32),
        'view_zenith_mean_deg': view_mean.astype(np.float32),
        'relative_azimuth_min_deg': relaz_min.astype(np.float32),
        'relative_azimuth_max_deg': relaz_max.astype(np.float32),
        'relative_azimuth_range_deg': (relaz_max - relaz_min).astype(np.float32),
        'local_solar_incidence_min_deg': inc_min.astype(np.float32),
        'local_solar_incidence_max_deg': inc_max.astype(np.float32),
        'local_solar_incidence_range_deg': (inc_max - inc_min).astype(np.float32),
        'brf_quality_level': point_quality,
    })

    names = list(acc['variable_names'])
    spectral_summary_cols: dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        cnt = acc['spec_count'][point_id, j].astype(np.float64)
        mean = np.divide(acc['spec_sum'][point_id, j], cnt, out=np.full(point_id.size, np.nan), where=cnt > 0)
        var = np.divide(acc['spec_sumsq'][point_id, j], cnt, out=np.full(point_id.size, np.nan), where=cnt > 0) - mean * mean
        var = np.maximum(var, 0.0)
        vmin = acc['spec_min'][point_id, j]
        vmax = acc['spec_max'][point_id, j]
        vmin[~np.isfinite(vmin)] = np.nan
        vmax[~np.isfinite(vmax)] = np.nan
        safe = str(name).replace(' ', '_')
        spectral_summary_cols[f'{safe}_mean'] = mean.astype(np.float32)
        spectral_summary_cols[f'{safe}_std'] = np.sqrt(var).astype(np.float32)
        spectral_summary_cols[f'{safe}_range'] = (vmax - vmin).astype(np.float32)
    if spectral_summary_cols:
        df = pd.concat([df, pd.DataFrame(spectral_summary_cols, index=df.index)], axis=1)
    return df.copy()


def _init_angular_accumulators() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    return {}


def _update_angular_stats(
    angular_acc: dict[tuple[str, str], dict[str, np.ndarray]],
    df: pd.DataFrame,
    spectra: np.ndarray,
    variable_names: list[str],
    stat_cfg: dict,
) -> None:
    if len(df) == 0:
        return
    bin_deg = float(stat_cfg.get('bin_deg', 5.0) or 5.0)
    if bin_deg <= 0:
        bin_deg = 5.0
    angle_cols = list(stat_cfg.get('angle_columns', ['view_zenith_deg']) or ['view_zenith_deg'])
    requested_vars = stat_cfg.get('variables') or variable_names
    requested_vars = [str(x) for x in requested_vars]
    var_indices = [(v, i) for i, v in enumerate(variable_names) if v in requested_vars]
    if not var_indices:
        var_indices = list(zip(variable_names, range(len(variable_names))))

    for angle_col in angle_cols:
        if angle_col not in df.columns:
            continue
        angles = pd.to_numeric(df[angle_col], errors='coerce').to_numpy(dtype=np.float64)
        if angle_col == 'relative_azimuth_deg':
            max_angle = 180.0
        else:
            max_angle = 90.0
        nbins = int(np.ceil(max_angle / bin_deg))
        bin_idx = np.floor(np.clip(angles, 0.0, max_angle - 1e-9) / bin_deg).astype(np.int32)
        angle_ok = np.isfinite(angles) & (angles >= 0.0) & (angles <= max_angle)
        for var_name, j in var_indices:
            vals = np.asarray(spectra[:, j], dtype=np.float64)
            good = angle_ok & np.isfinite(vals)
            if not np.any(good):
                continue
            key = (angle_col, var_name)
            if key not in angular_acc:
                angular_acc[key] = {
                    'count': np.zeros(nbins, dtype=np.int64),
                    'sum': np.zeros(nbins, dtype=np.float64),
                    'sumsq': np.zeros(nbins, dtype=np.float64),
                    'bin_deg': np.asarray([bin_deg], dtype=np.float64),
                    'max_angle': np.asarray([max_angle], dtype=np.float64),
                }
            acc = angular_acc[key]
            idx = bin_idx[good]
            v = vals[good]
            acc['count'] += np.bincount(idx, minlength=nbins).astype(np.int64)
            acc['sum'] += np.bincount(idx, weights=v, minlength=nbins)
            acc['sumsq'] += np.bincount(idx, weights=v * v, minlength=nbins)


def _finalize_angular_stats(angular_acc: dict[tuple[str, str], dict[str, np.ndarray]], ci_level: float = 0.95) -> pd.DataFrame:

    z_value = 1.96
    rows = []
    for (angle_col, var_name), acc in sorted(angular_acc.items()):
        count = acc['count'].astype(np.float64)
        s = acc['sum']
        ss = acc['sumsq']
        bin_deg = float(acc['bin_deg'][0])
        nbins = count.size
        mean = np.divide(s, count, out=np.full(nbins, np.nan), where=count > 0)
        var = np.divide(ss, count, out=np.full(nbins, np.nan), where=count > 0) - mean * mean
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        sem = np.divide(std, np.sqrt(count), out=np.full(nbins, np.nan), where=count > 0)
        ci95 = z_value * sem
        for i in range(nbins):
            rows.append({
                'angle_type': angle_col,
                'variable': var_name,
                'bin_left_deg': float(i * bin_deg),
                'bin_right_deg': float((i + 1) * bin_deg),
                'bin_center_deg': float((i + 0.5) * bin_deg),
                'n': int(count[i]),
                'mean': float(mean[i]) if np.isfinite(mean[i]) else np.nan,
                'std': float(std[i]) if np.isfinite(std[i]) else np.nan,
                'sem': float(sem[i]) if np.isfinite(sem[i]) else np.nan,
                'ci95': float(ci95[i]) if np.isfinite(ci95[i]) else np.nan,
                'ci_level': float(ci_level),
            })
    return pd.DataFrame(rows)


def _safe_import_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _write_directional_curve_plots(stats: pd.DataFrame, out_dir: Path, stat_cfg: dict, dpi: int = 180) -> None:
    if stats.empty:
        return
    plot_dir = ensure_dir(out_dir / 'directional_curves')
    plot_vars = list(stat_cfg.get('plot_variables', ['NDVI']) or ['NDVI'])
    min_n = int(stat_cfg.get('min_count_for_plot', 10) or 10)
    plt = _safe_import_matplotlib()
    for variable in plot_vars:
        for angle_type in stats['angle_type'].dropna().unique():
            cur = stats[(stats['variable'] == variable) & (stats['angle_type'] == angle_type)].copy()
            cur = cur[cur['n'] >= min_n]
            if cur.empty:
                continue
            x = cur['bin_center_deg'].to_numpy(dtype=np.float64)
            y = cur['mean'].to_numpy(dtype=np.float64)
            ci = cur['ci95'].to_numpy(dtype=np.float64)
            n = cur['n'].to_numpy(dtype=np.int64)
            fig = plt.figure(figsize=(7.2, 4.6))
            ax = fig.add_subplot(111)
            ax.plot(x, y, marker='o', linewidth=1.5)
            ax.fill_between(x, y - ci, y + ci, alpha=0.2)
            ax.set_xlabel(angle_type.replace('_', ' ').replace('deg', '(deg)'))
            ax.set_ylabel(variable)
            ax.set_title(f'{variable} directional curve with 95% CI')
            ax2 = ax.twinx()
            ax2.bar(x, n, width=max(1.0, float(cur['bin_right_deg'].iloc[0] - cur['bin_left_deg'].iloc[0]) * 0.75), alpha=0.15)
            ax2.set_ylabel('Sample count')
            fig.tight_layout()
            fig.savefig(plot_dir / f'{variable}_{angle_type}_curve_ci_count.png', dpi=dpi)
            plt.close(fig)


def _write_quality_summary(obs_quality_counts: dict[str, int], point_summary: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for level in ['high', 'medium', 'low']:
        rows.append({'scope': 'observation', 'brf_quality_level': level, 'count': int(obs_quality_counts.get(level, 0))})
    if not point_summary.empty and 'brf_quality_level' in point_summary.columns:
        vc = point_summary['brf_quality_level'].value_counts(dropna=False).to_dict()
        for level in ['high', 'medium', 'low']:
            rows.append({'scope': 'point', 'brf_quality_level': level, 'count': int(vc.get(level, 0))})
    pd.DataFrame(rows).to_csv(out_dir / 'brf_quality_level_summary.csv', index=False, encoding='utf-8-sig')


def _write_best_nadir_explanation(out_dir: Path) -> None:
    cn = """# 正射 / best-nadir proxy 对比解释\n\n`best-nadir proxy` 不是独立真值，而是从点级方向性观测中选取观测天顶角最小的一条观测，作为近天底二维正射表达的替代参考。该对比用于说明三维多视角 BRF 融合结果与近天底表达之间的差异，而不应被表述为绝对精度验证。\n\n若 3-D fused NDVI 与 best-nadir NDVI 的 MAE 或 RMSE 较小，说明二者在整体数值尺度上具有一定一致性。若相关系数或 $R^2$ 较低，则可能表明三维融合结果保留了树冠侧面、垂直结构、遮挡边缘和方向性反射造成的结构性差异。这种差异是本文 3-D BRF 产品相对于传统正射产品的主要意义之一。\n"""
    en = """# Interpretation of the orthographic / best-nadir proxy comparison\n\nThe `best-nadir proxy` is not an independent ground truth. It is the observation with the smallest view zenith angle for each point and is used as a nadir-like reference for a conventional 2-D orthographic representation. Therefore, this comparison assesses the difference between the 3-D directional BRF product and a nadir-like proxy, rather than the absolute radiometric accuracy.\n\nA small MAE or RMSE indicates that the 3-D fused NDVI and the best-nadir NDVI are comparable in their overall numerical scale. A low correlation coefficient or low $R^2$ may indicate that the 3-D fused product preserves structure-dependent and view-dependent differences related to canopy sides, vertical surfaces, occlusion boundaries, and directional reflectance effects. These differences are part of the added value of the proposed 3-D BRF product compared with conventional orthographic products.\n"""
    (out_dir / 'best_nadir_comparison_interpretation_CN.md').write_text(cn, encoding='utf-8')
    (out_dir / 'best_nadir_comparison_interpretation_EN.md').write_text(en, encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    quiet = is_quiet(cfg)
    out_dir = ensure_dir(cfg['outputs'].get('directional_brf_dir', Path(cfg['outputs']['fusion_dir']).parent / 'brf'))
    selection_dir = Path(cfg['outputs']['selection_dir'])
    point_dir = Path(cfg['outputs']['pointcloud_dir'])
    scene_db = read_table(Path(cfg['outputs']['scene_db_dir']) / 'scene_database.csv')
    scene_map = scene_db.drop_duplicates('scene_id').set_index('scene_id')

    vis_path = selection_dir / 'visible_observations.csv'
    if not vis_path.exists():
        raise FileNotFoundError(f'visible_observations.csv does not exist: {vis_path}')

    dcfg = dict(cfg.get('directional_brf', {}) or {})
    include_all_retained_views = bool(dcfg.get('include_all_retained_views', False))
    output_format = str(dcfg.get('output_format', 'parquet_and_csv')).strip().lower()
    max_csv_rows = int(dcfg.get('max_csv_rows', 100000))
    chunk_rows = max(1000, int(dcfg.get('chunk_rows', 200000)))
    workers = max(1, int(dcfg.get('workers', cfg.get('processing', {}).get('fusion_workers', 20))))
    write_full_spectra_npy = bool(dcfg.get('write_full_spectra_npy', True))
    parquet_include_spectra = bool(dcfg.get('parquet_include_spectra', False))
    csv_preview_include_spectra = bool(dcfg.get('csv_preview_include_spectra', False))
    enable_memmap_cache = bool(dcfg.get('enable_memmap_cache', True))
    use_persistent_executor = bool(dcfg.get('use_persistent_executor', True))
    write_observation_index_csv = bool(dcfg.get('write_observation_index_csv', False))
    allow_slow_row_count = bool(dcfg.get('allow_slow_csv_row_count', False))
    paper_fields_only = bool(dcfg.get('paper_fields_only', True))
    add_quality_level = bool(dcfg.get('add_quality_level', True))
    build_point_summary = bool(dcfg.get('build_point_summary', True))
    build_angular_bin_statistics = bool(dcfg.get('build_angular_bin_statistics', True))
    write_directional_curve_plots = bool(dcfg.get('write_directional_curve_plots', True))
    write_quality_summary = bool(dcfg.get('write_quality_summary', False))
    write_interpretation_files = bool(dcfg.get('write_interpretation_files', False))

    if output_format not in {'parquet_and_csv', 'parquet', 'csv'}:
        raise ValueError(f'Invalid directional_brf.output_format: {output_format}')

    parquet_pa = parquet_pq = None
    if 'parquet' in output_format:
        parquet_pa, parquet_pq = _import_pyarrow_or_raise()

    first_scene_id = str(scene_db.iloc[0]['scene_id'])
    band_indices, band_numbers, wavelengths_used, band_cols = _select_directional_bands(cfg, scene_map.loc[first_scene_id])
    if not band_indices:
        raise RuntimeError('No valid directional BRF bands were selected. Check bands and directional_brf.bands_for_table.')

    index_plus_one = [int(x) for x in cfg.get('processing', {}).get('index_band_numbers_stored_plus_one', DEFAULT_INDEX_PLUS_ONE_BANDS)]
    index_local_indices = [i for i, b in enumerate(band_numbers) if int(b) in set(index_plus_one)]

    point_xyz = np.load(point_dir / 'point_xyz.npy', mmap_mode='r')
    point_normals = np.load(point_dir / 'point_normals.npy', mmap_mode='r')
    point_verticality = np.load(point_dir / 'point_surface_verticality.npy', mmap_mode='r')
    point_normal_confidence = np.load(point_dir / 'point_normal_confidence.npy', mmap_mode='r')
    point_count = int(point_xyz.shape[0])

    total_rows, row_count_source = _infer_visible_row_count(cfg, vis_path, allow_slow_count=allow_slow_row_count)
    if total_rows <= 0 and write_full_spectra_npy:
        raise RuntimeError('Cannot determine the number of visible observations. Check selection_summary.json or set directional_brf.expected_observation_rows.')

    print(
        f'directional_brf: output_format={output_format}, workers={workers}, chunk_rows={chunk_rows}, '
        f'row_count={total_rows} ({row_count_source}), paper_fields_only={paper_fields_only}'
    )

    spectra_memmap = None
    spectra_path = out_dir / 'directional_brf_spectra.npy'
    if write_full_spectra_npy:
        spectra_memmap = np.lib.format.open_memmap(spectra_path, mode='w+', dtype=np.float32, shape=(int(total_rows), int(len(band_numbers))))

    csv_path = out_dir / 'point_directional_brf_observations_preview.csv'
    parquet_path = out_dir / 'point_directional_brf_observations.parquet'
    index_path = out_dir / 'directional_brf_observation_index.csv'
    for p in [csv_path, parquet_path, index_path]:
        if p.exists():
            p.unlink()

    point_acc = _init_summary_accumulators(point_count, band_cols) if build_point_summary else None
    angular_acc = _init_angular_accumulators() if build_angular_bin_statistics else None
    obs_quality_counts = {'high': 0, 'medium': 0, 'low': 0}

    parquet_writer = None
    pa = pq = None
    csv_written = 0
    global_offset = 0
    processed = 0

    header_cols = pd.read_csv(vis_path, nrows=0, encoding='utf-8-sig').columns.tolist()
    reader = pd.read_csv(
        vis_path,
        usecols=header_cols,
        dtype=_visible_dtype_map(header_cols),
        chunksize=chunk_rows,
        encoding='utf-8-sig',
    )

    executor = ThreadPoolExecutor(max_workers=workers) if (use_persistent_executor and workers > 1) else None
    try:
        for chunk_idx, vis in enumerate(reader, start=1):
            if len(vis) == 0:
                continue
            if 'keep_for_fusion' in vis.columns and not include_all_retained_views:
                vis = vis[to_bool_series(vis['keep_for_fusion'], default=True)].copy()
            if len(vis) == 0:
                continue
            vis['point_id'] = pd.to_numeric(vis['point_id'], errors='coerce').astype('Int64')
            vis = vis[vis['point_id'].notna()].copy()
            vis['point_id'] = vis['point_id'].astype(np.int64)
            vis = vis[(vis['point_id'] >= 0) & (vis['point_id'] < point_count)].copy()
            if vis.empty:
                continue
            vis = vis.reset_index(drop=True)

            pid = vis['point_id'].to_numpy(dtype=np.int64)
            xyz = np.asarray(point_xyz[pid], dtype=np.float32)
            normals = np.asarray(point_normals[pid], dtype=np.float32)
            n_obs = len(vis)
            vis.insert(1, 'observation_id', np.arange(global_offset, global_offset + n_obs, dtype=np.int64))
            vis['x'] = xyz[:, 0]
            vis['y'] = xyz[:, 1]
            vis['z'] = xyz[:, 2]
            vis['normal_x'] = normals[:, 0]
            vis['normal_y'] = normals[:, 1]
            vis['normal_z'] = normals[:, 2]
            vis['point_surface_verticality'] = np.asarray(point_verticality[pid], dtype=np.float32)
            vis['point_normal_confidence'] = np.asarray(point_normal_confidence[pid], dtype=np.float32)

            spectra = np.full((n_obs, len(band_numbers)), np.nan, dtype=np.float32)
            tasks = []
            for scene_id, group in vis.groupby('scene_id', sort=False):
                scene_id = str(scene_id)
                if scene_id not in scene_map.index:
                    continue
                row = scene_map.loc[scene_id].to_dict()
                tasks.append((scene_id, group, row))

            if workers <= 1 or len(tasks) <= 1:
                results = [_sample_one_scene(sid, grp, row, band_indices, index_local_indices, enable_memmap_cache) for sid, grp, row in tasks]
            else:
                ex = executor if executor is not None else ThreadPoolExecutor(max_workers=workers)
                futs = [ex.submit(_sample_one_scene, sid, grp, row, band_indices, index_local_indices, enable_memmap_cache) for sid, grp, row in tasks]
                results = [fu.result() for fu in as_completed(futs)]
                if executor is None:
                    ex.shutdown(wait=True)

            for row_idx, spec in results:
                spectra[row_idx, :] = spec

            if spectra_memmap is not None:
                end_offset = global_offset + n_obs
                if end_offset > spectra_memmap.shape[0]:
                    raise RuntimeError(f'directional_brf_spectra.npy was under-allocated: need={end_offset}, allocated={spectra_memmap.shape[0]}.')
                spectra_memmap[global_offset:end_offset, :] = spectra

            base_df = vis.reset_index(drop=True)
            if add_quality_level:
                base_df['brf_quality_level'] = _classify_observation_quality(base_df, dict(dcfg.get('quality_level', {}) or {}))
            else:
                base_df['brf_quality_level'] = 'unclassified'

            quality_counts = base_df['brf_quality_level'].astype(str).str.lower().value_counts().to_dict()
            for level in obs_quality_counts:
                obs_quality_counts[level] += int(quality_counts.get(level, 0))

            if point_acc is not None:
                _update_point_summary(point_acc, base_df, spectra, band_cols)
            if angular_acc is not None:
                _update_angular_stats(angular_acc, base_df, spectra, band_cols, dict(dcfg.get('angular_statistics', {}) or {}))

            if paper_fields_only:
                base_df = base_df[_paper_observation_columns(base_df)]

            if write_observation_index_csv:
                append_table(base_df, index_path, index=False)

            spectra_df = None
            if (parquet_include_spectra and 'parquet' in output_format) or (csv_preview_include_spectra and 'csv' in output_format and csv_written < max_csv_rows):
                spectra_df = pd.DataFrame(spectra, columns=band_cols)

            if 'csv' in output_format and csv_written < max_csv_rows:
                take = min(max_csv_rows - csv_written, len(base_df))
                if csv_preview_include_spectra and spectra_df is not None:
                    csv_df = pd.concat([base_df.iloc[:take].reset_index(drop=True), spectra_df.iloc[:take].reset_index(drop=True)], axis=1)
                else:
                    csv_df = base_df.iloc[:take]
                append_table(csv_df, csv_path, index=False)
                csv_written += int(take)

            if 'parquet' in output_format:
                if parquet_include_spectra and spectra_df is not None:
                    parquet_df = pd.concat([base_df, spectra_df], axis=1)
                else:
                    parquet_df = base_df
                if parquet_writer is None:
                    parquet_writer, pa, pq = _init_parquet_writer(parquet_path, parquet_df, parquet_pa, parquet_pq)
                else:
                    table = pa.Table.from_pandas(parquet_df, preserve_index=False)
                    parquet_writer.write_table(table)

            global_offset += n_obs
            processed += n_obs
            if (not quiet) or chunk_idx % 10 == 0:
                print(f'directional BRF chunk {chunk_idx}: processed={processed}/{total_rows}')
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if parquet_writer is not None:
        parquet_writer.close()
    if spectra_memmap is not None and hasattr(spectra_memmap, 'flush'):
        spectra_memmap.flush()

    band_meta = pd.DataFrame({
        'band_number': band_numbers,
        'band_index_zero_based': band_indices,
        'wavelength_nm': wavelengths_used,
        'variable_name': band_cols,
        'is_vegetation_index': [int(b) in INDEX_BAND_NAME_MAP for b in band_numbers],
        'stored_plus_one_corrected': [int(b) in set(index_plus_one) for b in band_numbers],
        'description': [
            {'NDVI': 'Normalized Difference Vegetation Index', 'RVI': 'Ratio Vegetation Index',
             'MSAVI': 'Modified Soil-Adjusted Vegetation Index', 'NDWI': 'Normalized Difference Water Index'}.get(name, f'Reflectance band {b}')
            for b, name in zip(band_numbers, band_cols)
        ],
    })
    band_meta.to_csv(out_dir / 'directional_brf_band_metadata.csv', index=False, encoding='utf-8-sig')

    point_summary_path = ''
    point_summary_csv = ''
    point_summary = pd.DataFrame()
    if point_acc is not None:
        point_summary = _finalize_point_summary(point_acc, np.asarray(point_xyz), dict(dcfg.get('point_quality', {}) or {}))
        if not point_summary.empty:
            point_summary_path = str(out_dir / 'point_brf_summary.parquet')
            point_summary.to_parquet(point_summary_path, index=False)
            point_summary_csv = str(out_dir / 'point_brf_summary_preview.csv')
            point_summary.head(100000).to_csv(point_summary_csv, index=False, encoding='utf-8-sig')

    angular_stats_path = ''
    if angular_acc is not None:
        stat_cfg = dict(dcfg.get('angular_statistics', {}) or {})
        angular_stats = _finalize_angular_stats(angular_acc, ci_level=float(stat_cfg.get('ci_level', 0.95) or 0.95))
        angular_stats_path = str(out_dir / 'angular_bin_statistics.csv')
        angular_stats.to_csv(angular_stats_path, index=False, encoding='utf-8-sig')
        if write_directional_curve_plots and not angular_stats.empty:
            _write_directional_curve_plots(angular_stats, out_dir, stat_cfg, dpi=int(cfg.get('experiments', {}).get('plot_dpi', 180) or 180))

    if write_quality_summary:
        _write_quality_summary(obs_quality_counts, point_summary, out_dir)
    if write_interpretation_files:
        _write_best_nadir_explanation(out_dir)

    save_json({
        'observation_count': int(processed),
        'source_visible_observation_rows': int(total_rows),
        'row_count_source': str(row_count_source),
        'band_count': int(len(band_numbers)),
        'band_numbers': [int(b) for b in band_numbers],
        'variable_names': band_cols,
        'csv_preview_rows': int(csv_written),
        'csv_preview_path': str(csv_path) if csv_path.exists() else '',
        'parquet_path': str(parquet_path) if parquet_path.exists() else '',
        'spectra_npy_path': str(spectra_path) if spectra_path.exists() else '',
        'band_metadata_path': str(out_dir / 'directional_brf_band_metadata.csv'),
        'point_brf_summary_path': point_summary_path,
        'point_brf_summary_preview_csv': point_summary_csv,
        'angular_bin_statistics_path': angular_stats_path,
        'quality_level_summary_path': (
            str(out_dir / 'brf_quality_level_summary.csv') if write_quality_summary else ''
        ),
        'parquet_include_spectra': bool(parquet_include_spectra),
        'csv_preview_include_spectra': bool(csv_preview_include_spectra),
        'paper_fields_only': bool(paper_fields_only),
        'enable_memmap_cache': bool(enable_memmap_cache),
        'use_persistent_executor': bool(use_persistent_executor),
        'workers': int(workers),
        'formula': 'BRF_{p,k}(lambda) = rho_k(u_{p,k}, v_{p,k}, lambda); observation_id indexes rows in directional_brf_spectra.npy',
    }, out_dir / 'directional_brf_summary.json')

    print(f'Output: {out_dir}')
    print(f'Point-level directional BRF observations: {processed}')
    print(f'BRF variables: {band_cols}')
    print(f'Spectra matrix: {spectra_path if spectra_path.exists() else "not written"}')
    print(f'Observation table: {parquet_path if parquet_path.exists() else "not written"}')
    print(f'Point summary: {point_summary_path or "not written"}')
    print(f'Angular statistics: {angular_stats_path or "not written"}')


if __name__ == '__main__':
    main()
