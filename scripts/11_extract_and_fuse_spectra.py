

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import (
    append_table,
    ensure_dir,
    get_band_selection,
    is_quiet,
    load_config,
    read_table,
    resolve_center_window,
    save_json,
)
from hyperspectral_pointcloud_fusion.envi import (
    bilinear_sample_selected_bands_bsq,
    parse_envi_hdr,
    read_envi_bsq_memmap,
    wavelength_list_from_hdr,
)


REQUIRED_SCENE_COLUMNS = [
    'hdr_path', 'data_path', 'samples', 'lines', 'bands',
    'data_type', 'byte_order', 'header_offset', 'reflectance_scale_factor',
]

FUSION_COLUMNS = [
    'point_id', 'scene_code', 'scene_id', 'u', 'v', 'range_m', 'offaxis_deg',
    'border_dist_px', 'view_zenith_deg', 'surface_view_cos',
    'surface_verticality', 'normal_confidence', 'visibility_score',
    'view_weight_raw', 'local_empty_cone_deg', 'blocker_count', 'is_clear',
    'keep_for_fusion',
]

FLOAT_COLUMNS = [
    'u', 'v', 'range_m', 'offaxis_deg', 'border_dist_px', 'view_zenith_deg',
    'surface_view_cos', 'surface_verticality', 'normal_confidence',
    'visibility_score', 'view_weight_raw', 'local_empty_cone_deg',
]

DEFAULT_INDEX_PLUS_ONE_BANDS = [165, 167, 168]
_CUBE_CACHE: dict[str, np.memmap] = {}
_CUBE_CACHE_LOCK = Lock()


def _open_cube(row: dict):
    return read_envi_bsq_memmap(
        row['data_path'],
        samples=int(row['samples']),
        lines=int(row['lines']),
        bands=int(row['bands']),
        data_type=int(row['data_type']),
        byte_order=int(row['byte_order']),
        header_offset=int(row['header_offset']),
    )


def _get_cube(row: dict, enable_cache: bool = True):
    if not enable_cache:
        return _open_cube(row)
    key = str(row['data_path'])
    with _CUBE_CACHE_LOCK:
        cube = _CUBE_CACHE.get(key)
        if cube is None:
            cube = _open_cube(row)
            _CUBE_CACHE[key] = cube
        return cube


def to_bool_series(s: pd.Series, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default).astype(bool)
    text = s.fillna(str(default)).astype(str).str.strip().str.lower()
    return text.isin(['true', '1', 'yes', 'y'])


def build_scene_table(cfg, scene_db: pd.DataFrame) -> pd.DataFrame:
    scene_tbl = scene_db.copy()
    missing = [c for c in REQUIRED_SCENE_COLUMNS if c not in scene_tbl.columns]
    if missing:
        manifest_path = Path(cfg['outputs']['inventory_dir']) / 'scene_manifest.csv'
        manifest = read_table(manifest_path)
        keep_cols = ['scene_id'] + [c for c in REQUIRED_SCENE_COLUMNS if c in manifest.columns]
        scene_tbl = scene_tbl.drop(columns=[c for c in REQUIRED_SCENE_COLUMNS if c in scene_tbl.columns], errors='ignore')
        scene_tbl = scene_tbl.merge(manifest[keep_cols], on='scene_id', how='left')
        still_missing = [c for c in REQUIRED_SCENE_COLUMNS if c not in scene_tbl.columns]
        if still_missing:
            raise KeyError(f'scene table missing required columns: {still_missing}')
    null_critical = scene_tbl['hdr_path'].isna() | scene_tbl['data_path'].isna()
    if bool(null_critical.any()):
        bad_ids = scene_tbl.loc[null_critical, 'scene_id'].astype(str).tolist()[:10]
        raise ValueError(f'scene table has records with missing hdr/data paths, example scene_id={bad_ids}')
    return scene_tbl


def mask_points_in_center_window(x: np.ndarray, y: np.ndarray, window: dict) -> np.ndarray:
    return (
        (x >= float(window['x_min'])) &
        (x <= float(window['x_max'])) &
        (y >= float(window['y_min'])) &
        (y <= float(window['y_max']))
    )


def coerce_visible_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    if 'keep_for_fusion' in df.columns:
        df = df[to_bool_series(df['keep_for_fusion'], default=False)].copy()
    if len(df) == 0:
        return df
    df['point_id'] = pd.to_numeric(df['point_id'], errors='coerce').astype('Int64')
    df = df[df['point_id'].notna()].copy()
    df['point_id'] = df['point_id'].astype(np.int64)
    if 'scene_code' in df.columns:
        df['scene_code'] = pd.to_numeric(df['scene_code'], errors='coerce').fillna(-1).astype(np.int16)
    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    if 'blocker_count' in df.columns:
        df['blocker_count'] = pd.to_numeric(df['blocker_count'], errors='coerce').fillna(0).astype(np.int16)
    if 'is_clear' in df.columns:
        df['is_clear'] = to_bool_series(df['is_clear'], default=False)
    return df


def ensure_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        'view_weight_raw': 1.0,
        'visibility_score': 0.0,
        'range_m': 0.0,
        'offaxis_deg': 0.0,
        'border_dist_px': 0.0,
        'view_zenith_deg': 0.0,
        'surface_view_cos': 1.0,
        'surface_verticality': 0.0,
        'normal_confidence': 0.0,
        'local_empty_cone_deg': 180.0,
        'blocker_count': 0,
        'is_clear': True,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    return df


def split_complete_points(chunk: pd.DataFrame, carry: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if carry is not None and len(carry) > 0:
        chunk = pd.concat([carry, chunk], ignore_index=True)
    if len(chunk) == 0:
        return chunk, None
    last_pid = int(chunk['point_id'].iloc[-1])
    carry_mask = chunk['point_id'].values == last_pid
    return chunk.loc[~carry_mask].copy(), chunk.loc[carry_mask].copy()


def add_weight_columns(vis: pd.DataFrame, proc: dict, mode: str, softmax_temp: float, raw_power: float) -> pd.DataFrame:
    vis = ensure_weight_columns(vis)
    vis = vis[np.isfinite(vis['visibility_score'].astype(float).values)].copy()
    if len(vis) == 0:
        return vis
    max_score = vis.groupby('point_id', sort=False)['visibility_score'].transform('max')

    if mode == 'raw':
        base = vis['view_weight_raw'].astype(float).values
    elif mode == 'hybrid':
        scaled = ((vis['visibility_score'] - max_score) / softmax_temp).clip(-60.0, 60.0)
        soft = np.exp(scaled)
        raw = np.power(np.maximum(vis['view_weight_raw'].astype(float), 1e-18), raw_power)
        base = soft * raw
    elif mode == 'surface_adaptive':
        adaptive = dict(proc.get('surface_adaptive_fusion_params', {}))
        score_temp = max(1e-6, float(adaptive.get('score_temperature', softmax_temp)))
        scaled = ((vis['visibility_score'] - max_score) / score_temp).clip(-60.0, 60.0)
        score_term = np.exp(scaled)
        range_term = np.exp(-np.maximum(vis['range_m'].astype(float).values, 0.0) / max(1e-6, float(adaptive.get('range_scale_m', 120.0))))
        offaxis_term = np.exp(-np.maximum(vis['offaxis_deg'].astype(float).values, 0.0) / max(1e-6, float(adaptive.get('offaxis_scale_deg', 60.0))))
        cone = np.maximum(vis['local_empty_cone_deg'].astype(float).values, 0.0)
        cone_floor = max(0.0, min(1.0, float(adaptive.get('cone_floor', 0.2))))
        cone_term = cone_floor + (1.0 - cone_floor) * (1.0 - np.exp(-cone / max(1e-6, float(adaptive.get('cone_scale_deg', 35.0)))))
        blocker_term = np.exp(-max(0.0, float(adaptive.get('blocker_penalty', 1.0))) * np.maximum(vis['blocker_count'].astype(float).values, 0.0))
        clear_term = np.where(to_bool_series(vis['is_clear'], default=False).values, 1.0, max(1e-6, float(adaptive.get('unclear_penalty', 0.1))))
        normal_floor = max(0.0, min(1.0, float(adaptive.get('normal_confidence_floor', 0.25))))
        normal_conf = np.clip(vis['normal_confidence'].astype(float).values, 0.0, 1.0)
        normal_reliability = normal_floor + (1.0 - normal_floor) * normal_conf
        align_floor = max(0.0, min(1.0, float(adaptive.get('surface_alignment_floor', 0.05))))
        align_power = max(1e-6, float(adaptive.get('surface_alignment_power', 1.0)))
        surface_cos = np.clip(vis['surface_view_cos'].astype(float).values, 0.0, 1.0)
        align_term = align_floor + (1.0 - align_floor) * (surface_cos ** align_power)
        vertical = np.clip(vis['surface_verticality'].astype(float).values, 0.0, 1.0)
        zenith_fraction = np.clip(vis['view_zenith_deg'].astype(float).values / max(1e-6, float(adaptive.get('view_zenith_scale_deg', 22.0))), 0.0, 1.0)
        vertical_term = np.exp(max(0.0, float(adaptive.get('vertical_view_strength', 12.0))) * vertical * zenith_fraction * normal_reliability)
        base = score_term * range_term * offaxis_term * cone_term * blocker_term * clear_term * align_term * vertical_term
    else:
        scaled = ((vis['visibility_score'] - max_score) / softmax_temp).clip(-60.0, 60.0)
        base = np.exp(scaled)

    vis['view_weight_base'] = base
    vis = vis[np.isfinite(vis['view_weight_base']) & (vis['view_weight_base'] > 0)].copy()
    if len(vis) == 0:
        return vis
    raw_sum = vis.groupby('point_id', sort=False)['view_weight_base'].transform('sum').replace(0.0, np.nan)
    vis['view_weight_norm'] = (vis['view_weight_base'] / raw_sum).fillna(0.0).astype(np.float32)
    return vis[vis['view_weight_norm'] > 0].copy()


def sample_scene_group(
    scene_id: str,
    scene_code: int,
    g0: pd.DataFrame,
    row: dict,
    band_indices: list[int],
    index_local_indices: list[int],
    center_width,
    center_height,
    center_policy: str,
    enable_memmap_cache: bool = True,
) -> dict:
    center_window = resolve_center_window(int(row['samples']), int(row['lines']), center_width, center_height)
    u_all = g0['u'].astype(float).values
    v_all = g0['v'].astype(float).values
    in_window = mask_points_in_center_window(u_all, v_all, center_window)
    skipped = int(np.count_nonzero(~in_window))
    g = g0.loc[in_window].copy() if center_policy == 'strict' else g0

    empty = {
        'scene_id': str(scene_id),
        'scene_code': int(scene_code),
        'pid': np.empty(0, dtype=np.int64),
        'weighted_spectra': np.empty((0, len(band_indices)), dtype=np.float64),
        'w': np.empty(0, dtype=np.float64),
        'range_m': np.empty(0, dtype=np.float64),
        'offaxis_deg': np.empty(0, dtype=np.float64),
        'border_px': np.empty(0, dtype=np.float64),
        'empty_cone': np.empty(0, dtype=np.float64),
        'view_zenith': np.empty(0, dtype=np.float64),
        'surface_view_cos': np.empty(0, dtype=np.float64),
        'surface_verticality': np.empty(0, dtype=np.float64),
        'is_clear': np.empty(0, dtype=bool),
        'outside_center_window_count': skipped,
    }
    if len(g) == 0:
        return empty

    cube = _get_cube(row, enable_cache=enable_memmap_cache)
    if center_policy == 'strict':
        cube = cube[
            :,
            int(center_window['y_min']):int(center_window['y_max']) + 1,
            int(center_window['x_min']):int(center_window['x_max']) + 1,
        ]
        x = g['u'].astype(float).values - float(center_window['x_min'])
        y = g['v'].astype(float).values - float(center_window['y_min'])
    else:
        x = g['u'].astype(float).values
        y = g['v'].astype(float).values

    spectra = bilinear_sample_selected_bands_bsq(
        cube,
        x=x,
        y=y,
        band_indices=band_indices,
        scale_factor=float(row['reflectance_scale_factor']),
    ).astype(np.float64)
    if index_local_indices:
        spectra[:, index_local_indices] -= 1.0
    w = g['view_weight_norm'].astype(np.float64).values

    return {
        'scene_id': str(scene_id),
        'scene_code': int(scene_code),
        'pid': g['point_id'].astype(np.int64).values,
        'weighted_spectra': spectra * w[:, None],
        'w': w,
        'range_m': g['range_m'].astype(np.float64).values,
        'offaxis_deg': g['offaxis_deg'].astype(np.float64).values,
        'border_px': g['border_dist_px'].astype(np.float64).values,
        'empty_cone': g['local_empty_cone_deg'].astype(np.float64).values,
        'view_zenith': g['view_zenith_deg'].astype(np.float64).values,
        'surface_view_cos': g['surface_view_cos'].astype(np.float64).values,
        'surface_verticality': g['surface_verticality'].astype(np.float64).values,
        'is_clear': to_bool_series(g['is_clear'], default=False).values,
        'outside_center_window_count': skipped,
    }


def new_scene_summary() -> dict:
    return {
        'fused_observation_count': 0,
        'contributing_point_count': 0,
        'weight_sum_in_scene': 0.0,
        'outside_center_window_count': 0,
    }


def compute_derived_reflectance_indices(fused: np.ndarray, band_numbers: np.ndarray) -> dict[str, np.ndarray]:

    band_to_idx = {int(b): i for i, b in enumerate(np.asarray(band_numbers).tolist())}
    arrays = {}
    if 81 in band_to_idx and 128 in band_to_idx:
        red = np.asarray(fused[:, band_to_idx[81]], dtype=np.float64)
        nir = np.asarray(fused[:, band_to_idx[128]], dtype=np.float64)
        denom = nir + red
        ndvi = np.full(fused.shape[0], np.nan, dtype=np.float32)
        ok = np.isfinite(red) & np.isfinite(nir) & (np.abs(denom) > 1e-12)
        ndvi[ok] = ((nir[ok] - red[ok]) / denom[ok]).astype(np.float32)
        arrays['ndvi_81_128_from_reflectance'] = ndvi
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    quiet = is_quiet(cfg)
    out_dir = ensure_dir(cfg['outputs']['fusion_dir'])
    selection_dir = Path(cfg['outputs']['selection_dir'])
    cand_dir = Path(cfg['outputs']['candidate_dir'])
    proc = dict(cfg.get('processing', {}))

    vis_table_path = selection_dir / 'visible_observations.csv'
    if not vis_table_path.exists():
        raise FileNotFoundError(f'visible observation table not found: {vis_table_path}')

    point_xyz = np.load(Path(cfg['outputs']['pointcloud_dir']) / 'point_xyz.npy', mmap_mode='r')
    n_points = int(point_xyz.shape[0])

    scene_db = read_table(Path(cfg['outputs']['scene_db_dir']) / 'scene_database.csv')
    scene_table = build_scene_table(cfg, scene_db)
    scene_map = scene_table.drop_duplicates('scene_id').set_index('scene_id')
    mapping_df = read_table(cand_dir / 'scene_id_mapping.csv')
    scene_code_lookup = {str(r.scene_id): int(r.scene_code) for r in mapping_df.itertuples(index=False)}

    first_scene = str(scene_table.iloc[0]['scene_id'])
    first_row = scene_map.loc[first_scene]
    hdr = parse_envi_hdr(first_row['hdr_path'])
    wavelengths = wavelength_list_from_hdr(hdr)
    band_indices, band_info = get_band_selection(cfg, int(first_row['bands']), wavelengths)
    band_numbers = np.asarray(band_info['band_numbers_one_based'], dtype=np.int32)
    wavelengths_used = (
        np.asarray([wavelengths[i] for i in band_indices], dtype=np.float32)
        if wavelengths else np.full(len(band_indices), np.nan, dtype=np.float32)
    )
    index_plus_one_bands = [int(x) for x in proc.get('index_band_numbers_stored_plus_one', DEFAULT_INDEX_PLUS_ONE_BANDS)]
    index_plus_one_set = set(index_plus_one_bands)
    index_local_indices = [i for i, band_no in enumerate(band_numbers.tolist()) if int(band_no) in index_plus_one_set]

    mode = str(proc.get('fusion_weight_mode', 'score_softmax')).strip().lower()
    softmax_temp = max(1e-6, float(proc.get('fusion_softmax_temperature', 40.0)))
    raw_power = max(0.0, float(proc.get('fusion_raw_weight_power', 1.0)))
    center_width = proc.get('center_window_width_px', 512)
    center_height = proc.get('center_window_height_px', center_width)
    center_policy = str(proc.get('center_window_policy', 'prefer')).strip().lower()
    fusion_workers = max(1, int(proc.get('fusion_workers', 1)))
    fusion_enable_memmap_cache = bool(proc.get('fusion_enable_memmap_cache', True))
    fusion_use_persistent_executor = bool(proc.get('fusion_use_persistent_executor', True))
    chunk_rows = max(1000, int(proc.get('fusion_csv_chunk_rows', 500000)))
    write_obs_table = bool(proc.get('write_fused_observation_table', False))
    write_point_summary = bool(proc.get('write_fused_point_summary', False))

    obs_table_path = out_dir / 'fused_observation_table.csv'
    if obs_table_path.exists():
        obs_table_path.unlink()
    point_summary_path = out_dir / 'fused_point_summary.csv'
    if point_summary_path.exists() and not write_point_summary:
        point_summary_path.unlink()

    if not quiet:
        print(f'point_count={n_points}, band_count={len(band_indices)}')
        print(f'fusion_weight_mode={mode}, fusion_workers={fusion_workers}, fusion_csv_chunk_rows={chunk_rows}, fusion_enable_memmap_cache={fusion_enable_memmap_cache}, fusion_use_persistent_executor={fusion_use_persistent_executor}')
        print(f'write_fused_observation_table={write_obs_table}, write_fused_point_summary={write_point_summary}')
        print(f'index_band_numbers_stored_plus_one={index_plus_one_bands}, selected_local_indices={index_local_indices}')

    fused_sum = np.zeros((n_points, len(band_indices)), dtype=np.float64)
    weight_sum = np.zeros(n_points, dtype=np.float64)
    clear_weight_sum = np.zeros(n_points, dtype=np.float64)
    weighted_range_sum = np.zeros(n_points, dtype=np.float64)
    weighted_offaxis_sum = np.zeros(n_points, dtype=np.float64)
    weighted_border_sum = np.zeros(n_points, dtype=np.float64)
    weighted_empty_cone_sum = np.zeros(n_points, dtype=np.float64)
    weighted_view_zenith_sum = np.zeros(n_points, dtype=np.float64)
    weighted_surface_view_cos_sum = np.zeros(n_points, dtype=np.float64)
    weighted_surface_verticality_sum = np.zeros(n_points, dtype=np.float64)
    primary_scene_code = np.full(n_points, -1, dtype=np.int16)
    primary_view_weight = np.zeros(n_points, dtype=np.float64)
    primary_range = np.full(n_points, np.nan, dtype=np.float64)
    primary_offaxis = np.full(n_points, np.nan, dtype=np.float64)
    primary_border = np.full(n_points, np.nan, dtype=np.float64)
    primary_empty_cone = np.full(n_points, np.nan, dtype=np.float64)
    primary_view_zenith = np.full(n_points, np.nan, dtype=np.float64)
    primary_surface_view_cos = np.full(n_points, np.nan, dtype=np.float64)
    entropy_acc = np.zeros(n_points, dtype=np.float64)
    contribution_counter = np.zeros(n_points, dtype=np.int32)

    available_cols = list(pd.read_csv(vis_table_path, nrows=0, encoding='utf-8-sig').columns)
    usecols = [c for c in FUSION_COLUMNS if c in available_cols]
    missing_required = sorted({'point_id', 'scene_id', 'u', 'v'} - set(usecols))
    if missing_required:
        raise KeyError(f'visible observation table missing required columns: {missing_required}')
    dtype_map = {
        'point_id': np.int64,
        'scene_code': np.int16,
        'scene_id': str,
        'u': np.float32,
        'v': np.float32,
        'range_m': np.float32,
        'offaxis_deg': np.float32,
        'border_dist_px': np.float32,
        'view_zenith_deg': np.float32,
        'surface_view_cos': np.float32,
        'surface_verticality': np.float32,
        'normal_confidence': np.float32,
        'visibility_score': np.float32,
        'view_weight_raw': np.float32,
        'local_empty_cone_deg': np.float32,
        'blocker_count': np.int16,
    }
    dtype_map = {k: v for k, v in dtype_map.items() if k in usecols}

    scene_summary = defaultdict(new_scene_summary)
    outside_center_window_count = 0
    processed_rows = 0
    sampled_rows = 0
    chunk_index = 0
    carry = None

    reader = pd.read_csv(
        vis_table_path,
        usecols=usecols,
        dtype=dtype_map,
        chunksize=chunk_rows,
        encoding='utf-8-sig',
    )

    persistent_fusion_executor = (
        ThreadPoolExecutor(max_workers=fusion_workers)
        if (fusion_workers > 1 and fusion_use_persistent_executor)
        else None
    )

    def process_chunk(vis_chunk: pd.DataFrame) -> tuple[int, int, int]:
        nonlocal outside_center_window_count
        if len(vis_chunk) == 0:
            return 0, 0, 0
        vis_chunk = add_weight_columns(vis_chunk, proc, mode, softmax_temp, raw_power)
        if len(vis_chunk) == 0:
            return 0, 0, 0
        if write_obs_table:
            append_table(vis_chunk, obs_table_path, index=False)

        pid_all = vis_chunk['point_id'].astype(np.int64).values
        w_all = vis_chunk['view_weight_norm'].astype(np.float64).values
        np.add.at(entropy_acc, pid_all, -(w_all * np.log(np.maximum(w_all, 1e-12))))

        groups = []
        for scene_id, g0 in vis_chunk.groupby('scene_id', sort=False):
            scene_id = str(scene_id)
            if scene_id not in scene_map.index:
                continue
            scene_code = scene_code_lookup.get(scene_id, int(g0['scene_code'].iloc[0]) if 'scene_code' in g0.columns else -1)
            groups.append((scene_id, int(scene_code), g0, scene_map.loc[scene_id].to_dict()))

        if fusion_workers <= 1 or len(groups) <= 1:
            results = [
                sample_scene_group(scene_id, scene_code, g0, row, band_indices, index_local_indices, center_width, center_height, center_policy, fusion_enable_memmap_cache)
                for scene_id, scene_code, g0, row in groups
            ]
        else:
            results = []
            executor = persistent_fusion_executor if persistent_fusion_executor is not None else ThreadPoolExecutor(max_workers=fusion_workers)
            futures = [
                executor.submit(sample_scene_group, scene_id, scene_code, g0, row, band_indices, index_local_indices, center_width, center_height, center_policy, fusion_enable_memmap_cache)
                for scene_id, scene_code, g0, row in groups
            ]
            for future in as_completed(futures):
                results.append(future.result())
            if persistent_fusion_executor is None:
                executor.shutdown(wait=True)

        local_sampled = 0
        for result in results:
            pid = result['pid']
            n = int(pid.size)
            summary = scene_summary[result['scene_id']]
            summary['outside_center_window_count'] += int(result['outside_center_window_count'])
            outside_center_window_count += int(result['outside_center_window_count'])
            if n == 0:
                continue

            weighted_spectra = result['weighted_spectra']
            for band_idx in range(weighted_spectra.shape[1]):
                np.add.at(fused_sum[:, band_idx], pid, weighted_spectra[:, band_idx])

            w = result['w']
            range_m = result['range_m']
            offaxis_deg = result['offaxis_deg']
            border_px = result['border_px']
            empty_cone = result['empty_cone']
            view_zenith = result['view_zenith']
            surface_view_cos = result['surface_view_cos']
            surface_verticality = result['surface_verticality']
            is_clear = result['is_clear']

            np.add.at(weight_sum, pid, w)
            np.add.at(clear_weight_sum, pid, w * is_clear.astype(np.float64))
            np.add.at(weighted_range_sum, pid, w * range_m)
            np.add.at(weighted_offaxis_sum, pid, w * offaxis_deg)
            np.add.at(weighted_border_sum, pid, w * border_px)
            np.add.at(weighted_empty_cone_sum, pid, w * empty_cone)
            np.add.at(weighted_view_zenith_sum, pid, w * view_zenith)
            np.add.at(weighted_surface_view_cos_sum, pid, w * surface_view_cos)
            np.add.at(weighted_surface_verticality_sum, pid, w * surface_verticality)
            np.add.at(contribution_counter, pid, 1)

            take_primary = w > primary_view_weight[pid]
            if np.any(take_primary):
                pid_take = pid[take_primary]
                primary_scene_code[pid_take] = int(result['scene_code'])
                primary_view_weight[pid_take] = w[take_primary]
                primary_range[pid_take] = range_m[take_primary]
                primary_offaxis[pid_take] = offaxis_deg[take_primary]
                primary_border[pid_take] = border_px[take_primary]
                primary_empty_cone[pid_take] = empty_cone[take_primary]
                primary_view_zenith[pid_take] = view_zenith[take_primary]
                primary_surface_view_cos[pid_take] = surface_view_cos[take_primary]

            summary['fused_observation_count'] += n
            summary['contributing_point_count'] += int(np.unique(pid).size)
            summary['weight_sum_in_scene'] += float(np.sum(w))
            local_sampled += n
        return int(len(vis_chunk)), local_sampled, len(groups)

    for chunk in reader:
        chunk = coerce_visible_chunk(chunk)
        ready, carry = split_complete_points(chunk, carry)
        chunk_index += 1
        done, sampled, scene_count = process_chunk(ready)
        processed_rows += done
        sampled_rows += sampled
        if (not quiet) or (chunk_index % 10 == 0):
            print(f'fusion chunk {chunk_index}: processed_rows={processed_rows}, sampled_rows={sampled_rows}, scenes={scene_count}')

    if carry is not None and len(carry) > 0:
        chunk_index += 1
        done, sampled, scene_count = process_chunk(carry)
        processed_rows += done
        sampled_rows += sampled
        print(f'fusion chunk {chunk_index}: processed_rows={processed_rows}, sampled_rows={sampled_rows}, scenes={scene_count}')
    elif quiet:
        print(f'fusion rows processed: {processed_rows}, sampled_rows={sampled_rows}')

    if persistent_fusion_executor is not None:
        persistent_fusion_executor.shutdown(wait=True)

    contributing_view_count = np.clip(contribution_counter, 0, np.iinfo(np.int16).max).astype(np.int16)
    fused = np.full((n_points, len(band_indices)), np.nan, dtype=np.float32)
    valid = weight_sum > 0
    fused[valid] = (fused_sum[valid] / weight_sum[valid, None]).astype(np.float32)

    clear_view_ratio = np.full(n_points, np.nan, dtype=np.float32)
    weighted_mean_range = np.full(n_points, np.nan, dtype=np.float32)
    weighted_mean_offaxis = np.full(n_points, np.nan, dtype=np.float32)
    weighted_mean_border = np.full(n_points, np.nan, dtype=np.float32)
    weighted_mean_empty_cone = np.full(n_points, np.nan, dtype=np.float32)
    weighted_mean_view_zenith = np.full(n_points, np.nan, dtype=np.float32)
    weighted_mean_surface_view_cos = np.full(n_points, np.nan, dtype=np.float32)
    weighted_mean_surface_verticality = np.full(n_points, np.nan, dtype=np.float32)
    primary_weight_ratio = np.full(n_points, np.nan, dtype=np.float32)
    view_weight_entropy = np.full(n_points, np.nan, dtype=np.float32)
    effective_view_count = np.full(n_points, np.nan, dtype=np.float32)

    clear_view_ratio[valid] = (clear_weight_sum[valid] / weight_sum[valid]).astype(np.float32)
    weighted_mean_range[valid] = (weighted_range_sum[valid] / weight_sum[valid]).astype(np.float32)
    weighted_mean_offaxis[valid] = (weighted_offaxis_sum[valid] / weight_sum[valid]).astype(np.float32)
    weighted_mean_border[valid] = (weighted_border_sum[valid] / weight_sum[valid]).astype(np.float32)
    weighted_mean_empty_cone[valid] = (weighted_empty_cone_sum[valid] / weight_sum[valid]).astype(np.float32)
    weighted_mean_view_zenith[valid] = (weighted_view_zenith_sum[valid] / weight_sum[valid]).astype(np.float32)
    weighted_mean_surface_view_cos[valid] = (weighted_surface_view_cos_sum[valid] / weight_sum[valid]).astype(np.float32)
    weighted_mean_surface_verticality[valid] = (weighted_surface_verticality_sum[valid] / weight_sum[valid]).astype(np.float32)
    primary_weight_ratio[valid] = (primary_view_weight[valid] / weight_sum[valid]).astype(np.float32)
    view_weight_entropy[valid] = entropy_acc[valid].astype(np.float32)
    effective_view_count[valid] = np.exp(entropy_acc[valid]).astype(np.float32)

    derived_arrays = compute_derived_reflectance_indices(fused=fused, band_numbers=band_numbers)

    np.save(out_dir / 'fused_point_spectra.npy', fused)
    np.save(out_dir / 'fused_band_indices.npy', np.asarray(band_indices, dtype=np.int32))
    np.save(out_dir / 'fused_band_numbers.npy', band_numbers)
    np.save(out_dir / 'fused_wavelengths_used.npy', wavelengths_used)
    np.save(out_dir / 'primary_scene_code.npy', primary_scene_code)
    np.save(out_dir / 'contributing_view_count.npy', contributing_view_count)
    np.save(out_dir / 'fused_weight_sum.npy', weight_sum.astype(np.float32))
    np.save(out_dir / 'primary_view_weight.npy', primary_view_weight.astype(np.float32))
    np.save(out_dir / 'primary_weight_ratio.npy', primary_weight_ratio)
    np.save(out_dir / 'clear_view_ratio.npy', clear_view_ratio)
    np.save(out_dir / 'weighted_mean_range_m.npy', weighted_mean_range)
    np.save(out_dir / 'weighted_mean_offaxis_deg.npy', weighted_mean_offaxis)
    np.save(out_dir / 'weighted_mean_border_dist_px.npy', weighted_mean_border)
    np.save(out_dir / 'weighted_mean_empty_cone_deg.npy', weighted_mean_empty_cone)
    np.save(out_dir / 'weighted_mean_view_zenith_deg.npy', weighted_mean_view_zenith)
    np.save(out_dir / 'weighted_mean_surface_view_cos.npy', weighted_mean_surface_view_cos)
    np.save(out_dir / 'weighted_mean_surface_verticality.npy', weighted_mean_surface_verticality)
    np.save(out_dir / 'primary_range_m.npy', primary_range.astype(np.float32))
    np.save(out_dir / 'primary_offaxis_deg.npy', primary_offaxis.astype(np.float32))
    np.save(out_dir / 'primary_border_dist_px.npy', primary_border.astype(np.float32))
    np.save(out_dir / 'primary_empty_cone_deg.npy', primary_empty_cone.astype(np.float32))
    np.save(out_dir / 'primary_view_zenith_deg.npy', primary_view_zenith.astype(np.float32))
    np.save(out_dir / 'primary_surface_view_cos.npy', primary_surface_view_cos.astype(np.float32))
    np.save(out_dir / 'view_weight_entropy.npy', view_weight_entropy)
    np.save(out_dir / 'effective_view_count.npy', effective_view_count)
    for name, arr in derived_arrays.items():
        np.save(out_dir / f'{name}.npy', arr)

    scene_summary_rows = []
    for scene_id, values in scene_summary.items():
        scene_summary_rows.append({
            'scene_id': str(scene_id),
            'fused_observation_count': int(values['fused_observation_count']),
            'contributing_point_count': int(values['contributing_point_count']),
            'weight_sum_in_scene': float(values['weight_sum_in_scene']),
            'outside_center_window_count': int(values['outside_center_window_count']),
        })
    pd.DataFrame(scene_summary_rows).to_csv(out_dir / 'fusion_scene_summary.csv', index=False, encoding='utf-8-sig')

    if write_point_summary:
        primary_scene_ids = np.array([''] * n_points, dtype=object)
        inv_scene_lookup = {int(v): k for k, v in scene_code_lookup.items()}
        valid_primary = primary_scene_code >= 0
        if np.any(valid_primary):
            primary_scene_ids[valid_primary] = [inv_scene_lookup.get(int(code), '') for code in primary_scene_code[valid_primary]]
        point_summary = pd.DataFrame({
            'point_id': np.arange(n_points, dtype=np.int64),
            'has_fused_spectrum': valid,
            'contributing_view_count': contributing_view_count,
            'primary_scene_code': primary_scene_code,
            'primary_scene_id': primary_scene_ids,
            'primary_weight_ratio': primary_weight_ratio,
            'clear_view_ratio': clear_view_ratio,
            'weighted_mean_range_m': weighted_mean_range,
            'weighted_mean_offaxis_deg': weighted_mean_offaxis,
            'weighted_mean_border_dist_px': weighted_mean_border,
            'weighted_mean_empty_cone_deg': weighted_mean_empty_cone,
            'weighted_mean_view_zenith_deg': weighted_mean_view_zenith,
            'weighted_mean_surface_view_cos': weighted_mean_surface_view_cos,
            'weighted_mean_surface_verticality': weighted_mean_surface_verticality,
            'view_weight_entropy': view_weight_entropy,
            'effective_view_count': effective_view_count,
        })
        point_summary.to_csv(point_summary_path, index=False, encoding='utf-8-sig')

    save_json({
        'point_count': int(n_points),
        'visible_observation_rows': int(processed_rows),
        'sampled_observation_rows': int(sampled_rows),
        'valid_spectrum_count': int(np.count_nonzero(valid)),
        'band_count': int(len(band_indices)),
        'fusion_workers': int(fusion_workers),
        'fusion_enable_memmap_cache': bool(fusion_enable_memmap_cache),
        'fusion_use_persistent_executor': bool(fusion_use_persistent_executor),
        'fusion_csv_chunk_rows': int(chunk_rows),
        'write_fused_observation_table': bool(write_obs_table),
        'write_fused_point_summary': bool(write_point_summary),
        'normalize_weights_per_point': True,
        'fusion_weight_mode': mode,
        'fusion_softmax_temperature': float(softmax_temp),
        'fusion_raw_weight_power': float(raw_power),
        'surface_adaptive_fusion_params': dict(proc.get('surface_adaptive_fusion_params', {})),
        'index_band_numbers_stored_plus_one': index_plus_one_bands,
        'derived_reflectance_indices': sorted(derived_arrays.keys()),
        'mean_contributing_view_count_valid': float(np.mean(contributing_view_count[valid])) if np.any(valid) else 0.0,
        'mean_primary_weight_ratio_valid': float(np.nanmean(primary_weight_ratio[valid])) if np.any(valid) else float('nan'),
        'mean_clear_view_ratio_valid': float(np.nanmean(clear_view_ratio[valid])) if np.any(valid) else float('nan'),
        'mean_weighted_view_zenith_deg_valid': float(np.nanmean(weighted_mean_view_zenith[valid])) if np.any(valid) else float('nan'),
        'mean_weighted_surface_view_cos_valid': float(np.nanmean(weighted_mean_surface_view_cos[valid])) if np.any(valid) else float('nan'),
        'center_window_policy': center_policy,
        'center_window_width_px': None if center_width is None else int(center_width),
        'center_window_height_px': None if center_height is None else int(center_height),
        'outside_center_window_count': int(outside_center_window_count),
    }, out_dir / 'fusion_summary.json')

    print(f'Valid fused points: {np.count_nonzero(valid)}')
    print(f'Mean contributing views per valid point: {float(np.mean(contributing_view_count[valid])) if np.any(valid) else 0.0:.3f}')


if __name__ == '__main__':
    main()
