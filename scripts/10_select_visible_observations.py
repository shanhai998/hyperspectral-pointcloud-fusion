
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from hyperspectral_pointcloud_fusion.common import load_config, ensure_dir, save_json, append_table, read_table, is_quiet
from hyperspectral_pointcloud_fusion.visibility import (
    evaluate_candidate_direction,
    build_selection_score,
    build_multiview_weight,
    build_selection_score_vec,
    build_multiview_weight_vec,
    evaluate_candidate_directions_batch,
)


G_XYZ = None
G_TREE = None

G_SCENE_CODE = None
G_U = None
G_V = None
G_RANGE = None
G_OFFAXIS = None
G_BORDER = None
G_SCORE = None
G_DIRX = None
G_DIRY = None
G_DIRZ = None
G_VIEW_ZENITH = None
G_VIEW_AZIMUTH = None
G_LOCAL_VIEW_COS_SIGNED = None
G_LOCAL_VIEW_ANGLE = None
G_SOLAR_ZENITH = None
G_SOLAR_AZIMUTH = None
G_RELATIVE_AZIMUTH = None
G_SUN_DIRX = None
G_SUN_DIRY = None
G_SUN_DIRZ = None
G_LOCAL_SOLAR_COS = None
G_LOCAL_SOLAR_INCIDENCE = None
G_IS_BACKLIT = None
G_SURFACE_VIEW_COS = None
G_SURFACE_VERTICALITY = None
G_NORMAL_CONFIDENCE = None
G_IMAGE_ZBUFFER_CLEAR = None

G_RADIUS = None
G_CLEARANCE = None
G_MIN_BLOCKER_ALONG = None
G_MIN_EMPTY_CONE_CLEAR_DEG = None
G_MIN_NEIGHBORS = None
G_SCORE_WEIGHTS = None
G_WEIGHT_PARAMS = None
G_SCENE_LIMIT = None
G_MAX_NEIGHBORS = None
G_NEIGHBOR_QUERY_MODE = None
G_KD_LEAFSIZE = None
G_MAX_VISIBLE_VIEWS = None
G_MAX_FALLBACK_VIEWS = None
G_KEEP_FALLBACK = None
G_FALLBACK_WEIGHT_PENALTY = None
G_KEEP_UNCLEAR_WITH_CLEAR = None
G_WORKER_SHARD_DIR = None
G_WORKER_SHARD_PATH = None
G_WORKER_ID = None

CANDIDATE_CACHE_KEYS = [
    'score', 'scene_code', 'u', 'v', 'range_m', 'offaxis_deg',
    'border_dist_px', 'dir_x', 'dir_y', 'dir_z', 'view_zenith_deg',
    'view_azimuth_deg', 'local_view_cos_signed', 'local_view_angle_deg',
    'solar_zenith_deg', 'solar_azimuth_deg', 'relative_azimuth_deg',
    'sun_dir_x', 'sun_dir_y', 'sun_dir_z', 'local_solar_cos',
    'local_solar_incidence_deg', 'is_backlit',
    'surface_view_cos', 'surface_verticality', 'normal_confidence',
]


TABLE_COLUMNS = [
    'point_id', 'scene_code', 'u', 'v', 'range_m', 'offaxis_deg', 'border_dist_px',
    'view_zenith_deg', 'view_azimuth_deg', 'local_view_cos_signed',
    'local_view_angle_deg', 'solar_zenith_deg', 'solar_azimuth_deg',
    'relative_azimuth_deg', 'sun_dir_x', 'sun_dir_y', 'sun_dir_z',
    'local_solar_cos', 'local_solar_incidence_deg', 'is_backlit',
    'surface_view_cos', 'surface_verticality', 'normal_confidence',
    'coarse_score', 'visibility_score', 'view_weight_raw', 'local_free_dir_x',
    'local_free_dir_y', 'local_free_dir_z', 'local_empty_cone_deg', 'blocker_count',
    'front_neighbor_count', 'local_neighbor_count', 'image_zbuffer_clear', 'clear_reject_reason', 'is_clear', 'selection_method',
    'keep_for_fusion', 'fusion_rank_within_point',
]


def _log_stage(message: str) -> None:
    print(message, flush=True)


def _log_done(label: str, t0: float) -> None:
    _log_stage(f'{label} done in {time.perf_counter() - t0:.1f}s')


def _chunk_array(arr: np.ndarray, chunk_size: int):
    n = len(arr)
    for i in range(0, n, chunk_size):
        yield arr[i:i + chunk_size]


def _default_clear_eval():
    return {
        'is_clear': True,
        'blocker_count': 0,
        'front_count': 0,
        'empty_cone_deg': 180.0,
        'nearest_blocker_along_m': np.inf,
        'nearest_blocker_perp_m': np.inf,
        'max_front_cos': -1.0,
    }


def _load_existing_candidate_memmap_cache(cache_dir: Path) -> dict | None:
    manifest_path = cache_dir / 'cache_manifest.json'
    if not manifest_path.exists():
        return None
    with open(manifest_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    for key in CANDIDATE_CACHE_KEYS:
        if key not in meta or not (cache_dir / f'{key}.npy').exists():
            return None
    score_shape = tuple(meta.get('score', {}).get('shape', []))
    if len(score_shape) < 2:
        return None
    meta['point_count'] = int(meta.get('point_count', score_shape[0]))
    meta['top_k'] = int(meta.get('top_k', score_shape[1]))
    return meta


def _prepare_candidate_memmap_cache(cand_npz_path: Path, cache_dir: Path, rebuild: bool = False) -> dict:
    ensure_dir(cache_dir)
    manifest_path = cache_dir / 'cache_manifest.json'

    if rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)
        ensure_dir(cache_dir)

    if not cand_npz_path.exists():
        raise FileNotFoundError(f'候选 npz 不存在，且未找到直接 memmap 缓存: {cand_npz_path}')

    source_mtime_ns = int(cand_npz_path.stat().st_mtime_ns)
    source_size = int(cand_npz_path.stat().st_size)

    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        ok = True
        for key in CANDIDATE_CACHE_KEYS:
            if not (cache_dir / f'{key}.npy').exists():
                ok = False
                break
        if ok and int(meta.get('source_npz_mtime_ns', -1)) == source_mtime_ns and int(meta.get('source_npz_size', -1)) == source_size:
            return meta

    print('Preparing candidate memmap cache for step 06 ...')
    cand = np.load(cand_npz_path)
    meta = {}
    for key in CANDIDATE_CACHE_KEYS:
        arr = np.asarray(cand[key])
        np.save(cache_dir / f'{key}.npy', arr)
        meta[key] = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}

    score_shape = tuple(meta['score']['shape'])
    meta['point_count'] = int(score_shape[0])
    meta['top_k'] = int(score_shape[1]) if len(score_shape) >= 2 else 1
    meta['source_npz_mtime_ns'] = source_mtime_ns
    meta['source_npz_size'] = source_size
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'Candidate memmap cache ready: {cache_dir}')
    return meta


def _find_candidate_points(scene_code_topk: np.ndarray, eval_scene_limit: int, chunk_rows: int) -> np.ndarray:
    chunks = []
    total_points = int(scene_code_topk.shape[0])
    for i0 in range(0, total_points, max(1, int(chunk_rows))):
        i1 = min(total_points, i0 + max(1, int(chunk_rows)))
        block = np.asarray(scene_code_topk[i0:i1, :eval_scene_limit])
        keep = np.any(block >= 0, axis=1)
        if np.any(keep):
            chunks.append(np.flatnonzero(keep).astype(np.int64) + i0)
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def _prepare_image_zbuffer_clear(
    cache_dir: Path,
    total_points: int,
    top_k: int,
    image_width: int,
    image_height: int,
    radius_px: int,
    depth_tolerance_m: float,
    rebuild: bool = False,
) -> Path:
    out_path = cache_dir / 'image_zbuffer_clear.npy'
    meta_path = cache_dir / 'image_zbuffer_clear_meta.json'
    wanted_meta = {
        'total_points': int(total_points),
        'top_k': int(top_k),
        'image_width': int(image_width),
        'image_height': int(image_height),
        'radius_px': int(radius_px),
        'depth_tolerance_m': float(depth_tolerance_m),
    }
    if out_path.exists() and meta_path.exists() and not rebuild:
        with open(meta_path, 'r', encoding='utf-8') as f:
            old_meta = json.load(f)
        if all(old_meta.get(k) == v for k, v in wanted_meta.items()):
            return out_path

    scene_code = np.load(cache_dir / 'scene_code.npy', mmap_mode='r')
    u = np.load(cache_dir / 'u.npy', mmap_mode='r')
    v = np.load(cache_dir / 'v.npy', mmap_mode='r')
    range_m = np.load(cache_dir / 'range_m.npy', mmap_mode='r')
    clear = np.zeros((total_points, top_k), dtype=bool)

    valid_scene_codes = np.unique(np.asarray(scene_code[scene_code >= 0], dtype=np.int16))
    print(f'Preparing image z-buffer occlusion masks for {len(valid_scene_codes)} scenes ...')
    radius_px = max(0, int(radius_px))
    filter_size = int(radius_px * 2 + 1)
    for idx, code in enumerate(valid_scene_codes, start=1):
        mask = np.asarray(scene_code == code)
        if not np.any(mask):
            continue
        point_idx, slot_idx = np.nonzero(mask)
        uu = np.rint(np.asarray(u[mask], dtype=np.float64)).astype(np.int32)
        vv = np.rint(np.asarray(v[mask], dtype=np.float64)).astype(np.int32)
        rr = np.asarray(range_m[mask], dtype=np.float32)
        inside = (
            np.isfinite(rr) &
            (uu >= 0) & (uu < image_width) &
            (vv >= 0) & (vv < image_height)
        )
        if not np.any(inside):
            continue
        uu = uu[inside]
        vv = vv[inside]
        rr = rr[inside]
        point_idx = point_idx[inside]
        slot_idx = slot_idx[inside]

        zbuf = np.full((image_height, image_width), np.inf, dtype=np.float32)
        np.minimum.at(zbuf, (vv, uu), rr)
        if radius_px > 0:
            zbuf = minimum_filter(zbuf, size=filter_size, mode='constant', cval=np.inf)
        nearest = zbuf[vv, uu]
        clear[point_idx, slot_idx] = rr <= (nearest + float(depth_tolerance_m))
        print(f'image z-buffer {idx}/{len(valid_scene_codes)}: scene_code={int(code)}, candidates={int(rr.size)}')

    np.save(out_path, clear)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(wanted_meta, f, ensure_ascii=False, indent=2)
    return out_path


def _init_worker(
    xyz_path: str,
    cache_dir: str,
    image_zbuffer_clear_path: str,
    radius: float,
    clearance: float,
    min_blocker_along: float,
    min_empty_cone_clear_deg: float,
    min_neighbors: int,
    score_weights: dict,
    weight_params: dict,
    scene_limit: int,
    max_neighbors: int,
    neighbor_query_mode: str,
    kd_leafsize: int,
    max_visible_views: int,
    max_fallback_views: int,
    keep_fallback: bool,
    fallback_weight_penalty: float,
    keep_unclear_with_clear: bool,
    kd_balanced_tree: bool = False,
    kd_compact_nodes: bool = False,
    worker_shard_dir: str = '',
):
    global G_XYZ, G_TREE
    global G_SCENE_CODE, G_U, G_V, G_RANGE, G_OFFAXIS, G_BORDER, G_SCORE, G_DIRX, G_DIRY, G_DIRZ
    global G_VIEW_ZENITH, G_VIEW_AZIMUTH, G_LOCAL_VIEW_COS_SIGNED, G_LOCAL_VIEW_ANGLE
    global G_SOLAR_ZENITH, G_SOLAR_AZIMUTH, G_RELATIVE_AZIMUTH, G_SUN_DIRX, G_SUN_DIRY, G_SUN_DIRZ
    global G_LOCAL_SOLAR_COS, G_LOCAL_SOLAR_INCIDENCE, G_IS_BACKLIT
    global G_SURFACE_VIEW_COS, G_SURFACE_VERTICALITY, G_NORMAL_CONFIDENCE, G_IMAGE_ZBUFFER_CLEAR
    global G_RADIUS, G_CLEARANCE, G_MIN_BLOCKER_ALONG, G_MIN_EMPTY_CONE_CLEAR_DEG, G_MIN_NEIGHBORS, G_SCORE_WEIGHTS, G_WEIGHT_PARAMS, G_SCENE_LIMIT, G_MAX_NEIGHBORS, G_NEIGHBOR_QUERY_MODE, G_KD_LEAFSIZE
    global G_MAX_VISIBLE_VIEWS, G_MAX_FALLBACK_VIEWS, G_KEEP_FALLBACK, G_FALLBACK_WEIGHT_PENALTY, G_KEEP_UNCLEAR_WITH_CLEAR
    global G_WORKER_SHARD_DIR, G_WORKER_SHARD_PATH, G_WORKER_ID

    xyz_path = Path(xyz_path)
    cache_dir = Path(cache_dir)

    G_XYZ = np.load(xyz_path, mmap_mode='r')
    G_SCENE_CODE = np.load(cache_dir / 'scene_code.npy', mmap_mode='r')
    G_U = np.load(cache_dir / 'u.npy', mmap_mode='r')
    G_V = np.load(cache_dir / 'v.npy', mmap_mode='r')
    G_RANGE = np.load(cache_dir / 'range_m.npy', mmap_mode='r')
    G_OFFAXIS = np.load(cache_dir / 'offaxis_deg.npy', mmap_mode='r')
    G_BORDER = np.load(cache_dir / 'border_dist_px.npy', mmap_mode='r')
    G_SCORE = np.load(cache_dir / 'score.npy', mmap_mode='r')
    G_DIRX = np.load(cache_dir / 'dir_x.npy', mmap_mode='r')
    G_DIRY = np.load(cache_dir / 'dir_y.npy', mmap_mode='r')
    G_DIRZ = np.load(cache_dir / 'dir_z.npy', mmap_mode='r')
    G_VIEW_ZENITH = np.load(cache_dir / 'view_zenith_deg.npy', mmap_mode='r')
    G_VIEW_AZIMUTH = np.load(cache_dir / 'view_azimuth_deg.npy', mmap_mode='r')
    G_LOCAL_VIEW_COS_SIGNED = np.load(cache_dir / 'local_view_cos_signed.npy', mmap_mode='r')
    G_LOCAL_VIEW_ANGLE = np.load(cache_dir / 'local_view_angle_deg.npy', mmap_mode='r')
    G_SOLAR_ZENITH = np.load(cache_dir / 'solar_zenith_deg.npy', mmap_mode='r')
    G_SOLAR_AZIMUTH = np.load(cache_dir / 'solar_azimuth_deg.npy', mmap_mode='r')
    G_RELATIVE_AZIMUTH = np.load(cache_dir / 'relative_azimuth_deg.npy', mmap_mode='r')
    G_SUN_DIRX = np.load(cache_dir / 'sun_dir_x.npy', mmap_mode='r')
    G_SUN_DIRY = np.load(cache_dir / 'sun_dir_y.npy', mmap_mode='r')
    G_SUN_DIRZ = np.load(cache_dir / 'sun_dir_z.npy', mmap_mode='r')
    G_LOCAL_SOLAR_COS = np.load(cache_dir / 'local_solar_cos.npy', mmap_mode='r')
    G_LOCAL_SOLAR_INCIDENCE = np.load(cache_dir / 'local_solar_incidence_deg.npy', mmap_mode='r')
    G_IS_BACKLIT = np.load(cache_dir / 'is_backlit.npy', mmap_mode='r')
    G_SURFACE_VIEW_COS = np.load(cache_dir / 'surface_view_cos.npy', mmap_mode='r')
    G_SURFACE_VERTICALITY = np.load(cache_dir / 'surface_verticality.npy', mmap_mode='r')
    G_NORMAL_CONFIDENCE = np.load(cache_dir / 'normal_confidence.npy', mmap_mode='r')
    G_IMAGE_ZBUFFER_CLEAR = np.load(image_zbuffer_clear_path, mmap_mode='r') if image_zbuffer_clear_path else None

    G_RADIUS = float(radius)
    G_CLEARANCE = float(clearance)
    G_MIN_BLOCKER_ALONG = float(min_blocker_along)
    G_MIN_EMPTY_CONE_CLEAR_DEG = float(min_empty_cone_clear_deg)
    G_MIN_NEIGHBORS = int(min_neighbors)
    G_SCORE_WEIGHTS = dict(score_weights)
    G_WEIGHT_PARAMS = dict(weight_params)
    G_SCENE_LIMIT = int(scene_limit)
    G_MAX_NEIGHBORS = int(max_neighbors)
    G_NEIGHBOR_QUERY_MODE = str(neighbor_query_mode or 'bounded_knn').strip().lower()
    G_KD_LEAFSIZE = int(kd_leafsize)
    G_MAX_VISIBLE_VIEWS = int(max_visible_views)
    G_MAX_FALLBACK_VIEWS = int(max_fallback_views)
    G_KEEP_FALLBACK = bool(keep_fallback)
    G_FALLBACK_WEIGHT_PENALTY = float(fallback_weight_penalty)
    G_KEEP_UNCLEAR_WITH_CLEAR = bool(keep_unclear_with_clear)


    G_WORKER_SHARD_DIR = str(worker_shard_dir) if worker_shard_dir else ''
    G_WORKER_ID = f"{os.getpid()}_{id(G_XYZ) & 0xffffff:06x}"
    G_WORKER_SHARD_PATH = (
        str(Path(G_WORKER_SHARD_DIR) / f'shard_{G_WORKER_ID}.csv')
        if G_WORKER_SHARD_DIR
        else ''
    )

    G_TREE = cKDTree(
        np.asarray(G_XYZ, dtype=np.float64),
        leafsize=G_KD_LEAFSIZE,
        compact_nodes=bool(kd_compact_nodes),
        balanced_tree=bool(kd_balanced_tree),
        copy_data=False,
    )


def _process_point_chunk(point_ids: np.ndarray, shard_path_override: str = '') -> dict:


    point_ids = np.asarray(point_ids, dtype=np.int64)
    n_points = int(point_ids.size)
    if n_points == 0:
        return {
            'point_ids': point_ids,
            'visible_count_arr': np.zeros(0, dtype=np.int16),
            'clear_count_arr': np.zeros(0, dtype=np.int16),
            'kept_count_arr': np.zeros(0, dtype=np.int16),
            'used_fallback_arr': np.zeros(0, dtype=bool),
            'emitted_rows': 0,
            'shard_path': shard_path_override or G_WORKER_SHARD_PATH or '',
            'table_df': None,
        }

    query_xyz = np.asarray(G_XYZ[point_ids], dtype=np.float64)
    neighbor_idx_block = None
    neigh_lists = None
    if G_MAX_NEIGHBORS > 0 and G_NEIGHBOR_QUERY_MODE in {'bounded_knn', 'knn', 'query'}:


        k_neighbors = max(2, int(G_MAX_NEIGHBORS) + 1)
        _dist_block, neighbor_idx_block = G_TREE.query(
            query_xyz,
            k=k_neighbors,
            distance_upper_bound=G_RADIUS,
        )
        if neighbor_idx_block.ndim == 1:
            neighbor_idx_block = neighbor_idx_block[:, None]
    else:
        neigh_lists = G_TREE.query_ball_point(query_xyz, r=G_RADIUS)

    S = int(G_SCENE_LIMIT)


    scene_code_block = np.asarray(G_SCENE_CODE[point_ids, :S], dtype=np.int16)
    u_block = np.asarray(G_U[point_ids, :S], dtype=np.float32)
    v_block = np.asarray(G_V[point_ids, :S], dtype=np.float32)
    range_block = np.asarray(G_RANGE[point_ids, :S], dtype=np.float32)
    offaxis_block = np.asarray(G_OFFAXIS[point_ids, :S], dtype=np.float32)
    border_block = np.asarray(G_BORDER[point_ids, :S], dtype=np.float32)
    score_block = np.asarray(G_SCORE[point_ids, :S], dtype=np.float32)
    dirx_block = np.asarray(G_DIRX[point_ids, :S], dtype=np.float32)
    diry_block = np.asarray(G_DIRY[point_ids, :S], dtype=np.float32)
    dirz_block = np.asarray(G_DIRZ[point_ids, :S], dtype=np.float32)
    view_zenith_block = np.asarray(G_VIEW_ZENITH[point_ids, :S], dtype=np.float32)
    view_azimuth_block = np.asarray(G_VIEW_AZIMUTH[point_ids, :S], dtype=np.float32)
    local_view_cos_signed_block = np.asarray(G_LOCAL_VIEW_COS_SIGNED[point_ids, :S], dtype=np.float32)
    local_view_angle_block = np.asarray(G_LOCAL_VIEW_ANGLE[point_ids, :S], dtype=np.float32)
    solar_zenith_block = np.asarray(G_SOLAR_ZENITH[point_ids, :S], dtype=np.float32)
    solar_azimuth_block = np.asarray(G_SOLAR_AZIMUTH[point_ids, :S], dtype=np.float32)
    relative_azimuth_block = np.asarray(G_RELATIVE_AZIMUTH[point_ids, :S], dtype=np.float32)
    sun_dirx_block = np.asarray(G_SUN_DIRX[point_ids, :S], dtype=np.float32)
    sun_diry_block = np.asarray(G_SUN_DIRY[point_ids, :S], dtype=np.float32)
    sun_dirz_block = np.asarray(G_SUN_DIRZ[point_ids, :S], dtype=np.float32)
    local_solar_cos_block = np.asarray(G_LOCAL_SOLAR_COS[point_ids, :S], dtype=np.float32)
    local_solar_incidence_block = np.asarray(G_LOCAL_SOLAR_INCIDENCE[point_ids, :S], dtype=np.float32)
    is_backlit_block = np.asarray(G_IS_BACKLIT[point_ids, :S], dtype=np.int8)
    surface_view_cos_block = np.asarray(G_SURFACE_VIEW_COS[point_ids, :S], dtype=np.float32)
    surface_verticality_block = np.asarray(G_SURFACE_VERTICALITY[point_ids, :S], dtype=np.float32)
    normal_confidence_block = np.asarray(G_NORMAL_CONFIDENCE[point_ids, :S], dtype=np.float32)
    if G_IMAGE_ZBUFFER_CLEAR is not None:
        image_zbuffer_clear_block = np.asarray(G_IMAGE_ZBUFFER_CLEAR[point_ids, :S], dtype=bool)
    else:
        image_zbuffer_clear_block = np.ones((n_points, S), dtype=bool)

    visible_count_arr = np.zeros(n_points, dtype=np.int16)
    clear_count_arr = np.zeros(n_points, dtype=np.int16)
    kept_count_arr = np.zeros(n_points, dtype=np.int16)
    used_fallback_arr = np.zeros(n_points, dtype=bool)


    out_point_ids: list[np.ndarray] = []
    out_scene_code: list[np.ndarray] = []
    out_u: list[np.ndarray] = []
    out_v: list[np.ndarray] = []
    out_range: list[np.ndarray] = []
    out_offaxis: list[np.ndarray] = []
    out_border: list[np.ndarray] = []
    out_view_zenith: list[np.ndarray] = []
    out_view_azimuth: list[np.ndarray] = []
    out_local_view_cos_signed: list[np.ndarray] = []
    out_local_view_angle: list[np.ndarray] = []
    out_solar_zenith: list[np.ndarray] = []
    out_solar_azimuth: list[np.ndarray] = []
    out_relative_azimuth: list[np.ndarray] = []
    out_sun_dirx: list[np.ndarray] = []
    out_sun_diry: list[np.ndarray] = []
    out_sun_dirz: list[np.ndarray] = []
    out_local_solar_cos: list[np.ndarray] = []
    out_local_solar_incidence: list[np.ndarray] = []
    out_is_backlit: list[np.ndarray] = []
    out_surface_view_cos: list[np.ndarray] = []
    out_surface_verticality: list[np.ndarray] = []
    out_normal_confidence: list[np.ndarray] = []
    out_coarse: list[np.ndarray] = []
    out_visibility_score: list[np.ndarray] = []
    out_view_weight_raw: list[np.ndarray] = []
    out_dirx: list[np.ndarray] = []
    out_diry: list[np.ndarray] = []
    out_dirz: list[np.ndarray] = []
    out_empty_cone: list[np.ndarray] = []
    out_blocker: list[np.ndarray] = []
    out_front_count: list[np.ndarray] = []
    out_local_neigh: list[np.ndarray] = []
    out_image_zbuf: list[np.ndarray] = []
    out_reject_reason: list[list[str]] = []
    out_is_clear: list[np.ndarray] = []
    out_selection_method: list[list[str]] = []
    out_fusion_rank: list[np.ndarray] = []

    min_cone = float(G_MIN_EMPTY_CONE_CLEAR_DEG)
    clearance_m = float(G_CLEARANCE)
    min_blocker_along_m = float(G_MIN_BLOCKER_ALONG)
    max_visible = max(1, int(G_MAX_VISIBLE_VIEWS))
    max_fallback = max(1, int(G_MAX_FALLBACK_VIEWS))

    xyz_pid = np.asarray(G_XYZ[point_ids], dtype=np.float64)

    for local_idx in range(n_points):
        row_scene_code = scene_code_block[local_idx]
        slots = np.flatnonzero(row_scene_code >= 0)
        if slots.size == 0:
            continue


        pid = int(point_ids[local_idx])
        if neighbor_idx_block is not None:
            neigh_idx = np.asarray(neighbor_idx_block[local_idx], dtype=np.int64)
            neigh_idx = neigh_idx[(neigh_idx >= 0) & (neigh_idx < G_XYZ.shape[0]) & (neigh_idx != pid)]
        else:
            neigh = neigh_lists[local_idx]
            neigh_idx = np.asarray(neigh, dtype=np.int64)
            neigh_idx = neigh_idx[neigh_idx != pid]

        if neigh_idx.size > 0:
            offsets = np.asarray(G_XYZ[neigh_idx], dtype=np.float64) - xyz_pid[local_idx][None, :]
            if neighbor_idx_block is None and G_MAX_NEIGHBORS > 0 and offsets.shape[0] > G_MAX_NEIGHBORS:
                dist2 = np.einsum('ij,ij->i', offsets, offsets)
                keep = np.argpartition(dist2, G_MAX_NEIGHBORS - 1)[:G_MAX_NEIGHBORS]
                offsets = offsets[keep]
            local_neighbor_count = int(offsets.shape[0])
        else:
            offsets = np.empty((0, 3), dtype=np.float64)
            local_neighbor_count = 0


        directions = np.stack([
            dirx_block[local_idx, slots].astype(np.float64),
            diry_block[local_idx, slots].astype(np.float64),
            dirz_block[local_idx, slots].astype(np.float64),
        ], axis=1)

        n_slots = int(slots.size)

        if offsets.shape[0] < G_MIN_NEIGHBORS:


            is_clear_geom = np.ones(n_slots, dtype=bool)
            blocker_count_geom = np.zeros(n_slots, dtype=np.int32)
            front_count_geom = np.zeros(n_slots, dtype=np.int32)
            empty_cone_geom = np.full(n_slots, 180.0, dtype=np.float32)
        else:
            eval_batch = evaluate_candidate_directions_batch(
                neighbor_offsets=offsets,
                directions=directions,
                clearance_m=clearance_m,
                min_blocker_along_m=min_blocker_along_m,
            )
            is_clear_geom = eval_batch['is_clear'].astype(bool)
            blocker_count_geom = eval_batch['blocker_count'].astype(np.int32)
            front_count_geom = eval_batch['front_count'].astype(np.int32)
            empty_cone_geom = eval_batch['empty_cone_deg'].astype(np.float32)


        coarse = score_block[local_idx, slots].astype(np.float64)
        offaxis = offaxis_block[local_idx, slots].astype(np.float64)
        rng_m = range_block[local_idx, slots].astype(np.float64)
        border = border_block[local_idx, slots].astype(np.float64)
        view_zenith = view_zenith_block[local_idx, slots].astype(np.float64)
        view_azimuth = view_azimuth_block[local_idx, slots].astype(np.float64)
        local_view_cos_signed = local_view_cos_signed_block[local_idx, slots].astype(np.float64)
        local_view_angle = local_view_angle_block[local_idx, slots].astype(np.float64)
        solar_zenith = solar_zenith_block[local_idx, slots].astype(np.float64)
        solar_azimuth = solar_azimuth_block[local_idx, slots].astype(np.float64)
        relative_azimuth = relative_azimuth_block[local_idx, slots].astype(np.float64)
        sun_dirx = sun_dirx_block[local_idx, slots].astype(np.float64)
        sun_diry = sun_diry_block[local_idx, slots].astype(np.float64)
        sun_dirz = sun_dirz_block[local_idx, slots].astype(np.float64)
        local_solar_cos = local_solar_cos_block[local_idx, slots].astype(np.float64)
        local_solar_incidence = local_solar_incidence_block[local_idx, slots].astype(np.float64)
        is_backlit = is_backlit_block[local_idx, slots].astype(np.int8)
        surface_view_cos = surface_view_cos_block[local_idx, slots].astype(np.float64)
        surface_verticality = surface_verticality_block[local_idx, slots].astype(np.float64)
        normal_confidence = normal_confidence_block[local_idx, slots].astype(np.float64)
        scene_code_slots = row_scene_code[slots].astype(np.int32)
        u_vals = u_block[local_idx, slots].astype(np.float64)
        v_vals = v_block[local_idx, slots].astype(np.float64)
        img_zbuf = image_zbuffer_clear_block[local_idx, slots]


        reject_tube = blocker_count_geom > 0
        reject_narrow = (
            (min_cone > 0.0)
            & (front_count_geom > 0)
            & (empty_cone_geom < min_cone)
        )
        reject_img = ~img_zbuf

        is_clear_arr = is_clear_geom.copy()
        blocker_count_arr = blocker_count_geom.copy()
        front_count_arr = front_count_geom.copy()
        empty_cone_arr = empty_cone_geom.copy()


        is_clear_arr &= ~reject_narrow
        blocker_count_arr = np.where(reject_narrow, blocker_count_arr + 1, blocker_count_arr)


        is_clear_arr &= ~reject_img
        blocker_count_arr = np.where(reject_img, blocker_count_arr + 1, blocker_count_arr)
        front_count_arr = np.where(reject_img, front_count_arr + 1, front_count_arr)
        empty_cone_arr = np.where(reject_img, np.minimum(empty_cone_arr, 0.0), empty_cone_arr)


        reject_reason_slots = np.full(n_slots, 'none', dtype=object)
        for s_idx in range(n_slots):
            reasons = []
            if reject_tube[s_idx]:
                reasons.append('tube_blocker')
            if reject_narrow[s_idx]:
                reasons.append('narrow_empty_cone')
            if reject_img[s_idx]:
                reasons.append('image_zbuffer')
            if reasons:
                reject_reason_slots[s_idx] = '+'.join(dict.fromkeys(reasons))


        visibility_score_arr = build_selection_score_vec(
            coarse_score=coarse,
            empty_cone_deg=empty_cone_arr,
            blocker_count=blocker_count_arr,
            offaxis_deg=offaxis,
            range_m=rng_m,
            weights=G_SCORE_WEIGHTS,
            view_zenith_deg=view_zenith,
            surface_view_cos=surface_view_cos,
            surface_verticality=surface_verticality,
            normal_confidence=normal_confidence,
        )
        raw_weight_arr = build_multiview_weight_vec(
            coarse_score=coarse,
            empty_cone_deg=empty_cone_arr,
            blocker_count=blocker_count_arr,
            offaxis_deg=offaxis,
            range_m=rng_m,
            border_dist_px=border,
            is_clear=is_clear_arr,
            params=G_WEIGHT_PARAMS,
            view_zenith_deg=view_zenith,
            surface_view_cos=surface_view_cos,
            surface_verticality=surface_verticality,
            normal_confidence=normal_confidence,
        )

        clear_mask = is_clear_arr
        visible_count_arr[local_idx] = int(n_slots)
        clear_count_arr[local_idx] = int(np.count_nonzero(clear_mask))


        neg_score = -visibility_score_arr
        neg_raw = -raw_weight_arr
        neg_svc = -surface_view_cos
        neg_vz_vv = -(view_zenith * surface_verticality)
        neg_cone = -empty_cone_arr
        neg_minus_off = offaxis
        neg_border = -border
        neg_minus_rng = rng_m

        order = np.lexsort((
            neg_minus_rng, neg_border, neg_minus_off, neg_cone,
            neg_vz_vv, neg_svc, neg_raw, neg_score,
        ))

        kept_is_fallback = False
        if clear_mask.any():
            if G_KEEP_UNCLEAR_WITH_CLEAR:
                keep_order = order[:max_visible]
            else:
                clear_order = order[clear_mask[order]]
                keep_order = clear_order[:max_visible]
        elif G_KEEP_FALLBACK:
            keep_order = order[:max_fallback]
            kept_is_fallback = True
            used_fallback_arr[local_idx] = True
        else:
            keep_order = np.empty(0, dtype=np.int64)

        if keep_order.size == 0:
            continue

        k = int(keep_order.size)
        kept_count_arr[local_idx] = int(k)

        final_raw_weight = raw_weight_arr[keep_order]
        if kept_is_fallback:
            final_raw_weight = final_raw_weight * float(G_FALLBACK_WEIGHT_PENALTY)

        selection_method = []
        if kept_is_fallback:
            selection_method = ['fallback_keep'] * k
        else:
            kept_clear = clear_mask[keep_order]
            selection_method = [
                'clear_keep' if kept_clear[i] else 'unclear_keep' for i in range(k)
            ]

        out_point_ids.append(np.full(k, pid, dtype=np.int64))
        out_scene_code.append(scene_code_slots[keep_order])
        out_u.append(u_vals[keep_order])
        out_v.append(v_vals[keep_order])
        out_range.append(rng_m[keep_order])
        out_offaxis.append(offaxis[keep_order])
        out_border.append(border[keep_order])
        out_view_zenith.append(view_zenith[keep_order])
        out_view_azimuth.append(view_azimuth[keep_order])
        out_local_view_cos_signed.append(local_view_cos_signed[keep_order])
        out_local_view_angle.append(local_view_angle[keep_order])
        out_solar_zenith.append(solar_zenith[keep_order])
        out_solar_azimuth.append(solar_azimuth[keep_order])
        out_relative_azimuth.append(relative_azimuth[keep_order])
        out_sun_dirx.append(sun_dirx[keep_order])
        out_sun_diry.append(sun_diry[keep_order])
        out_sun_dirz.append(sun_dirz[keep_order])
        out_local_solar_cos.append(local_solar_cos[keep_order])
        out_local_solar_incidence.append(local_solar_incidence[keep_order])
        out_is_backlit.append(is_backlit[keep_order])
        out_surface_view_cos.append(surface_view_cos[keep_order])
        out_surface_verticality.append(surface_verticality[keep_order])
        out_normal_confidence.append(normal_confidence[keep_order])
        out_coarse.append(coarse[keep_order])
        out_visibility_score.append(visibility_score_arr[keep_order])
        out_view_weight_raw.append(final_raw_weight)
        out_dirx.append(directions[keep_order, 0])
        out_diry.append(directions[keep_order, 1])
        out_dirz.append(directions[keep_order, 2])
        out_empty_cone.append(empty_cone_arr[keep_order])
        out_blocker.append(blocker_count_arr[keep_order])
        out_front_count.append(front_count_arr[keep_order])
        out_local_neigh.append(np.full(k, local_neighbor_count, dtype=np.int32))
        out_image_zbuf.append(img_zbuf[keep_order])
        out_reject_reason.append([reject_reason_slots[i] for i in keep_order])
        out_is_clear.append(is_clear_arr[keep_order])
        out_selection_method.append(selection_method)
        out_fusion_rank.append(np.arange(1, k + 1, dtype=np.int32))

    emitted_rows = int(sum(a.size for a in out_point_ids))

    if emitted_rows == 0:
        return {
            'point_ids': point_ids,
            'visible_count_arr': visible_count_arr,
            'clear_count_arr': clear_count_arr,
            'kept_count_arr': kept_count_arr,
            'used_fallback_arr': used_fallback_arr,
            'emitted_rows': 0,
            'shard_path': shard_path_override or G_WORKER_SHARD_PATH or '',
            'table_df': None,
        }


    is_clear_output = np.concatenate(out_is_clear).astype(bool)
    selection_method_output = np.concatenate([np.asarray(x, dtype=object) for x in out_selection_method])


    fallback_output = np.char.find(selection_method_output.astype(str), 'fallback') >= 0
    keep_for_fusion_output = is_clear_output | fallback_output

    df = pd.DataFrame({
        'point_id': np.concatenate(out_point_ids),
        'scene_code': np.concatenate(out_scene_code),
        'u': np.concatenate(out_u),
        'v': np.concatenate(out_v),
        'range_m': np.concatenate(out_range),
        'offaxis_deg': np.concatenate(out_offaxis),
        'border_dist_px': np.concatenate(out_border),
        'view_zenith_deg': np.concatenate(out_view_zenith),
        'view_azimuth_deg': np.concatenate(out_view_azimuth),
        'local_view_cos_signed': np.concatenate(out_local_view_cos_signed),
        'local_view_angle_deg': np.concatenate(out_local_view_angle),
        'solar_zenith_deg': np.concatenate(out_solar_zenith),
        'solar_azimuth_deg': np.concatenate(out_solar_azimuth),
        'relative_azimuth_deg': np.concatenate(out_relative_azimuth),
        'sun_dir_x': np.concatenate(out_sun_dirx),
        'sun_dir_y': np.concatenate(out_sun_diry),
        'sun_dir_z': np.concatenate(out_sun_dirz),
        'local_solar_cos': np.concatenate(out_local_solar_cos),
        'local_solar_incidence_deg': np.concatenate(out_local_solar_incidence),
        'is_backlit': np.concatenate(out_is_backlit).astype(np.int8),
        'surface_view_cos': np.concatenate(out_surface_view_cos),
        'surface_verticality': np.concatenate(out_surface_verticality),
        'normal_confidence': np.concatenate(out_normal_confidence),
        'coarse_score': np.concatenate(out_coarse),
        'visibility_score': np.concatenate(out_visibility_score),
        'view_weight_raw': np.concatenate(out_view_weight_raw),
        'local_free_dir_x': np.concatenate(out_dirx),
        'local_free_dir_y': np.concatenate(out_diry),
        'local_free_dir_z': np.concatenate(out_dirz),
        'local_empty_cone_deg': np.concatenate(out_empty_cone),
        'blocker_count': np.concatenate(out_blocker),
        'front_neighbor_count': np.concatenate(out_front_count),
        'local_neighbor_count': np.concatenate(out_local_neigh),
        'image_zbuffer_clear': np.concatenate(out_image_zbuf),
        'clear_reject_reason': np.concatenate([np.asarray(x, dtype=object) for x in out_reject_reason]),
        'is_clear': is_clear_output,
        'selection_method': selection_method_output,
        'keep_for_fusion': keep_for_fusion_output,
        'fusion_rank_within_point': np.concatenate(out_fusion_rank),
    })


    target_shard_path = shard_path_override or G_WORKER_SHARD_PATH
    if target_shard_path:
        write_header = not os.path.exists(target_shard_path)
        df.to_csv(
            target_shard_path,
            mode='w' if write_header else 'a',
            header=write_header,
            index=False,
            encoding='utf-8-sig' if write_header else 'utf-8',
        )
        return {
            'point_ids': point_ids,
            'visible_count_arr': visible_count_arr,
            'clear_count_arr': clear_count_arr,
            'kept_count_arr': kept_count_arr,
            'used_fallback_arr': used_fallback_arr,
            'emitted_rows': emitted_rows,
            'shard_path': target_shard_path,
            'table_df': None,
        }

    return {
        'point_ids': point_ids,
        'visible_count_arr': visible_count_arr,
        'clear_count_arr': clear_count_arr,
        'kept_count_arr': kept_count_arr,
        'used_fallback_arr': used_fallback_arr,
        'emitted_rows': emitted_rows,
        'shard_path': '',
        'table_df': df,
    }


def _write_scene_summary(visible_csv_path: Path, mapping_path: Path, out_dir: Path, chunksize: int = 500000) -> None:

    mapping = read_table(mapping_path)
    if not visible_csv_path.exists():
        mapping.to_csv(out_dir / 'selection_scene_summary.csv', index=False, encoding='utf-8-sig')
        return

    available_cols = pd.read_csv(visible_csv_path, nrows=0, encoding='utf-8-sig').columns.tolist()
    mean_cols = [
        'visibility_score', 'view_weight_raw', 'range_m', 'offaxis_deg',
        'view_zenith_deg', 'view_azimuth_deg', 'local_view_angle_deg',
        'solar_zenith_deg', 'solar_azimuth_deg', 'relative_azimuth_deg',
        'local_solar_incidence_deg',
    ]
    usecols = [c for c in ['point_id', 'scene_code', 'scene_id', 'keep_for_fusion', 'is_clear'] + mean_cols if c in available_cols]
    dtype_map = {
        'point_id': 'int64',
        'scene_code': 'int16',
        'scene_id': str,
        'visibility_score': 'float32',
        'view_weight_raw': 'float32',
        'range_m': 'float32',
        'offaxis_deg': 'float32',
        'view_zenith_deg': 'float32',
        'view_azimuth_deg': 'float32',
        'local_view_angle_deg': 'float32',
        'solar_zenith_deg': 'float32',
        'solar_azimuth_deg': 'float32',
        'relative_azimuth_deg': 'float32',
        'local_solar_incidence_deg': 'float32',
    }
    dtype_map = {k: v for k, v in dtype_map.items() if k in usecols}

    stats: dict[tuple[int, str], dict] = {}
    carry = None

    def _as_bool(v: pd.Series, default: bool = False) -> pd.Series:
        if pd.api.types.is_bool_dtype(v):
            return v.fillna(default).astype(bool)
        return v.fillna(str(default)).astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y'])

    def _process(df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            return
        if 'keep_for_fusion' in df.columns:
            df = df[_as_bool(df['keep_for_fusion'], default=False)].copy()
        if len(df) == 0:
            return
        if 'is_clear' in df.columns:
            df['_is_clear_int'] = _as_bool(df['is_clear'], default=False).astype(np.int32)
        else:
            df['_is_clear_int'] = 0
        for (scene_code, scene_id), g in df.groupby(['scene_code', 'scene_id'], sort=False):
            key = (int(scene_code), str(scene_id))
            st = stats.setdefault(key, {
                'kept_observation_count': 0,
                'contributing_point_count': 0,
                'clear_selected_count': 0,
                **{f'{c}_sum': 0.0 for c in mean_cols},
                **{f'{c}_n': 0 for c in mean_cols},
            })
            st['kept_observation_count'] += int(len(g))
            st['contributing_point_count'] += int(g['point_id'].nunique())
            st['clear_selected_count'] += int(g['_is_clear_int'].sum())
            for c in mean_cols:
                if c in g.columns:
                    vals = pd.to_numeric(g[c], errors='coerce')
                    n = int(vals.notna().sum())
                    if n:
                        st[f'{c}_sum'] += float(vals.sum(skipna=True))
                        st[f'{c}_n'] += n

    reader = pd.read_csv(visible_csv_path, usecols=usecols, dtype=dtype_map, chunksize=chunksize, encoding='utf-8-sig')
    for chunk in reader:
        if carry is not None and len(carry) > 0:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        if len(chunk) == 0:
            carry = None
            continue
        last_pid = int(chunk['point_id'].iloc[-1])
        carry_mask = chunk['point_id'].values == last_pid
        _process(chunk.loc[~carry_mask].copy())
        carry = chunk.loc[carry_mask].copy()
    _process(carry)

    rows = []
    for (scene_code, scene_id), st in stats.items():
        row = {
            'scene_code': int(scene_code),
            'scene_id': str(scene_id),
            'kept_observation_count': int(st['kept_observation_count']),
            'contributing_point_count': int(st['contributing_point_count']),
            'clear_selected_count': int(st['clear_selected_count']),
        }
        for c in mean_cols:
            n = int(st.get(f'{c}_n', 0))
            row['mean_' + c.replace('view_weight_raw', 'raw_weight').replace('range_m', 'range_m')] = (float(st[f'{c}_sum']) / n) if n else np.nan
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    summary = mapping.merge(aggregated, on=['scene_code', 'scene_id'], how='left') if len(rows) else mapping.copy()
    summary.to_csv(out_dir / 'selection_scene_summary.csv', index=False, encoding='utf-8-sig')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    quiet = is_quiet(cfg)
    out_dir = ensure_dir(cfg['outputs']['selection_dir'])
    point_dir = Path(cfg['outputs']['pointcloud_dir'])
    cand_dir = Path(cfg['outputs']['candidate_dir'])

    proc = dict(cfg.get('processing', {}))
    radius = float(proc.get('local_visibility_radius_m', 0.8))
    clearance = float(proc.get('local_blocker_clearance_m', 0.08))
    min_blocker_along = float(proc.get('local_blocker_min_along_m', 0.0))
    min_empty_cone_clear = float(proc.get('local_empty_cone_clear_min_deg', 0.0))
    min_neighbors = int(proc.get('local_min_neighbor_count', 10))
    score_weights = dict(proc.get('selection_score_weights', {}))
    weight_params = dict(proc.get('fusion_weight_params', {}))
    kd_leafsize = int(proc.get('kd_leafsize', 32))
    kd_balanced_tree = bool(proc.get('selection_kdtree_balanced_tree', False))
    kd_compact_nodes = bool(proc.get('selection_kdtree_compact_nodes', False))

    selection_workers = int(proc.get('selection_workers', max(1, (os.cpu_count() or 1) // 2)))
    selection_backend = str(proc.get('selection_backend', 'threads')).strip().lower()
    if selection_backend in ('thread', 'threadpool'):
        selection_backend = 'threads'
    if selection_backend in ('process', 'processes', 'processpool'):
        selection_backend = 'processes'
    if selection_backend not in ('threads', 'processes'):
        print(f'Unknown selection_backend={selection_backend!r}; falling back to threads')
        selection_backend = 'threads'
    selection_chunk_size = int(proc.get('selection_chunk_size', 12000))
    selection_log_every_points = int(proc.get('selection_log_every_points', 200000))
    eval_scene_limit = int(proc.get('selection_scene_limit_per_point', proc.get('candidate_scene_limit_per_point', 8)))
    local_max_neighbors = int(proc.get('local_max_neighbors', 128))
    neighbor_query_mode = str(proc.get('selection_neighbor_query_mode', 'bounded_knn')).strip().lower()
    max_visible_views = int(proc.get('max_visible_views_per_point', 4))
    keep_fallback = bool(proc.get('keep_fallback_when_no_clear', True))
    fallback_weight_penalty = float(proc.get('fallback_weight_penalty', 0.15))
    keep_unclear_with_clear = bool(proc.get('keep_unclear_views_with_clear', True))
    max_fallback_views = int(proc.get('max_fallback_views_per_point', min(4, max_visible_views)))
    image_zbuffer_enabled = bool(proc.get('image_zbuffer_occlusion_enabled', False))
    image_zbuffer_radius_px = int(proc.get('image_zbuffer_radius_px', 1))
    image_zbuffer_depth_tolerance_m = float(proc.get('image_zbuffer_depth_tolerance_m', 0.2))
    image_zbuffer_rebuild = bool(proc.get('image_zbuffer_rebuild', False))

    point_xyz_path = point_dir / 'point_xyz.npy'
    cand_npz_path = cand_dir / 'point_topk_candidates.npz'
    mapping_path = cand_dir / 'scene_id_mapping.csv'

    t0 = time.perf_counter()
    _log_stage('Preparing candidate cache metadata ...')
    cache_rebuild = bool(proc.get('selection_rebuild_candidate_cache', False))
    direct_cache_dir = cand_dir / 'point_topk_candidates_memmap'
    direct_cache_meta = _load_existing_candidate_memmap_cache(direct_cache_dir)
    if direct_cache_meta is not None:
        cache_dir = direct_cache_dir
        cache_meta = direct_cache_meta
    else:
        cache_dir = out_dir / '_candidate_memmap_cache'
        cache_meta = _prepare_candidate_memmap_cache(cand_npz_path=cand_npz_path, cache_dir=cache_dir, rebuild=cache_rebuild)
    _log_done('Preparing candidate cache metadata', t0)
    total_points = int(cache_meta['point_count'])
    top_k = int(cache_meta['top_k'])
    eval_scene_limit = min(eval_scene_limit, top_k)

    t0 = time.perf_counter()
    _log_stage(f'Finding candidate points: eval_scene_limit={eval_scene_limit}, scan_chunk_rows={int(proc.get("candidate_scan_chunk_size", 200000))} ...')
    scene_code_topk = np.load(cache_dir / 'scene_code.npy', mmap_mode='r')
    candidate_scan_chunk_size = int(proc.get('candidate_scan_chunk_size', 200000))
    candidate_points = _find_candidate_points(scene_code_topk, eval_scene_limit, candidate_scan_chunk_size)
    _log_done('Finding candidate points', t0)

    image_zbuffer_clear_path = ''
    if image_zbuffer_enabled:
        t0 = time.perf_counter()
        _log_stage('Preparing image z-buffer clear cache ...')
        image_zbuffer_clear_path = str(_prepare_image_zbuffer_clear(
            cache_dir=cache_dir,
            total_points=total_points,
            top_k=top_k,
            image_width=int(cfg.get('camera_model', {}).get('image_width', 1886)),
            image_height=int(cfg.get('camera_model', {}).get('image_height', 1886)),
            radius_px=image_zbuffer_radius_px,
            depth_tolerance_m=image_zbuffer_depth_tolerance_m,
            rebuild=image_zbuffer_rebuild,
        ))
        _log_done('Preparing image z-buffer clear cache', t0)

    t0 = time.perf_counter()
    _log_stage('Reading scene mapping ...')
    mapping = pd.read_csv(mapping_path)
    scene_lookup = {int(r.scene_code): str(r.scene_id) for r in mapping.itertuples(index=False)}
    _log_done('Reading scene mapping', t0)

    print(f'Total points: {total_points}')
    print(f'Points with candidates: {len(candidate_points)}')
    if not quiet:
        print(f'Candidate scene limit used in step 06: {eval_scene_limit}')
        print(f'Max visible views kept per point: {max_visible_views}')
        print(f'selection_backend={selection_backend}, selection_workers={selection_workers}, selection_chunk_size={selection_chunk_size}, local_max_neighbors={local_max_neighbors}, neighbor_query_mode={neighbor_query_mode}')
        print(f'kdtree leafsize={kd_leafsize}, balanced_tree={kd_balanced_tree}, compact_nodes={kd_compact_nodes}')
        print(f'keep_unclear_views_with_clear={keep_unclear_with_clear}, max_fallback_views_per_point={max_fallback_views}')
        print(f'local_blocker_clearance_m={clearance}, local_blocker_min_along_m={min_blocker_along}, local_empty_cone_clear_min_deg={min_empty_cone_clear}')
        print(f'image_zbuffer_occlusion_enabled={image_zbuffer_enabled}, radius_px={image_zbuffer_radius_px}, depth_tolerance_m={image_zbuffer_depth_tolerance_m}')

    t0 = time.perf_counter()
    _log_stage('Allocating selection summary arrays ...')
    visible_count = np.zeros(total_points, dtype=np.int16)
    clear_count = np.zeros(total_points, dtype=np.int16)
    kept_count = np.zeros(total_points, dtype=np.int16)
    used_fallback = np.zeros(total_points, dtype=bool)
    _log_done('Allocating selection summary arrays', t0)

    csv_path = out_dir / 'visible_observations.csv'
    if csv_path.exists():
        csv_path.unlink()


    use_worker_shards = bool(proc.get('selection_use_worker_shards', True))
    shard_dir: Path | None = None
    if use_worker_shards:
        shard_dir = out_dir / '_visible_observations_shards'
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        ensure_dir(shard_dir)

    t0 = time.perf_counter()
    _log_stage('Building point chunk list ...')
    chunks = list(_chunk_array(candidate_points, selection_chunk_size))
    _log_done(f'Building point chunk list: chunks={len(chunks)}', t0)
    init_args = (
        str(point_xyz_path), str(cache_dir), image_zbuffer_clear_path, radius, clearance, min_blocker_along, min_empty_cone_clear, min_neighbors,
        score_weights, weight_params, eval_scene_limit, local_max_neighbors,
        neighbor_query_mode, kd_leafsize, max_visible_views, max_fallback_views, keep_fallback, fallback_weight_penalty, keep_unclear_with_clear,
        kd_balanced_tree, kd_compact_nodes,
        str(shard_dir) if shard_dir is not None else '',
    )

    processed = 0
    emitted_rows = 0
    shard_paths: set[str] = set()


    def append_df_to_main_csv(df: pd.DataFrame) -> int:
        df = df.copy()
        df['scene_id'] = df['scene_code'].map(scene_lookup)
        cols = (
            ['point_id', 'scene_code', 'scene_id']
            + [c for c in TABLE_COLUMNS if c not in ('point_id', 'scene_code')]
        )
        df = df[cols]
        append_table(df, csv_path, index=False)
        return int(len(df))

    def flush_result(result: dict) -> None:
        nonlocal emitted_rows
        pids = result['point_ids']
        visible_count[pids] = result['visible_count_arr']
        clear_count[pids] = result['clear_count_arr']
        kept_count[pids] = result['kept_count_arr']
        used_fallback[pids] = result['used_fallback_arr']

        if result.get('shard_path'):
            shard_paths.add(str(result['shard_path']))
            emitted_rows += int(result.get('emitted_rows', 0))
            return

        df = result.get('table_df')
        if df is None or len(df) == 0:
            return
        emitted_rows += append_df_to_main_csv(df)

    if selection_workers <= 1 or len(chunks) <= 1:
        _log_stage('Starting sequential visibility processing ...')
        _init_worker(*init_args)
        for ci, chunk in enumerate(chunks):
            shard_override = str(shard_dir / f'shard_seq_{ci:06d}.csv') if shard_dir is not None else ''
            result = _process_point_chunk(chunk, shard_override)
            flush_result(result)
            processed += len(result['point_ids'])
            if (processed // selection_log_every_points) != ((processed - len(result['point_ids'])) // selection_log_every_points):
                print(f'Visibility progress: {processed}/{len(candidate_points)} | emitted_rows={emitted_rows}')
    elif selection_backend == 'threads':


        thread_init_args = init_args[:-1] + ('',)
        _log_stage('Initializing shared thread backend ...')
        _init_worker(*thread_init_args)
        _log_stage(f'Submitting {len(chunks)} visibility chunk task(s) to {selection_workers} thread(s) ...')
        with ThreadPoolExecutor(max_workers=selection_workers) as executor:
            futures = []
            for ci, chunk in enumerate(chunks):
                shard_override = str(shard_dir / f'shard_thread_{ci:06d}.csv') if shard_dir is not None else ''
                futures.append(executor.submit(_process_point_chunk, chunk, shard_override))
            _log_stage('All visibility chunk tasks submitted; waiting for completed chunks ...')
            for future in as_completed(futures):
                result = future.result()
                flush_result(result)
                processed += len(result['point_ids'])
                if (processed // selection_log_every_points) != ((processed - len(result['point_ids'])) // selection_log_every_points):
                    print(f'Visibility progress: {processed}/{len(candidate_points)} | emitted_rows={emitted_rows}')
    else:


        _log_stage(f'Submitting {len(chunks)} visibility chunk task(s) to {selection_workers} process(es) ...')
        with ProcessPoolExecutor(max_workers=selection_workers, initializer=_init_worker, initargs=init_args) as executor:
            futures = [executor.submit(_process_point_chunk, chunk) for chunk in chunks]
            _log_stage('All visibility chunk tasks submitted; waiting for completed chunks ...')
            for future in as_completed(futures):
                result = future.result()
                flush_result(result)
                processed += len(result['point_ids'])
                if (processed // selection_log_every_points) != ((processed - len(result['point_ids'])) // selection_log_every_points):
                    print(f'Visibility progress: {processed}/{len(candidate_points)} | emitted_rows={emitted_rows}')

    print(f'Visibility progress: {processed}/{len(candidate_points)} | emitted_rows={emitted_rows}')


    if shard_paths:
        print(f'Merging {len(shard_paths)} worker shard CSV(s) into {csv_path} ...')
        ensure_dir(csv_path.parent)
        if csv_path.exists():
            csv_path.unlink()
        merge_chunk_rows = int(proc.get('selection_shard_merge_chunk_rows', 500000))
        for shard_path in sorted(shard_paths):
            for df_part in pd.read_csv(shard_path, chunksize=merge_chunk_rows):
                df_part['scene_id'] = df_part['scene_code'].map(scene_lookup)
                cols = (
                    ['point_id', 'scene_code', 'scene_id']
                    + [c for c in TABLE_COLUMNS if c not in ('point_id', 'scene_code')]
                )
                df_part = df_part[cols]
                append_table(df_part, csv_path, index=False)
        if shard_dir is not None and shard_dir.exists():
            shutil.rmtree(shard_dir, ignore_errors=True)

    _write_scene_summary(csv_path, mapping_path, out_dir, chunksize=int(proc.get('selection_shard_merge_chunk_rows', 500000)))

    save_json({
        'point_count': int(total_points),
        'candidate_point_count': int(len(candidate_points)),
        'points_with_kept_views': int(np.count_nonzero(kept_count > 0)),
        'points_using_fallback': int(np.count_nonzero(used_fallback)),
        'emitted_visible_rows': int(emitted_rows),
        'visibility_radius_m': radius,
        'blocker_clearance_m': clearance,
        'blocker_min_along_m': min_blocker_along,
        'empty_cone_clear_min_deg': min_empty_cone_clear,
        'selection_backend': str(selection_backend),
        'selection_workers': int(selection_workers),
        'selection_chunk_size': int(selection_chunk_size),
        'selection_scene_limit_per_point': int(eval_scene_limit),
        'max_visible_views_per_point': int(max_visible_views),
        'max_fallback_views_per_point': int(max_fallback_views),
        'keep_fallback_when_no_clear': bool(keep_fallback),
        'fallback_weight_penalty': float(fallback_weight_penalty),
        'keep_unclear_views_with_clear': bool(keep_unclear_with_clear),
        'local_max_neighbors': int(local_max_neighbors),
        'image_zbuffer_occlusion_enabled': bool(image_zbuffer_enabled),
        'image_zbuffer_radius_px': int(image_zbuffer_radius_px),
        'image_zbuffer_depth_tolerance_m': float(image_zbuffer_depth_tolerance_m),
        'candidate_cache_dir': str(cache_dir),
    }, out_dir / 'selection_summary.json')

    print(f'Kept visible observation rows: {emitted_rows}')
    print(f'Points with fusion views: {np.count_nonzero(kept_count > 0)}')


if __name__ == '__main__':
    main()
