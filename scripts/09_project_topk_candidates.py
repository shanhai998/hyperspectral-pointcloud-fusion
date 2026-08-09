
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import load_config, read_table, ensure_dir, save_json, chunk_ranges, resolve_center_window, is_quiet
from hyperspectral_pointcloud_fusion.geometry import project_world_points_pinhole


CANDIDATE_ARRAY_SPECS = [
    ('score', np.float32, -np.inf),
    ('scene_code', np.int16, -1),
    ('u', np.float32, np.nan),
    ('v', np.float32, np.nan),
    ('range_m', np.float32, np.nan),
    ('offaxis_deg', np.float32, np.nan),
    ('border_dist_px', np.float32, np.nan),
    ('dir_x', np.float32, np.nan),
    ('dir_y', np.float32, np.nan),
    ('dir_z', np.float32, np.nan),
    ('view_zenith_deg', np.float32, np.nan),
    ('view_azimuth_deg', np.float32, np.nan),
    ('local_view_cos_signed', np.float32, np.nan),
    ('local_view_angle_deg', np.float32, np.nan),
    ('solar_zenith_deg', np.float32, np.nan),
    ('solar_azimuth_deg', np.float32, np.nan),
    ('relative_azimuth_deg', np.float32, np.nan),
    ('sun_dir_x', np.float32, np.nan),
    ('sun_dir_y', np.float32, np.nan),
    ('sun_dir_z', np.float32, np.nan),
    ('local_solar_cos', np.float32, np.nan),
    ('local_solar_incidence_deg', np.float32, np.nan),
    ('is_backlit', np.int8, -1),
    ('surface_view_cos', np.float32, np.nan),
    ('surface_verticality', np.float32, np.nan),
    ('normal_confidence', np.float32, np.nan),
]


def mask_points_in_center_window(u: np.ndarray, v: np.ndarray, window: dict) -> np.ndarray:
    return (
        (u >= float(window['x_min'])) &
        (u <= float(window['x_max'])) &
        (v >= float(window['y_min'])) &
        (v <= float(window['y_max']))
    )


def center_window_border_distance(u: np.ndarray, v: np.ndarray, window: dict) -> np.ndarray:
    return np.minimum.reduce([
        np.asarray(u, dtype=np.float64) - float(window['x_min']),
        np.asarray(v, dtype=np.float64) - float(window['y_min']),
        float(window['x_max']) - np.asarray(u, dtype=np.float64),
        float(window['y_max']) - np.asarray(v, dtype=np.float64),
    ])


def signed_distance_to_center_window(u: np.ndarray, v: np.ndarray, window: dict) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    inside = mask_points_in_center_window(u, v, window)
    signed = np.empty(u.shape, dtype=np.float64)
    if np.any(inside):
        signed[inside] = center_window_border_distance(u[inside], v[inside], window)
    if np.any(~inside):
        dx = np.maximum.reduce([
            np.full(np.count_nonzero(~inside), float(window['x_min'])) - u[~inside],
            u[~inside] - np.full(np.count_nonzero(~inside), float(window['x_max'])),
            np.zeros(np.count_nonzero(~inside), dtype=np.float64),
        ])
        dy = np.maximum.reduce([
            np.full(np.count_nonzero(~inside), float(window['y_min'])) - v[~inside],
            v[~inside] - np.full(np.count_nonzero(~inside), float(window['y_max'])),
            np.zeros(np.count_nonzero(~inside), dtype=np.float64),
        ])
        signed[~inside] = -np.sqrt(dx * dx + dy * dy)
    return signed


def fill_rows(arr: np.ndarray, value: float, chunk_rows: int) -> None:
    n = int(arr.shape[0])
    for i0, i1 in chunk_ranges(n, chunk_rows):
        arr[i0:i1] = value
    if hasattr(arr, 'flush'):
        arr.flush()


def initialize_candidate_arrays(out_dir: Path, n_points: int, top_k: int, use_memmap: bool, init_chunk_rows: int) -> tuple[dict[str, np.ndarray], Path | None]:
    shape = (int(n_points), int(top_k))
    arrays: dict[str, np.ndarray] = {}
    memmap_dir = None
    if use_memmap:
        memmap_dir = ensure_dir(out_dir / 'point_topk_candidates_memmap')
        manifest_path = memmap_dir / 'cache_manifest.json'
        if manifest_path.exists():
            manifest_path.unlink()
    for name, dtype, fill_value in CANDIDATE_ARRAY_SPECS:
        if use_memmap:
            arr = np.lib.format.open_memmap(memmap_dir / f'{name}.npy', mode='w+', dtype=np.dtype(dtype), shape=shape)
            fill_rows(arr, fill_value, init_chunk_rows)
        else:
            arr = np.full(shape, fill_value, dtype=np.dtype(dtype))
        arrays[name] = arr
    return arrays, memmap_dir


def sort_candidate_arrays(arrays: dict[str, np.ndarray], n_points: int, chunk_rows: int) -> None:
    for i0, i1 in chunk_ranges(n_points, chunk_rows):
        score_block = np.asarray(arrays['score'][i0:i1]).copy()
        order = np.argsort(-score_block, axis=1)
        rows = np.arange(i1 - i0)[:, None]
        for name, _dtype, _fill in CANDIDATE_ARRAY_SPECS:
            block = np.asarray(arrays[name][i0:i1]).copy()
            arrays[name][i0:i1] = block[rows, order]
    for arr in arrays.values():
        if hasattr(arr, 'flush'):
            arr.flush()


def write_memmap_manifest(memmap_dir: Path, arrays: dict[str, np.ndarray], n_points: int, top_k: int) -> None:
    meta = {}
    for name, _dtype, _fill in CANDIDATE_ARRAY_SPECS:
        arr = arrays[name]
        meta[name] = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}
    meta['point_count'] = int(n_points)
    meta['top_k'] = int(top_k)
    meta['source'] = 'step05_direct_memmap'
    with open(memmap_dir / 'cache_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    quiet = is_quiet(cfg)
    out_dir = ensure_dir(cfg['outputs']['candidate_dir'])
    scene_root = Path(cfg['outputs']['scene_db_dir']) / 'scenes'
    point_dir = Path(cfg['outputs']['pointcloud_dir'])

    scene_db = read_table(Path(cfg['outputs']['scene_db_dir']) / 'scene_database.csv')
    xyz = np.load(point_dir / 'point_xyz.npy', mmap_mode='r')
    point_normals = np.load(point_dir / 'point_normals.npy', mmap_mode='r')
    point_verticality = np.load(point_dir / 'point_surface_verticality.npy', mmap_mode='r')
    point_normal_confidence = np.load(point_dir / 'point_normal_confidence.npy', mmap_mode='r')
    n_points = int(xyz.shape[0])
    proc = dict(cfg.get('processing', {}))
    top_k = int(proc.get('candidate_scene_limit_per_point', 8))
    chunk_size = int(proc.get('projection_chunk_size', 200000))
    sort_chunk_size = int(proc.get('candidate_sort_chunk_size', max(20000, min(chunk_size, 100000))))
    init_chunk_size = int(proc.get('candidate_init_chunk_size', max(20000, min(chunk_size, 200000))))
    storage_mode = str(proc.get('candidate_storage_mode', 'memmap')).strip().lower()
    use_memmap = storage_mode != 'ram'
    write_candidate_npz = bool(proc.get('write_candidate_npz', not use_memmap))
    max_range = float(proc.get('max_range_m', 300.0))
    center_window_width = proc.get('center_window_width_px', 512)
    center_window_height = proc.get('center_window_height_px', center_window_width)
    center_window_policy = str(proc.get('center_window_policy', 'prefer')).strip().lower()
    coarse_border_weight = float(proc.get('coarse_border_weight', 0.02))
    coarse_offaxis_weight = float(proc.get('coarse_offaxis_weight', 1.0))
    coarse_range_weight = float(proc.get('coarse_range_weight', 0.01))
    coarse_surface_alignment_weight = float(proc.get('coarse_surface_alignment_weight', 0.0))
    coarse_vertical_view_zenith_weight = float(proc.get('coarse_vertical_view_zenith_weight', 0.0))

    arrays, memmap_dir = initialize_candidate_arrays(out_dir, n_points, top_k, use_memmap, init_chunk_size)
    score_topk = arrays['score']
    scene_code_topk = arrays['scene_code']
    u_topk = arrays['u']
    v_topk = arrays['v']
    range_topk = arrays['range_m']
    offaxis_topk = arrays['offaxis_deg']
    border_topk = arrays['border_dist_px']
    dirx_topk = arrays['dir_x']
    diry_topk = arrays['dir_y']
    dirz_topk = arrays['dir_z']
    view_zenith_topk = arrays['view_zenith_deg']
    view_azimuth_topk = arrays['view_azimuth_deg']
    local_view_cos_signed_topk = arrays['local_view_cos_signed']
    local_view_angle_topk = arrays['local_view_angle_deg']
    solar_zenith_topk = arrays['solar_zenith_deg']
    solar_azimuth_topk = arrays['solar_azimuth_deg']
    relative_azimuth_topk = arrays['relative_azimuth_deg']
    sun_dirx_topk = arrays['sun_dir_x']
    sun_diry_topk = arrays['sun_dir_y']
    sun_dirz_topk = arrays['sun_dir_z']
    local_solar_cos_topk = arrays['local_solar_cos']
    local_solar_incidence_topk = arrays['local_solar_incidence_deg']
    is_backlit_topk = arrays['is_backlit']
    surface_view_cos_topk = arrays['surface_view_cos']
    surface_verticality_topk = arrays['surface_verticality']
    normal_confidence_topk = arrays['normal_confidence']

    scene_id_mapping = []
    scene_rows = []
    for scene_code, row in enumerate(scene_db.itertuples(index=False)):
        meta_path = scene_root / str(row.scene_id) / 'scene_meta.json'
        if not meta_path.exists():
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        cam_xyz = np.asarray(meta['camera_xyz'], dtype=np.float64)
        R_wc = np.asarray(meta['world_to_camera_R'], dtype=np.float64)
        cam_model = dict(meta['camera_model'])
        width = int(cam_model['image_width'])
        height = int(cam_model['image_height'])
        fov_h = float(cam_model['fov_h_deg'])
        fov_v = float(cam_model['fov_v_deg'])
        flip_u = bool(cam_model.get('flip_u', False))
        flip_v = bool(cam_model.get('flip_v', False))
        center_window = resolve_center_window(width, height, center_window_width, center_window_height)
        solar_zenith_scene = float(getattr(row, 'solar_zenith_deg', np.nan)) if hasattr(row, 'solar_zenith_deg') else float(meta.get('solar_zenith_deg', np.nan))
        solar_azimuth_scene = float(getattr(row, 'solar_azimuth_deg', np.nan)) if hasattr(row, 'solar_azimuth_deg') else float(meta.get('solar_azimuth_deg', np.nan))
        if hasattr(row, 'sun_dir_x') and hasattr(row, 'sun_dir_y') and hasattr(row, 'sun_dir_z'):
            sun_dir_scene = np.array([float(row.sun_dir_x), float(row.sun_dir_y), float(row.sun_dir_z)], dtype=np.float64)
        else:
            sun_dir_scene = np.asarray(meta.get('sun_dir_xyz', [np.nan, np.nan, np.nan]), dtype=np.float64)
        if not np.all(np.isfinite(sun_dir_scene)):
            sun_dir_scene = np.full(3, np.nan, dtype=np.float64)
        scene_projected_inside_count = 0
        scene_center_window_count = 0
        scene_eligible_count = 0
        scene_candidate_count = 0
        for i0, i1 in chunk_ranges(n_points, chunk_size):
            pts = np.asarray(xyz[i0:i1], dtype=np.float64)
            u, v, rng, offaxis, _, inside = project_world_points_pinhole(
                pts,
                cam_xyz,
                R_wc,
                width,
                height,
                fov_h,
                fov_v,
                flip_u=flip_u,
                flip_v=flip_v,
            )
            inside &= np.isfinite(rng) & (rng <= max_range)
            scene_projected_inside_count += int(np.count_nonzero(inside))
            if not np.any(inside):
                continue

            in_center = mask_points_in_center_window(u, v, center_window)
            scene_center_window_count += int(np.count_nonzero(inside & in_center))
            if center_window_policy == 'strict':
                eligible = inside & in_center
            else:
                eligible = inside
            scene_eligible_count += int(np.count_nonzero(eligible))
            if not np.any(eligible):
                continue

            signed_center_dist = signed_distance_to_center_window(u, v, center_window)
            center_border = np.maximum(signed_center_dist, 0.0)
            pid = np.arange(i0, i1, dtype=np.int64)[eligible]
            rel = cam_xyz[None, :] - pts[eligible]
            rel_norm = np.linalg.norm(rel, axis=1, keepdims=True)
            rel_norm = np.maximum(rel_norm, 1e-12)
            d = rel / rel_norm
            normals = np.asarray(point_normals[pid], dtype=np.float64)
            local_view_cos_signed = np.einsum('ij,ij->i', normals, d)
            surface_view_cos = np.abs(local_view_cos_signed).clip(0.0, 1.0)
            surface_verticality = np.asarray(point_verticality[pid], dtype=np.float64).clip(0.0, 1.0)
            normal_confidence = np.asarray(point_normal_confidence[pid], dtype=np.float64).clip(0.0, 1.0)
            view_zenith_deg = np.degrees(np.arccos(np.clip(d[:, 2], -1.0, 1.0)))
            view_azimuth_deg = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360.0) % 360.0
            local_view_angle_deg = np.degrees(np.arccos(np.clip(local_view_cos_signed, -1.0, 1.0)))
            if np.all(np.isfinite(sun_dir_scene)):
                local_solar_cos = np.einsum('ij,j->i', normals, sun_dir_scene)
                local_solar_incidence_deg = np.degrees(np.arccos(np.clip(local_solar_cos, -1.0, 1.0)))
                is_backlit = local_solar_cos <= 0.0
            else:
                local_solar_cos = np.full(pid.size, np.nan, dtype=np.float64)
                local_solar_incidence_deg = np.full(pid.size, np.nan, dtype=np.float64)
                is_backlit = np.full(pid.size, False, dtype=bool)
            if np.isfinite(solar_azimuth_scene):
                relative_azimuth_deg = ((solar_azimuth_scene - view_azimuth_deg + 180.0) % 360.0) - 180.0
            else:
                relative_azimuth_deg = np.full(pid.size, np.nan, dtype=np.float64)
            geometry_bonus = (
                coarse_surface_alignment_weight * surface_view_cos * (0.25 + 0.75 * normal_confidence)
                + coarse_vertical_view_zenith_weight * view_zenith_deg * surface_verticality * (0.25 + 0.75 * normal_confidence)
            )
            new_score = (
                coarse_border_weight * signed_center_dist[eligible] -
                coarse_offaxis_weight * offaxis[eligible] -
                coarse_range_weight * rng[eligible] +
                geometry_bonus
            ).astype(np.float32)
            cur = score_topk[pid]
            min_slot = np.argmin(cur, axis=1)
            cur_min = cur[np.arange(pid.size), min_slot]
            take = new_score > cur_min
            if not np.any(take):
                continue
            rows = pid[take]
            slots = min_slot[take]
            score_topk[rows, slots] = new_score[take]
            scene_code_topk[rows, slots] = int(scene_code)
            u_topk[rows, slots] = u[eligible][take].astype(np.float32)
            v_topk[rows, slots] = v[eligible][take].astype(np.float32)
            range_topk[rows, slots] = rng[eligible][take].astype(np.float32)
            offaxis_topk[rows, slots] = offaxis[eligible][take].astype(np.float32)
            border_topk[rows, slots] = center_border[eligible][take].astype(np.float32)
            dirx_topk[rows, slots] = d[take, 0].astype(np.float32)
            diry_topk[rows, slots] = d[take, 1].astype(np.float32)
            dirz_topk[rows, slots] = d[take, 2].astype(np.float32)
            view_zenith_topk[rows, slots] = view_zenith_deg[take].astype(np.float32)
            view_azimuth_topk[rows, slots] = view_azimuth_deg[take].astype(np.float32)
            local_view_cos_signed_topk[rows, slots] = local_view_cos_signed[take].astype(np.float32)
            local_view_angle_topk[rows, slots] = local_view_angle_deg[take].astype(np.float32)
            solar_zenith_topk[rows, slots] = np.float32(solar_zenith_scene)
            solar_azimuth_topk[rows, slots] = np.float32(solar_azimuth_scene)
            relative_azimuth_topk[rows, slots] = relative_azimuth_deg[take].astype(np.float32)
            sun_dirx_topk[rows, slots] = np.float32(sun_dir_scene[0])
            sun_diry_topk[rows, slots] = np.float32(sun_dir_scene[1])
            sun_dirz_topk[rows, slots] = np.float32(sun_dir_scene[2])
            local_solar_cos_topk[rows, slots] = local_solar_cos[take].astype(np.float32)
            local_solar_incidence_topk[rows, slots] = local_solar_incidence_deg[take].astype(np.float32)
            is_backlit_topk[rows, slots] = is_backlit[take].astype(np.int8)
            surface_view_cos_topk[rows, slots] = surface_view_cos[take].astype(np.float32)
            surface_verticality_topk[rows, slots] = surface_verticality[take].astype(np.float32)
            normal_confidence_topk[rows, slots] = normal_confidence[take].astype(np.float32)
            scene_candidate_count += int(np.count_nonzero(take))
        scene_rows.append({
            'scene_code': int(scene_code),
            'scene_id': str(row.scene_id),
            'projected_inside_count': int(scene_projected_inside_count),
            'center_window_count': int(scene_center_window_count),
            'eligible_count': int(scene_eligible_count),
            'candidate_kept_count': int(scene_candidate_count),
            'center_window_policy': center_window_policy,
            'center_window_x_min': int(center_window['x_min']),
            'center_window_x_max': int(center_window['x_max']),
            'center_window_y_min': int(center_window['y_min']),
            'center_window_y_max': int(center_window['y_max']),
            'center_window_width_px': int(center_window['window_width']),
            'center_window_height_px': int(center_window['window_height']),
        })
        scene_id_mapping.append({'scene_code': int(scene_code), 'scene_id': str(row.scene_id)})
        if not quiet:
            print(
                f'[{scene_code + 1}/{len(scene_db)}] {row.scene_id}: '
                f'inside={scene_projected_inside_count}, '
                f'center={scene_center_window_count}, '
                f'eligible={scene_eligible_count}, '
                f'kept={scene_candidate_count}'
            )

    sort_candidate_arrays(arrays, n_points, sort_chunk_size)
    if memmap_dir is not None:
        write_memmap_manifest(memmap_dir, arrays, n_points, top_k)

    if write_candidate_npz:
        np.savez_compressed(
            out_dir / 'point_topk_candidates.npz',
            score=np.asarray(score_topk),
            scene_code=np.asarray(scene_code_topk),
            u=np.asarray(u_topk),
            v=np.asarray(v_topk),
            range_m=np.asarray(range_topk),
            offaxis_deg=np.asarray(offaxis_topk),
            border_dist_px=np.asarray(border_topk),
            dir_x=np.asarray(dirx_topk),
            dir_y=np.asarray(diry_topk),
            dir_z=np.asarray(dirz_topk),
            view_zenith_deg=np.asarray(view_zenith_topk),
            view_azimuth_deg=np.asarray(view_azimuth_topk),
            local_view_cos_signed=np.asarray(local_view_cos_signed_topk),
            local_view_angle_deg=np.asarray(local_view_angle_topk),
            solar_zenith_deg=np.asarray(solar_zenith_topk),
            solar_azimuth_deg=np.asarray(solar_azimuth_topk),
            relative_azimuth_deg=np.asarray(relative_azimuth_topk),
            sun_dir_x=np.asarray(sun_dirx_topk),
            sun_dir_y=np.asarray(sun_diry_topk),
            sun_dir_z=np.asarray(sun_dirz_topk),
            local_solar_cos=np.asarray(local_solar_cos_topk),
            local_solar_incidence_deg=np.asarray(local_solar_incidence_topk),
            is_backlit=np.asarray(is_backlit_topk),
            surface_view_cos=np.asarray(surface_view_cos_topk),
            surface_verticality=np.asarray(surface_verticality_topk),
            normal_confidence=np.asarray(normal_confidence_topk),
        )
    pd.DataFrame(scene_rows).to_csv(out_dir / 'candidate_scene_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(scene_id_mapping).to_csv(out_dir / 'scene_id_mapping.csv', index=False, encoding='utf-8-sig')
    save_json({
        'point_count': n_points,
        'top_k': top_k,
        'scene_count': int(len(scene_rows)),
        'max_range_m': max_range,
        'center_window_policy': center_window_policy,
        'center_window_width_px': None if center_window_width is None else int(center_window_width),
        'center_window_height_px': None if center_window_height is None else int(center_window_height),
        'coarse_surface_alignment_weight': float(coarse_surface_alignment_weight),
        'coarse_vertical_view_zenith_weight': float(coarse_vertical_view_zenith_weight),
        'candidate_storage_mode': storage_mode,
        'candidate_memmap_dir': None if memmap_dir is None else str(memmap_dir),
        'write_candidate_npz': bool(write_candidate_npz),
    }, out_dir / 'candidate_summary.json')
    if memmap_dir is not None:
        print(f'Output: {memmap_dir}')
    if write_candidate_npz:
        print(f'Output: {out_dir / "point_topk_candidates.npz"}')


if __name__ == '__main__':
    main()
