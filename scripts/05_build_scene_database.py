
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import numpy as np
import pandas as pd
from PIL import Image

from hyperspectral_pointcloud_fusion.common import load_config, read_table, ensure_dir, save_json, percentile_stretch, get_scene_workers, is_quiet
from hyperspectral_pointcloud_fusion.envi import parse_envi_hdr, wavelength_list_from_hdr, read_envi_bsq_memmap
from hyperspectral_pointcloud_fusion.geometry import build_camera_frame, build_world_to_camera_matrix


RGB_BAND_ORDER = [73, 51, 31]


def truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


def normalize_vector(vec: np.ndarray) -> tuple[bool, np.ndarray]:
    vec = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n <= 1e-12:
        return False, np.full(3, np.nan, dtype=np.float64)
    return True, vec / n


def finite_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def build_scene_preview_rgb(cube_bsq: np.ndarray, rgb_band_numbers: list[int], scale_factor: float) -> tuple[np.ndarray, list[dict]]:
    bands = cube_bsq.shape[0]
    available = [b for b in rgb_band_numbers if 1 <= int(b) <= int(bands)]
    if len(available) != 3:
        raise ValueError(f'预览 RGB 波段不可用，要求={rgb_band_numbers}, 实际 bands={bands}')
    channels = []
    logs = []
    for band_no in available:
        band = np.asarray(cube_bsq[int(band_no) - 1], dtype=np.float64)
        if scale_factor not in (0, 1, 1.0):
            band = band / float(scale_factor)
        band = band * 10000.0
        valid = np.isfinite(band)
        ch, lo, hi = percentile_stretch(band, valid, q_low=0.0005, q_high=0.9995)
        channels.append(ch)
        logs.append({'band_number_one_based': int(band_no), 'stretch_min': lo, 'stretch_max': hi, 'input_transform': 'reflectance * 10000'})
    rgb = np.dstack(channels).astype(np.uint8)
    return rgb, logs


def process_one_scene(task: tuple[dict, dict, dict, list[int], str]):
    scene_row, pose_row, cam_cfg, rgb_band_numbers, out_root_str = task
    out_root = Path(out_root_str)
    scene_id = str(scene_row['scene_id'])
    scene_dir = ensure_dir(out_root / scene_id)
    hdr = parse_envi_hdr(scene_row['hdr_path'])
    wavelengths = wavelength_list_from_hdr(hdr)
    cube = read_envi_bsq_memmap(scene_row['data_path'], samples=int(scene_row['samples']), lines=int(scene_row['lines']), bands=int(scene_row['bands']), data_type=int(scene_row['data_type']), byte_order=int(scene_row['byte_order']), header_offset=int(scene_row['header_offset']))

    cam_xyz = np.array([pose_row['cam_x'], pose_row['cam_y'], pose_row['cam_z']], dtype=np.float64)
    anchor_xyz = np.array([
        pose_row.get('anchor_x', np.nan),
        pose_row.get('anchor_y', np.nan),
        pose_row.get('anchor_z', np.nan),
    ], dtype=np.float64)
    forward = np.array([pose_row['forward_x'], pose_row['forward_y'], pose_row['forward_z']], dtype=np.float64)
    camera_frame_source = 'pose_forward'
    if truthy(pose_row.get('anchor_updates_forward', False)) and np.all(np.isfinite(anchor_xyz)):
        ok, anchor_forward = normalize_vector(anchor_xyz - cam_xyz)
        if ok:
            forward = anchor_forward
            camera_frame_source = 'camera_to_anchor'
    right, down, forward = build_camera_frame(forward=forward, roll_deg=float(cam_cfg.get('roll_deg', 0.0)))
    R_wc = build_world_to_camera_matrix(right, down, forward)

    cam_local = np.array([pose_row['cam_local_x'], pose_row['cam_local_y'], pose_row['cam_local_z']], dtype=np.float64)
    anchor_local = np.array([pose_row['anchor_local_x'], pose_row['anchor_local_y'], pose_row['anchor_local_z']], dtype=np.float64)

    preview_rgb, preview_logs = build_scene_preview_rgb(cube, rgb_band_numbers=rgb_band_numbers, scale_factor=float(scene_row['reflectance_scale_factor']))
    Image.fromarray(preview_rgb).save(scene_dir / f'{scene_id}_rgb_73_51_31.png')

    vis_depth = float(np.linalg.norm(cam_local - anchor_local)) if np.all(np.isfinite(anchor_local)) else max(1.0, float(np.linalg.norm(cam_local)))
    fov_h = float(cam_cfg.get('fov_h_deg', 35.0))
    fov_v = float(cam_cfg.get('fov_v_deg', 35.0))
    half_w = np.tan(np.deg2rad(fov_h * 0.5)) * vis_depth
    half_h = np.tan(np.deg2rad(fov_v * 0.5)) * vis_depth
    plane_center_local = cam_local + forward * vis_depth
    plane_corners_local = {
        'ul': (plane_center_local - right * half_w - down * half_h).tolist(),
        'ur': (plane_center_local + right * half_w - down * half_h).tolist(),
        'lr': (plane_center_local + right * half_w + down * half_h).tolist(),
        'll': (plane_center_local - right * half_w + down * half_h).tolist(),
    }

    scene_meta = {
        'scene_id': scene_id,
        'hdr_path': str(scene_row['hdr_path']),
        'data_path': str(scene_row['data_path']),
        'samples': int(scene_row['samples']),
        'lines': int(scene_row['lines']),
        'bands': int(scene_row['bands']),
        'data_type': int(scene_row['data_type']),
        'byte_order': int(scene_row['byte_order']),
        'header_offset': int(scene_row['header_offset']),
        'reflectance_scale_factor': float(scene_row['reflectance_scale_factor']),
        'wavelength_count': int(len(wavelengths)),
        'wavelengths': wavelengths,
        'camera_xyz': cam_xyz.tolist(),
        'camera_local_xyz': cam_local.tolist(),
        'anchor_world_xyz': anchor_xyz.tolist(),
        'anchor_local_xyz': anchor_local.tolist(),
        'anchor_source': str(pose_row.get('anchor_source', 'unknown')),
        'anchor_hit_count': int(finite_float(pose_row.get('anchor_hit_count', 0), 0.0)),
        'anchor_range_m': finite_float(pose_row.get('anchor_range_m', np.nan), np.nan),
        'camera_frame_source': camera_frame_source,
        'forward_xyz': forward.tolist(),
        'right_xyz': right.tolist(),
        'down_xyz': down.tolist(),
        'world_to_camera_R': R_wc.tolist(),
        'camera_model': {
            'image_width': int(cam_cfg.get('image_width', scene_row['samples'])),
            'image_height': int(cam_cfg.get('image_height', scene_row['lines'])),
            'fov_h_deg': fov_h,
            'fov_v_deg': fov_v,
            'flip_u': bool(cam_cfg.get('flip_u', False)),
            'flip_v': bool(cam_cfg.get('flip_v', False)),
            'camera_forward_axis': str(cam_cfg.get('camera_forward_axis', 'z')),
            'image_v_axis': str(cam_cfg.get('image_v_axis', 'down')),
            'roll_deg': float(cam_cfg.get('roll_deg', 0.0)),
        },
        'preview_rgb_path': str(scene_dir / f'{scene_id}_rgb_73_51_31.png'),
        'preview_logs': preview_logs,
        'plane_center_local_xyz': plane_center_local.tolist(),
        'plane_corners_local_xyz': plane_corners_local,
        'visualization_depth_m': vis_depth,
        'datetime_local_iso': str(pose_row.get('datetime_local_iso', '')),
        'datetime_utc_iso': str(pose_row.get('datetime_utc_iso', '')),
        'solar_zenith_deg': finite_float(pose_row.get('solar_zenith_deg', np.nan), np.nan),
        'solar_elevation_deg': finite_float(pose_row.get('solar_elevation_deg', np.nan), np.nan),
        'solar_azimuth_deg': finite_float(pose_row.get('solar_azimuth_deg', np.nan), np.nan),
        'sun_dir_xyz': [
            finite_float(pose_row.get('sun_dir_x', np.nan), np.nan),
            finite_float(pose_row.get('sun_dir_y', np.nan), np.nan),
            finite_float(pose_row.get('sun_dir_z', np.nan), np.nan),
        ],
        'solar_geometry_valid': bool(truthy(pose_row.get('solar_geometry_valid', False))),
    }
    save_json(scene_meta, scene_dir / 'scene_meta.json')
    return {
        'scene_id': scene_id,
        'preview_rgb_path': scene_meta['preview_rgb_path'],
        'plane_center_local_x': float(plane_center_local[0]),
        'plane_center_local_y': float(plane_center_local[1]),
        'plane_center_local_z': float(plane_center_local[2]),
        'visualization_depth_m': float(vis_depth),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['outputs']['scene_db_dir'])
    scene_root = ensure_dir(out_dir / 'scenes')

    manifest_path = Path(cfg['outputs'].get('scene_quality_dir', cfg['outputs']['inventory_dir'])) / 'scene_manifest_filtered.csv'
    if not manifest_path.exists():
        manifest_path = Path(cfg['outputs']['inventory_dir']) / 'scene_manifest_filtered.csv'
    if not manifest_path.exists():
        manifest_path = Path(cfg['outputs']['inventory_dir']) / 'scene_manifest.csv'
    manifest = read_table(manifest_path)
    pose = read_table(Path(cfg['outputs']['crs_dir']) / 'scene_pose_aligned.csv')
    pose_map = pose.set_index('scene_id')
    cam_cfg = dict(cfg.get('camera_model', {}))
    rgb_band_numbers = list(cam_cfg.get('preview_rgb_band_numbers', RGB_BAND_ORDER))

    tasks = []
    for _, row in manifest.iterrows():
        sid = str(row['scene_id'])
        if sid not in pose_map.index:
            continue
        tasks.append((row.to_dict(), pose_map.loc[sid].to_dict(), cam_cfg, rgb_band_numbers, str(scene_root)))

    workers = max(1, min(get_scene_workers(cfg), len(tasks) if tasks else 1))
    quiet = is_quiet(cfg)
    if not quiet:
        print(f'scene_workers={workers}, tasks={len(tasks)}')
    rows = []
    if workers <= 1:
        for task in tasks:
            rows.append(process_one_scene(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_one_scene, task) for task in tasks]
            for fu in as_completed(futures):
                rows.append(fu.result())

    scene_info = manifest.copy()
    if rows:
        derived_df = pd.DataFrame(rows)
        db = scene_info.merge(pose, on='scene_id', how='left').merge(derived_df, on='scene_id', how='left')
    else:
        db = scene_info.merge(pose, on='scene_id', how='left')
    db.to_csv(out_dir / 'scene_database.csv', index=False, encoding='utf-8-sig')
    save_json({'scene_count': int(len(db)), 'scene_root': str(scene_root)}, out_dir / 'scene_database_summary.json')
    print(f'Output: {out_dir / "scene_database.csv"}')


if __name__ == '__main__':
    main()
