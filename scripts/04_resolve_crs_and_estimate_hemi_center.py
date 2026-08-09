

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import load_config, read_table, write_table, load_json, save_json, ensure_dir
from hyperspectral_pointcloud_fusion.geometry import (
    transform_lonlat_to_projected,
    compute_view_residual_deg,
    build_camera_forward_vector,
    intersect_ray_with_plane_z,
    compute_spherical_from_center,
)
from hyperspectral_pointcloud_fusion.plyio import read_ply_xyz


def score_candidate(df: pd.DataFrame, candidate: dict, point_bbox: dict, angle_cfg: dict, vertical_offset_m: float):
    ok = df['has_brdf'].fillna(False).astype(bool).values
    if not np.any(ok):
        raise ValueError('没有可用于 CRS 计算的 BRDF 记录')

    lon = df.loc[ok, 'cam_lon_wgs84'].astype(float).values
    lat = df.loc[ok, 'cam_lat_wgs84'].astype(float).values
    cam_x, cam_y = transform_lonlat_to_projected(lon, lat, candidate['src_crs'], candidate['dst_crs'], candidate.get('post_ops', {}))
    cam_z = df.loc[ok, 'cam_alt_raw'].astype(float).values.copy() + float(vertical_offset_m)

    has_target = ok & df['has_bref'].fillna(False).astype(bool).values
    az_res = np.full(np.count_nonzero(ok), np.nan, dtype=np.float64)
    ze_res = np.full(np.count_nonzero(ok), np.nan, dtype=np.float64)

    if np.any(has_target):
        lon_t = df.loc[has_target, 'target_lon_wgs84'].astype(float).values
        lat_t = df.loc[has_target, 'target_lat_wgs84'].astype(float).values
        tx, ty = transform_lonlat_to_projected(lon_t, lat_t, candidate['src_crs'], candidate['dst_crs'], candidate.get('post_ops', {}))


        placeholder_target_z = cam_z[df.loc[ok].index.get_indexer(df.loc[has_target].index)] - 1.0
        idx_map = df.loc[ok].index.get_indexer(df.loc[has_target].index)
        cam_xyz = np.column_stack([cam_x[idx_map], cam_y[idx_map], cam_z[idx_map]])
        target_xyz = np.column_stack([tx, ty, placeholder_target_z])
        az_tmp, ze_tmp = compute_view_residual_deg(
            cam_xyz,
            target_xyz,
            df.loc[has_target, 'view_azimuth_deg'].astype(float).values,
            df.loc[has_target, 'view_zenith_deg'].astype(float).values,
            angle_cfg,
        )
        az_res[idx_map] = az_tmp
        ze_res[idx_map] = ze_tmp

    xmin, xmax = point_bbox['xmin'], point_bbox['xmax']
    ymin, ymax = point_bbox['ymin'], point_bbox['ymax']
    margin_x = max(30.0, (xmax - xmin) * 0.2)
    margin_y = max(30.0, (ymax - ymin) * 0.2)
    in_x = (cam_x >= xmin - margin_x) & (cam_x <= xmax + margin_x)
    in_y = (cam_y >= ymin - margin_y) & (cam_y <= ymax + margin_y)
    bbox_hits = np.count_nonzero(in_x & in_y)

    x_center = 0.5 * (xmin + xmax)
    y_center = 0.5 * (ymin + ymax)
    center_dist = float(np.nanmedian(np.sqrt((cam_x - x_center) ** 2 + (cam_y - y_center) ** 2)))
    med_az = float(np.nanmedian(az_res)) if np.any(np.isfinite(az_res)) else 999.0
    med_ze = float(np.nanmedian(ze_res)) if np.any(np.isfinite(ze_res)) else 999.0

    score = bbox_hits * 1000.0 - center_dist - 2.0 * med_az - 2.0 * med_ze
    return {
        'name': candidate['name'],
        'score': float(score),
        'bbox_hits': int(bbox_hits),
        'median_center_distance_m': center_dist,
        'median_az_residual_deg': med_az,
        'median_zenith_residual_deg': med_ze,
    }


def robust_center_xy(anchors_xy: np.ndarray) -> tuple[float, float]:
    return float(np.median(anchors_xy[:, 0])), float(np.median(anchors_xy[:, 1]))


def normalize_vector(vec: np.ndarray) -> tuple[bool, np.ndarray]:
    vec = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n <= 1e-12:
        return False, np.full(3, np.nan, dtype=np.float64)
    return True, vec / n


def ray_ground_anchor(cam_xyz: np.ndarray, forward: np.ndarray, ground_z: float) -> tuple[str, np.ndarray, int, float]:
    ok, anchor = intersect_ray_with_plane_z(cam_xyz, forward, ground_z)
    if ok and np.all(np.isfinite(anchor)):
        return 'ray_ground_intersection', anchor, 0, float(np.linalg.norm(anchor - cam_xyz))
    return 'none', np.full(3, np.nan, dtype=np.float64), 0, np.nan


def bref_xy_anchor(row: pd.Series, ground_z: float) -> tuple[str, np.ndarray, int, float]:

    if bool(row.get('has_bref', False)) and np.isfinite(row.get('target_x', np.nan)) and np.isfinite(row.get('target_y', np.nan)):
        anchor = np.array([float(row['target_x']), float(row['target_y']), float(ground_z)], dtype=np.float64)
        cam = np.array([float(row['cam_x']), float(row['cam_y']), float(row['cam_z'])], dtype=np.float64)
        return 'bref_target_xy', anchor, 0, float(np.linalg.norm(anchor - cam))
    return 'none', np.full(3, np.nan, dtype=np.float64), 0, np.nan


def first_hit_pointcloud_anchor(cam_xyz: np.ndarray, forward: np.ndarray, points_xyz: np.ndarray, center_cfg: dict) -> tuple[str, np.ndarray, int, float]:


    radius = float(center_cfg.get('first_hit_ray_radius_m', 1.0))
    min_points = int(center_cfg.get('first_hit_min_points', 8))
    min_distance = float(center_cfg.get('first_hit_min_distance_m', 1.0))
    max_distance = float(center_cfg.get('first_hit_max_distance_m', center_cfg.get('max_range_m', 300.0)))
    depth_quantile = float(center_cfg.get('first_hit_depth_quantile', 0.05))
    cluster_depth = float(center_cfg.get('first_hit_cluster_depth_m', 1.0))

    ok, fwd = normalize_vector(forward)
    if not ok or radius <= 0.0 or min_points <= 0:
        return 'none', np.full(3, np.nan, dtype=np.float64), 0, np.nan

    rel = points_xyz - np.asarray(cam_xyz, dtype=np.float64)[None, :]
    along = rel @ fwd
    rel2 = np.einsum('ij,ij->i', rel, rel)
    perp2 = np.maximum(rel2 - along * along, 0.0)
    mask = (along >= min_distance) & (along <= max_distance) & (perp2 <= radius * radius)
    hit_count = int(np.count_nonzero(mask))
    if hit_count < min_points:
        return 'none', np.full(3, np.nan, dtype=np.float64), hit_count, np.nan

    hit_along = along[mask]
    depth_quantile = float(np.clip(depth_quantile, 0.0, 1.0))
    near_depth = float(np.quantile(hit_along, depth_quantile))
    cluster_mask = hit_along <= near_depth + max(0.0, cluster_depth)
    if int(np.count_nonzero(cluster_mask)) < min_points:
        order = np.argsort(hit_along)
        selected_along = hit_along[order[:min(min_points, hit_count)]]
    else:
        selected_along = hit_along[cluster_mask]
    anchor_range = float(np.median(selected_along))
    anchor = np.asarray(cam_xyz, dtype=np.float64) + fwd * anchor_range
    return 'first_hit_pointcloud', anchor, hit_count, anchor_range


def resolve_anchor(row: pd.Series, forward: np.ndarray, points_xyz: np.ndarray, ground_z: float, center_cfg: dict) -> tuple[str, np.ndarray, int, float, float]:
    cam_xyz = np.array([row['cam_x'], row['cam_y'], row['cam_z']], dtype=np.float64)
    method = str(center_cfg.get('anchor_method', 'first_hit_pointcloud')).lower().strip()
    fallback = str(center_cfg.get('first_hit_fallback', 'ray_ground_intersection')).lower().strip()
    ray_radius = float(center_cfg.get('first_hit_ray_radius_m', 1.0))

    if method == 'first_hit_pointcloud':
        source, anchor, hit_count, anchor_range = first_hit_pointcloud_anchor(cam_xyz, forward, points_xyz, center_cfg)
        if source != 'none':
            return source, anchor, hit_count, anchor_range, ray_radius
        if fallback == 'bref_target_xy':
            source, anchor, _, anchor_range = bref_xy_anchor(row, ground_z)
            if source != 'none':
                return 'fallback_bref_target_xy', anchor, hit_count, anchor_range, ray_radius
        source, anchor, _, anchor_range = ray_ground_anchor(cam_xyz, forward, ground_z)
        if source != 'none':
            return 'fallback_ray_ground_intersection', anchor, hit_count, anchor_range, ray_radius
        return 'none', anchor, hit_count, anchor_range, ray_radius

    if method == 'bref_target_xy' or bool(center_cfg.get('prefer_bref_target_xy', False)):
        source, anchor, hit_count, anchor_range = bref_xy_anchor(row, ground_z)
        if source != 'none':
            return source, anchor, hit_count, anchor_range, ray_radius

    source, anchor, hit_count, anchor_range = ray_ground_anchor(cam_xyz, forward, ground_z)
    return source, anchor, hit_count, anchor_range, ray_radius


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['outputs']['crs_dir'])

    pose_path = Path(cfg['outputs'].get('scene_quality_dir', cfg['outputs']['metadata_dir'])) / 'scene_pose_filtered.csv'
    if not pose_path.exists():
        pose_path = Path(cfg['outputs']['metadata_dir']) / 'scene_pose_filtered.csv'
    if not pose_path.exists():
        pose_path = Path(cfg['outputs']['metadata_dir']) / 'scene_pose_with_solar.csv'
    if not pose_path.exists():
        pose_path = Path(cfg['outputs']['metadata_dir']) / 'scene_pose_raw.csv'
    pose = read_table(pose_path)
    point = read_ply_xyz(cfg['paths']['pointcloud_ply'])
    x = np.asarray(point['x'], dtype=np.float64)
    y = np.asarray(point['y'], dtype=np.float64)
    z = np.asarray(point['z'], dtype=np.float64)
    points_xyz = np.column_stack([x, y, z])
    bbox = {
        'xmin': float(np.min(x)), 'xmax': float(np.max(x)),
        'ymin': float(np.min(y)), 'ymax': float(np.max(y)),
        'zmin': float(np.min(z)), 'zmax': float(np.max(z)),
    }

    angle_cfg = dict(cfg.get('camera_model', {}))
    candidate_file = cfg['crs']['candidate_file']
    candidate_path = Path(candidate_file)
    if not candidate_path.is_absolute():
        candidate_path = Path(cfg['_project_dir']) / candidate_file
    candidates = load_json(candidate_path)['candidates']
    forced = str(cfg['crs'].get('force_candidate_name', '')).strip()


    manual_offset_cfg = cfg['crs'].get('manual_vertical_offset_m', 0.0)
    if manual_offset_cfg is None:
        z_offset = 0.0
        vertical_note = 'no vertical offset applied (manual_vertical_offset_m=null)'
    else:
        z_offset = float(manual_offset_cfg)
        vertical_note = f'manual vertical offset applied: cam_z += {z_offset:.3f} m'

    scores = []
    for c in candidates:
        if forced and c['name'] != forced:
            continue
        scores.append(score_candidate(pose, c, bbox, angle_cfg, z_offset))
    if not scores:
        raise RuntimeError('没有可用的 CRS 候选')
    score_df = pd.DataFrame(scores).sort_values('score', ascending=False).reset_index(drop=True)
    best_name = score_df.loc[0, 'name']
    best = [c for c in candidates if c['name'] == best_name][0]

    cam_x, cam_y = transform_lonlat_to_projected(
        pose['cam_lon_wgs84'].fillna(0).astype(float).values,
        pose['cam_lat_wgs84'].fillna(0).astype(float).values,
        best['src_crs'], best['dst_crs'], best.get('post_ops', {}),
    )
    target_x, target_y = transform_lonlat_to_projected(
        pose['target_lon_wgs84'].fillna(0).astype(float).values,
        pose['target_lat_wgs84'].fillna(0).astype(float).values,
        best['src_crs'], best['dst_crs'], best.get('post_ops', {}),
    )
    cam_z = pose['cam_alt_raw'].astype(float).values.copy() + z_offset


    target_z = pose['target_alt_raw'].astype(float).values.copy() + z_offset

    out = pose.copy()
    out['cam_x'] = cam_x
    out['cam_y'] = cam_y
    out['cam_z'] = cam_z
    out['target_x'] = target_x
    out['target_y'] = target_y
    out['target_z'] = target_z
    out['horizontal_crs_name'] = best_name
    out['horizontal_crs_epsg'] = best['dst_crs']
    out['vertical_offset_m'] = z_offset
    out['pose_quality_flag'] = np.where(out['has_brdf'].fillna(False), 'usable', 'missing_brdf')


    forward_xyz = []
    for _, row in out.iterrows():
        if bool(row['has_brdf']) and pd.notna(row['view_azimuth_deg']) and pd.notna(row['view_zenith_deg']):
            fwd = build_camera_forward_vector(float(row['view_azimuth_deg']), float(row['view_zenith_deg']), angle_cfg)
        else:

            fwd = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        forward_xyz.append(fwd)
    forward_xyz = np.vstack(forward_xyz)
    out['forward_x'] = forward_xyz[:, 0]
    out['forward_y'] = forward_xyz[:, 1]
    out['forward_z'] = forward_xyz[:, 2]


    has_target = out['has_brdf'].fillna(False) & out['has_bref'].fillna(False)
    out['azimuth_residual_deg'] = np.nan
    out['zenith_residual_deg'] = np.nan
    if np.any(has_target):
        cam_xyz = out.loc[has_target, ['cam_x', 'cam_y', 'cam_z']].to_numpy(dtype=np.float64)

        tgt_xyz = out.loc[has_target, ['target_x', 'target_y']].to_numpy(dtype=np.float64)
        tgt_xyz = np.column_stack([tgt_xyz, cam_xyz[:, 2] - 1.0])
        az_res, ze_res = compute_view_residual_deg(
            cam_xyz, tgt_xyz,
            out.loc[has_target, 'view_azimuth_deg'].astype(float).values,
            out.loc[has_target, 'view_zenith_deg'].astype(float).values,
            angle_cfg,
        )
        out.loc[has_target, 'azimuth_residual_deg'] = az_res
        out.loc[has_target, 'zenith_residual_deg'] = ze_res


    center_cfg = cfg.get('center_estimation', {})
    ground_z = (
        float(center_cfg.get('manual_ground_z_m'))
        if center_cfg.get('manual_ground_z_m', None) is not None
        else float(np.quantile(z, float(center_cfg.get('ground_z_quantile', 0.05))))
    )
    anchors = []
    anchors_z = []
    anchor_rows = []
    updated_forward_xyz = forward_xyz.copy()
    use_anchor_forward = bool(center_cfg.get('use_anchor_to_update_forward', False))
    for row_pos, (_, row) in enumerate(out.iterrows()):
        scene_id = str(row['scene_id'])
        cam_xyz = np.array([row['cam_x'], row['cam_y'], row['cam_z']], dtype=np.float64)
        forward = forward_xyz[row_pos]
        source, anchor_xyz, hit_count, anchor_range, ray_radius = resolve_anchor(row, forward, points_xyz, ground_z, center_cfg)
        anchor_updates_forward = False
        if np.all(np.isfinite(anchor_xyz)):
            anchors.append(anchor_xyz[:2])
            anchors_z.append(float(anchor_xyz[2]))
            if use_anchor_forward:
                ok, anchor_forward = normalize_vector(anchor_xyz - cam_xyz)
                if ok:
                    updated_forward_xyz[row_pos] = anchor_forward
                    anchor_updates_forward = True
        anchor_rows.append({
            'scene_id': scene_id,
            'anchor_source': source,
            'anchor_x': float(anchor_xyz[0]) if np.isfinite(anchor_xyz[0]) else np.nan,
            'anchor_y': float(anchor_xyz[1]) if np.isfinite(anchor_xyz[1]) else np.nan,
            'anchor_z': float(anchor_xyz[2]) if np.isfinite(anchor_xyz[2]) else np.nan,
            'anchor_hit_count': int(hit_count),
            'anchor_range_m': float(anchor_range) if np.isfinite(anchor_range) else np.nan,
            'anchor_ray_radius_m': float(ray_radius),
            'anchor_updates_forward': bool(anchor_updates_forward),
        })

    out['forward_x'] = updated_forward_xyz[:, 0]
    out['forward_y'] = updated_forward_xyz[:, 1]
    out['forward_z'] = updated_forward_xyz[:, 2]


    if center_cfg.get('manual_center_xyz', None) is not None:
        center_xyz = np.asarray(center_cfg.get('manual_center_xyz'), dtype=np.float64)
    elif anchors:
        anchors_xy = np.asarray(anchors, dtype=np.float64)
        cx, cy = robust_center_xy(anchors_xy)
        if str(center_cfg.get('center_z_source', 'anchor_median')).lower().strip() == 'anchor_median' and anchors_z:
            center_z = float(np.median(np.asarray(anchors_z, dtype=np.float64)))
        else:
            center_z = ground_z
        center_xyz = np.array([cx, cy, center_z], dtype=np.float64)
    elif np.any(out['has_bref'].fillna(False)):
        cx = float(np.nanmedian(out['target_x'].astype(float).values))
        cy = float(np.nanmedian(out['target_y'].astype(float).values))
        center_xyz = np.array([cx, cy, ground_z], dtype=np.float64)
    else:
        center_xyz = np.array([float(np.median(x)), float(np.median(y)), ground_z], dtype=np.float64)

    cam_xyz_all = out[['cam_x', 'cam_y', 'cam_z']].to_numpy(dtype=np.float64)
    cam_dist_to_center = np.linalg.norm(cam_xyz_all - center_xyz[None, :], axis=1)
    hemi_radius = (
        float(np.nanmedian(cam_dist_to_center[np.isfinite(cam_dist_to_center)]))
        if np.any(np.isfinite(cam_dist_to_center))
        else np.nan
    )

    out['cam_local_x'] = out['cam_x'].astype(float) - float(center_xyz[0])
    out['cam_local_y'] = out['cam_y'].astype(float) - float(center_xyz[1])
    out['cam_local_z'] = out['cam_z'].astype(float) - float(center_xyz[2])
    out['target_local_x'] = out['target_x'].astype(float) - float(center_xyz[0])
    out['target_local_y'] = out['target_y'].astype(float) - float(center_xyz[1])
    out['target_local_z'] = out['target_z'].astype(float) - float(center_xyz[2])
    out['anchor_x'] = [r['anchor_x'] for r in anchor_rows]
    out['anchor_y'] = [r['anchor_y'] for r in anchor_rows]
    out['anchor_z'] = [r['anchor_z'] for r in anchor_rows]
    out['anchor_source'] = [r['anchor_source'] for r in anchor_rows]
    out['anchor_hit_count'] = [r['anchor_hit_count'] for r in anchor_rows]
    out['anchor_range_m'] = [r['anchor_range_m'] for r in anchor_rows]
    out['anchor_ray_radius_m'] = [r['anchor_ray_radius_m'] for r in anchor_rows]
    out['anchor_updates_forward'] = [r['anchor_updates_forward'] for r in anchor_rows]
    out['anchor_local_x'] = out['anchor_x'].astype(float) - float(center_xyz[0])
    out['anchor_local_y'] = out['anchor_y'].astype(float) - float(center_xyz[1])
    out['anchor_local_z'] = out['anchor_z'].astype(float) - float(center_xyz[2])
    cam_r, cam_az, cam_ze = compute_spherical_from_center(cam_xyz_all, center_xyz)
    out['cam_radius_from_center_m'] = cam_r
    out['cam_center_azimuth_deg'] = cam_az
    out['cam_center_zenith_deg'] = cam_ze

    anchor_source_counts: dict[str, int] = {}
    for r in anchor_rows:
        src = str(r.get('anchor_source', 'unknown'))
        anchor_source_counts[src] = anchor_source_counts.get(src, 0) + 1
    anchor_success = sum(v for k, v in anchor_source_counts.items() if 'none' not in k.lower())
    anchor_total = len(anchor_rows)

    write_table(score_df, out_dir / 'crs_score_report.csv', index=False)
    write_table(out, out_dir / 'scene_pose_aligned.csv', index=False)
    write_table(pd.DataFrame(anchor_rows), out_dir / 'scene_anchor_points.csv', index=False)
    save_json({
        'best_candidate': best,
        'point_bbox': bbox,
        'hemisphere_center_xyz': center_xyz.tolist(),
        'hemisphere_radius_m': hemi_radius,
        'ground_plane_z_m': ground_z,
        'anchor_method': str(center_cfg.get('anchor_method', 'first_hit_pointcloud')),
        'center_z_source': str(center_cfg.get('center_z_source', 'anchor_median')),
        'use_anchor_to_update_forward': use_anchor_forward,
        'v8new_vertical_offset': {
            'manual_vertical_offset_m': None if manual_offset_cfg is None else float(z_offset),
            'applied': float(z_offset),
            'note': vertical_note,
        },
        'v8new_anchor_summary': {
            'total': anchor_total,
            'success': anchor_success,
            'by_source': anchor_source_counts,
            'success_rate': float(anchor_success / anchor_total) if anchor_total > 0 else 0.0,
        },
    }, out_dir / 'hemi_solution.json')
    save_json({
        'vertical_offset_m': float(z_offset),
        'source': 'manual' if manual_offset_cfg is not None else 'none',
        'note': cfg['crs'].get('default_vertical_note', ''),
    }, out_dir / 'vertical_alignment.json')
    print(f'最佳 CRS 候选: {best_name}')
    print(f'半球中心: {center_xyz.tolist()}')
    print(f'半球半径(中位): {hemi_radius:.3f}')
    print(f'anchor 成功率: {anchor_success}/{anchor_total}')
    print(f'垂直偏移（手动）: z_offset={z_offset:.3f} m  ({vertical_note})')


if __name__ == '__main__':
    main()
