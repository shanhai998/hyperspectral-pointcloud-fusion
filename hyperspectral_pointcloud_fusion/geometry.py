

import math
from typing import Dict, Tuple

import numpy as np
from pyproj import Transformer


def rotation_matrix_axis(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / n
    a = math.radians(float(deg))
    c = math.cos(a)
    s = math.sin(a)
    C = 1.0 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ], dtype=np.float64)


def build_observation_direction_vector(azimuth_deg: float, zenith_deg: float, azimuth_reference: str = 'north_cw', zenith_reference: str = 'from_nadir') -> np.ndarray:


    az_ref = str(azimuth_reference).lower().strip()
    ze_ref = str(zenith_reference).lower().strip()
    az = math.radians(float(azimuth_deg))
    ze = math.radians(float(zenith_deg))
    if ze_ref in {'from_up', 'from_zenith'}:
        ze_from_up = ze
    elif ze_ref in {'from_down', 'from_nadir'}:
        ze_from_up = math.pi - ze
    else:
        raise ValueError(f'暂不支持的 zenith_reference: {zenith_reference}')
    if az_ref == 'north_cw':
        vec = np.array([
            math.sin(ze_from_up) * math.sin(az),
            math.sin(ze_from_up) * math.cos(az),
            math.cos(ze_from_up),
        ], dtype=np.float64)
    elif az_ref == 'east_ccw':
        vec = np.array([
            math.sin(ze_from_up) * math.cos(az),
            math.sin(ze_from_up) * math.sin(az),
            math.cos(ze_from_up),
        ], dtype=np.float64)
    else:
        raise ValueError(f'暂不支持的 azimuth_reference: {azimuth_reference}')
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return vec / n


def build_camera_forward_vector(azimuth_deg: float, zenith_deg: float, angle_cfg: Dict) -> np.ndarray:


    obs = build_observation_direction_vector(
        azimuth_deg=azimuth_deg,
        zenith_deg=zenith_deg,
        azimuth_reference=str(angle_cfg.get('azimuth_reference', 'north_cw')),
        zenith_reference=str(angle_cfg.get('zenith_reference', 'from_nadir')),
    )
    observation_is_target_to_sensor = bool(angle_cfg.get('observation_is_target_to_sensor', False))
    camera_forward_is_opposite_observation = bool(angle_cfg.get('camera_forward_is_opposite_observation', False))
    forward = obs.copy()

    if observation_is_target_to_sensor and camera_forward_is_opposite_observation:
        forward = -obs
    elif (not observation_is_target_to_sensor) and camera_forward_is_opposite_observation:
        forward = -obs
    n = float(np.linalg.norm(forward))
    if n <= 1e-12:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return forward / n


def build_camera_frame(forward: np.ndarray, world_up: np.ndarray | None = None, roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.asarray(forward, dtype=np.float64)
    forward /= max(1e-12, float(np.linalg.norm(forward)))
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        world_up = np.asarray(world_up, dtype=np.float64)
        world_up /= max(1e-12, float(np.linalg.norm(world_up)))
    right = np.cross(forward, world_up)
    nr = float(np.linalg.norm(right))
    if nr <= 1e-12:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, forward))) > 0.95:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback)
        nr = float(np.linalg.norm(right))
    right /= max(1e-12, nr)
    down = np.cross(forward, right)
    down /= max(1e-12, float(np.linalg.norm(down)))
    if abs(float(roll_deg)) > 1e-9:
        Rr = rotation_matrix_axis(forward, roll_deg)
        right = Rr @ right
        down = Rr @ down
    return right, down, forward


def build_world_to_camera_matrix(right: np.ndarray, down: np.ndarray, forward: np.ndarray) -> np.ndarray:
    return np.vstack([
        np.asarray(right, dtype=np.float64),
        np.asarray(down, dtype=np.float64),
        np.asarray(forward, dtype=np.float64),
    ])


def project_world_points_pinhole(xyz: np.ndarray, cam_xyz: np.ndarray, world_to_camera_R: np.ndarray, width: int, height: int, fov_h_deg: float, fov_v_deg: float, flip_u: bool = False, flip_v: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rel = np.asarray(xyz, dtype=np.float64) - np.asarray(cam_xyz, dtype=np.float64)[None, :]
    pc = (np.asarray(world_to_camera_R, dtype=np.float64) @ rel.T).T
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    fov_h = math.radians(float(fov_h_deg))
    fov_v = math.radians(float(fov_v_deg))
    fx = ((float(width) - 1.0) * 0.5) / math.tan(fov_h * 0.5)
    fy = ((float(height) - 1.0) * 0.5) / math.tan(fov_v * 0.5)
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    valid_z = z > 1e-8
    u = np.full(z.shape, np.nan, dtype=np.float64)
    v = np.full(z.shape, np.nan, dtype=np.float64)
    u[valid_z] = fx * (x[valid_z] / z[valid_z]) + cx
    v[valid_z] = fy * (y[valid_z] / z[valid_z]) + cy
    if flip_u:
        u = (float(width) - 1.0) - u
    if flip_v:
        v = (float(height) - 1.0) - v
    inside = valid_z & (u >= 0.0) & (u <= (float(width) - 1.0)) & (v >= 0.0) & (v <= (float(height) - 1.0))
    rng = np.linalg.norm(rel, axis=1)
    offaxis = np.degrees(np.arctan2(np.sqrt(x * x + y * y), np.maximum(z, 1e-12)))
    border_dist = np.minimum.reduce([u, v, (float(width) - 1.0) - u, (float(height) - 1.0) - v])
    return u, v, rng, offaxis, border_dist, inside


def make_transformer(src_crs: str, dst_crs: str) -> Transformer:
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def apply_post_ops(x: np.ndarray, y: np.ndarray, ops: Dict) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    if ops.get('swap_xy', False):
        x, y = y, x
    if ops.get('subtract_x_zone_prefix_million') is not None:
        x -= float(ops['subtract_x_zone_prefix_million']) * 1e6
    if ops.get('subtract_y_zone_prefix_million') is not None:
        y -= float(ops['subtract_y_zone_prefix_million']) * 1e6
    x += float(ops.get('x_offset_m', 0.0) or 0.0)
    y += float(ops.get('y_offset_m', 0.0) or 0.0)
    return x, y


def transform_lonlat_to_projected(lon, lat, src_crs: str, dst_crs: str, ops: Dict) -> Tuple[np.ndarray, np.ndarray]:
    transformer = make_transformer(src_crs, dst_crs)
    x, y = transformer.transform(lon, lat)
    return apply_post_ops(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), ops)


def compute_view_residual_deg(cam_xyz: np.ndarray, target_xyz: np.ndarray, view_azimuth_deg: np.ndarray, view_zenith_deg: np.ndarray, angle_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:


    if bool(angle_cfg.get('observation_is_target_to_sensor', False)):
        obs_vec = np.asarray(cam_xyz, dtype=np.float64) - np.asarray(target_xyz, dtype=np.float64)
    else:
        obs_vec = np.asarray(target_xyz, dtype=np.float64) - np.asarray(cam_xyz, dtype=np.float64)
    dist = np.linalg.norm(obs_vec, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        if str(angle_cfg.get('azimuth_reference', 'north_cw')).lower().strip() == 'north_cw':
            pred_az = np.degrees(np.arctan2(obs_vec[:, 0], obs_vec[:, 1]))
        else:
            pred_az = np.degrees(np.arctan2(obs_vec[:, 1], obs_vec[:, 0]))
        pred_az = np.mod(pred_az, 360.0)

        pred_zenith = np.degrees(np.arccos(np.clip(obs_vec[:, 2] / np.maximum(dist, 1e-12), -1.0, 1.0)))
        if str(angle_cfg.get('zenith_reference', 'from_nadir')).lower().strip() in {'from_down', 'from_nadir'}:
            pred_zenith = 180.0 - pred_zenith
    az_res = np.abs(((pred_az - view_azimuth_deg + 180.0) % 360.0) - 180.0)
    ze_res = np.abs(pred_zenith - view_zenith_deg)
    return az_res, ze_res


def intersect_ray_with_plane_z(origin_xyz: np.ndarray, direction_xyz: np.ndarray, plane_z: float) -> tuple[bool, np.ndarray]:
    origin = np.asarray(origin_xyz, dtype=np.float64)
    direction = np.asarray(direction_xyz, dtype=np.float64)
    dz = float(direction[2])
    if abs(dz) <= 1e-12:
        return False, np.full(3, np.nan, dtype=np.float64)
    t = (float(plane_z) - float(origin[2])) / dz
    if t <= 0.0:
        return False, np.full(3, np.nan, dtype=np.float64)
    return True, origin + t * direction


def compute_spherical_from_center(points_xyz: np.ndarray, center_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = np.asarray(points_xyz, dtype=np.float64) - np.asarray(center_xyz, dtype=np.float64)[None, :]
    r = np.linalg.norm(rel, axis=1)
    az = np.degrees(np.arctan2(rel[:, 0], rel[:, 1]))
    az = np.mod(az, 360.0)
    ze = np.degrees(np.arccos(np.clip(rel[:, 2] / np.maximum(r, 1e-12), -1.0, 1.0)))
    return r, az, ze
