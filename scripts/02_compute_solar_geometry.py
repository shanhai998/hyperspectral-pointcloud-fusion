

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    ZoneInfo = None

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import load_config, read_table, write_table, ensure_dir, save_json


def _resolve_timezone(tz_name: str):
    name = str(tz_name or 'Asia/Shanghai').strip() or 'Asia/Shanghai'
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass

    normalized = name.lower().replace('_', '/')
    if normalized in {'utc', 'z', 'gmt'}:
        return timezone.utc
    if normalized in {'asia/shanghai', 'asia/chongqing', 'asia/harbin', 'asia/beijing', 'prc', 'cst', 'china'}:
        return timezone(timedelta(hours=8), name='Asia/Shanghai')
    return timezone(timedelta(hours=8), name='Asia/Shanghai')


def _parse_time(value, tz_name: str, assume_naive_local: bool = True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'nat', 'none'}:
        return pd.NaT

    ts = pd.to_datetime(text, errors='coerce')
    if pd.isna(ts):
        return pd.NaT
    tz = _resolve_timezone(tz_name)
    if ts.tzinfo is None:
        if assume_naive_local:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(tz)
    return ts


def _julian_day_from_utc(ts_utc: pd.Timestamp) -> float:

    return float(ts_utc.timestamp() / 86400.0 + 2440587.5)


def _solar_position_noaa(dt_local: pd.Timestamp, lat_deg: float, lon_deg: float):


    if pd.isna(dt_local) or not np.isfinite(lat_deg) or not np.isfinite(lon_deg):
        return np.nan, np.nan, np.nan

    dt_utc = dt_local.tz_convert(timezone.utc)
    jd = _julian_day_from_utc(dt_utc)
    T = (jd - 2451545.0) / 36525.0

    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)

    Mrad = np.deg2rad(M)
    C = (
        np.sin(Mrad) * (1.914602 - T * (0.004817 + 0.000014 * T))
        + np.sin(2.0 * Mrad) * (0.019993 - 0.000101 * T)
        + np.sin(3.0 * Mrad) * 0.000289
    )
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lambda_app = true_long - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))

    eps0 = 23.0 + (26.0 + ((21.448 - T * (46.815 + T * (0.00059 - T * 0.001813)))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * np.cos(np.deg2rad(omega))

    eps_rad = np.deg2rad(eps)
    lam_rad = np.deg2rad(lambda_app)
    decl_rad = np.arcsin(np.sin(eps_rad) * np.sin(lam_rad))

    y = np.tan(eps_rad / 2.0) ** 2
    L0_rad = np.deg2rad(L0)
    eq_time = 4.0 * np.rad2deg(
        y * np.sin(2.0 * L0_rad)
        - 2.0 * e * np.sin(Mrad)
        + 4.0 * e * y * np.sin(Mrad) * np.cos(2.0 * L0_rad)
        - 0.5 * y * y * np.sin(4.0 * L0_rad)
        - 1.25 * e * e * np.sin(2.0 * Mrad)
    )


    utc = dt_utc.to_pydatetime()
    minutes_utc = utc.hour * 60.0 + utc.minute + utc.second / 60.0 + utc.microsecond / 60000000.0
    true_solar_time = (minutes_utc + eq_time + 4.0 * lon_deg) % 1440.0
    hour_angle_deg = true_solar_time / 4.0 - 180.0
    if hour_angle_deg < -180.0:
        hour_angle_deg += 360.0

    lat_rad = np.deg2rad(lat_deg)
    ha_rad = np.deg2rad(hour_angle_deg)
    cos_zen = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)
    )
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zenith_deg = float(np.rad2deg(np.arccos(cos_zen)))
    elevation_deg = 90.0 - zenith_deg


    az_rad = np.arctan2(
        np.sin(ha_rad),
        np.cos(ha_rad) * np.sin(lat_rad) - np.tan(decl_rad) * np.cos(lat_rad),
    )
    azimuth_deg = (float(np.rad2deg(az_rad)) + 180.0) % 360.0
    return zenith_deg, elevation_deg, azimuth_deg


def _sun_vector_from_zenith_azimuth(zenith_deg: float, azimuth_deg: float):
    if not np.isfinite(zenith_deg) or not np.isfinite(azimuth_deg):
        return np.nan, np.nan, np.nan
    th = np.deg2rad(zenith_deg)
    ph = np.deg2rad(azimuth_deg)

    sx = np.sin(th) * np.sin(ph)
    sy = np.sin(th) * np.cos(ph)
    sz = np.cos(th)
    return float(sx), float(sy), float(sz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['outputs'].get('solar_geometry_dir', Path(cfg['outputs']['metadata_dir']) / 'solar_geometry'))
    meta_dir = Path(cfg['outputs']['metadata_dir'])
    pose_path = meta_dir / 'scene_pose_raw.csv'
    pose = read_table(pose_path)

    sol_cfg = dict(cfg.get('solar_geometry', {}))
    enabled = bool(sol_cfg.get('enabled', True))
    tz_name = str(sol_cfg.get('timezone', 'Asia/Shanghai'))
    assume_local = bool(sol_cfg.get('assume_naive_time_is_local', True))
    min_elev = float(sol_cfg.get('min_solar_elevation_deg', 5.0))

    rows = []
    for _, r in pose.iterrows():
        scene_id = str(r['scene_id'])
        t_raw = r.get('acquisition_time_raw', '')
        lat = float(r.get('cam_lat_wgs84', np.nan))
        lon = float(r.get('cam_lon_wgs84', np.nan))
        if enabled:
            ts = _parse_time(t_raw, tz_name=tz_name, assume_naive_local=assume_local)
            if pd.isna(ts):
                zen = elev = az = np.nan
                ts_local = ''
                ts_utc = ''
            else:
                zen, elev, az = _solar_position_noaa(ts, lat, lon)
                ts_local = ts.isoformat()
                ts_utc = ts.tz_convert(timezone.utc).isoformat()
        else:
            zen = elev = az = np.nan
            ts_local = ''
            ts_utc = ''
        sx, sy, sz = _sun_vector_from_zenith_azimuth(zen, az)
        rows.append({
            'scene_id': scene_id,
            'acquisition_time_raw': t_raw,
            'acquisition_time_source': r.get('acquisition_time_source', ''),
            'datetime_local_iso': ts_local,
            'datetime_utc_iso': ts_utc,
            'solar_zenith_deg': zen,
            'solar_elevation_deg': elev,
            'solar_azimuth_deg': az,
            'sun_dir_x': sx,
            'sun_dir_y': sy,
            'sun_dir_z': sz,
            'solar_geometry_valid': bool(np.isfinite(zen) and np.isfinite(az)),
            'solar_low_elevation_flag': bool(np.isfinite(elev) and elev < min_elev),
            'solar_time_timezone': tz_name,
            'solar_time_assumption': 'naive_as_local' if assume_local else 'naive_as_utc',
        })

    solar = pd.DataFrame(rows)
    write_table(solar, out_dir / 'scene_solar_geometry.csv', index=False)

    merged = pose.merge(solar.drop(columns=['acquisition_time_raw', 'acquisition_time_source'], errors='ignore'), on='scene_id', how='left')
    write_table(merged, out_dir / 'scene_pose_with_solar.csv', index=False)

    write_table(merged, meta_dir / 'scene_pose_with_solar.csv', index=False)

    save_json({
        'scene_count': int(len(solar)),
        'valid_solar_geometry_count': int(solar['solar_geometry_valid'].sum()) if len(solar) else 0,
        'timezone': tz_name,
        'assume_naive_time_is_local': assume_local,
        'note': 'Solar geometry was computed from the GPS时间 column parsed in step 01.'
    }, out_dir / 'solar_geometry_summary.json')

    print(f'输出: {out_dir / "scene_solar_geometry.csv"}')
    print(f'有效太阳几何: {int(solar["solar_geometry_valid"].sum())}/{len(solar)}')


if __name__ == '__main__':
    main()
