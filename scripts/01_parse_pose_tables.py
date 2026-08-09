
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import load_config, read_table, write_table, ensure_dir, normalize_text, extract_scene_key_from_excel_label


def build_column_map(df: pd.DataFrame):
    return {normalize_text(c): c for c in df.columns}


def require_col(col_map, aliases):
    for a in aliases:
        k = normalize_text(a)
        if k in col_map:
            return col_map[k]
    raise KeyError(f'缺少列: {aliases}')


def optional_col(col_map, aliases):
    for a in aliases:
        k = normalize_text(a)
        if k in col_map:
            return col_map[k]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['outputs']['metadata_dir'])

    manifest = read_table(Path(cfg['outputs']['inventory_dir']) / 'scene_manifest.csv')
    brdf = read_table(cfg['paths']['brdf_excel'])
    bref = read_table(cfg['paths']['bref_excel'])

    brdf_cols = build_column_map(brdf)
    bref_cols = build_column_map(bref)

    brdf_label = require_col(brdf_cols, ['#Label'])
    brdf_lon = require_col(brdf_cols, ['相机位置（经度）'])
    brdf_lat = require_col(brdf_cols, ['相机位置（纬度）'])
    brdf_alt = require_col(brdf_cols, ['相机位置（海拔）'])
    brdf_zen = require_col(brdf_cols, ['观测天顶角'])
    brdf_azi = require_col(brdf_cols, ['观测方位角'])
    brdf_gps_time = optional_col(brdf_cols, ['GPS时间', 'gps时间', 'GPSTime', 'gps_time', 'acquisition_time'])

    bref_label = require_col(bref_cols, ['#Label'])
    bref_lon = require_col(bref_cols, ['X/Longitude', 'Longitude'])
    bref_lat = require_col(bref_cols, ['Y/Latitude', 'Latitude'])
    bref_alt = require_col(bref_cols, ['Z/Altitude', 'Altitude'])
    bref_azi = require_col(bref_cols, ['方位角'])
    bref_zen = require_col(bref_cols, ['天顶角'])
    bref_gps_time = optional_col(bref_cols, ['GPS时间', 'gps时间', 'GPSTime', 'gps_time', 'acquisition_time'])

    brdf = brdf.copy()
    bref = bref.copy()
    brdf['scene_id'] = brdf[brdf_label].map(extract_scene_key_from_excel_label)
    bref['scene_id'] = bref[bref_label].map(extract_scene_key_from_excel_label)

    write_table(brdf, out_dir / 'brdf_raw.csv', index=False)
    write_table(bref, out_dir / 'bref_raw.csv', index=False)

    brdf_map = brdf.drop_duplicates('scene_id').set_index('scene_id')
    bref_map = bref.drop_duplicates('scene_id').set_index('scene_id')

    rows = []
    for _, r in manifest.iterrows():
        scene_id = r['scene_id']
        has_brdf = scene_id in brdf_map.index
        has_bref = scene_id in bref_map.index
        row = {'scene_id': scene_id, 'has_brdf': has_brdf, 'has_bref': has_bref}
        if has_brdf:
            b = brdf_map.loc[scene_id]
            row.update({
                'label_pan_brdf': b[brdf_label],
                'cam_lon_wgs84': float(b[brdf_lon]),
                'cam_lat_wgs84': float(b[brdf_lat]),
                'cam_alt_raw': float(b[brdf_alt]),
                'view_zenith_deg': float(b[brdf_zen]),
                'view_azimuth_deg': float(b[brdf_azi]),
                'gps_time_brdf': '' if brdf_gps_time is None or pd.isna(b[brdf_gps_time]) else str(b[brdf_gps_time]),
            })
        else:
            row.update({'label_pan_brdf': '', 'cam_lon_wgs84': np.nan, 'cam_lat_wgs84': np.nan, 'cam_alt_raw': np.nan, 'view_zenith_deg': np.nan, 'view_azimuth_deg': np.nan, 'gps_time_brdf': ''})
        if has_bref:
            b = bref_map.loc[scene_id]
            row.update({
                'label_pan_bref': b[bref_label],
                'target_lon_wgs84': float(b[bref_lon]),
                'target_lat_wgs84': float(b[bref_lat]),
                'target_alt_raw': float(b[bref_alt]),
                'target_azimuth_deg': float(b[bref_azi]),
                'target_zenith_deg': float(b[bref_zen]),
                'gps_time_bref': '' if bref_gps_time is None or pd.isna(b[bref_gps_time]) else str(b[bref_gps_time]),
            })
        else:
            row.update({'label_pan_bref': '', 'target_lon_wgs84': np.nan, 'target_lat_wgs84': np.nan, 'target_alt_raw': np.nan, 'target_azimuth_deg': np.nan, 'target_zenith_deg': np.nan, 'gps_time_bref': ''})
        row['match_status'] = 'matched_brdf_bref' if (has_brdf and has_bref) else ('matched_brdf_only' if has_brdf else ('matched_bref_only' if has_bref else 'missing_both'))
        rows.append(row)

    out = pd.DataFrame(rows)
    prefer = str(cfg.get('solar_geometry', {}).get('prefer_time_source', 'brdf')).strip().lower()
    if prefer == 'bref':
        first_col, second_col = 'gps_time_bref', 'gps_time_brdf'
    else:
        first_col, second_col = 'gps_time_brdf', 'gps_time_bref'
    out['acquisition_time_raw'] = out[first_col].where(out[first_col].astype(str).str.strip() != '', out[second_col])
    out['acquisition_time_source'] = np.where(out[first_col].astype(str).str.strip() != '', first_col.replace('gps_time_', ''), np.where(out[second_col].astype(str).str.strip() != '', second_col.replace('gps_time_', ''), 'missing'))
    write_table(out, out_dir / 'scene_pose_raw.csv', index=False)
    write_table(out[['scene_id', 'match_status']], out_dir / 'pose_match_report.csv', index=False)
    print(f'输出: {out_dir / "scene_pose_raw.csv"}')
    print(out['match_status'].value_counts(dropna=False))


if __name__ == '__main__':
    main()
