

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import load_config, read_table, write_table, ensure_dir, save_json


def _as_bool_series(s: pd.Series, default=False):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default).astype(bool)
    return s.fillna(str(default)).astype(str).str.strip().str.lower().isin(['1', 'true', 'yes', 'y'])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['outputs'].get('scene_quality_dir', Path(cfg['outputs']['metadata_dir']) / 'scene_quality'))
    inv_dir = Path(cfg['outputs']['inventory_dir'])
    meta_dir = Path(cfg['outputs']['metadata_dir'])
    solar_dir = Path(cfg['outputs'].get('solar_geometry_dir', meta_dir))

    manifest = read_table(inv_dir / 'scene_manifest.csv')
    pose_solar_path = solar_dir / 'scene_pose_with_solar.csv'
    if not pose_solar_path.exists():
        pose_solar_path = meta_dir / 'scene_pose_with_solar.csv'
    if pose_solar_path.exists():
        pose = read_table(pose_solar_path)
    else:
        pose = read_table(meta_dir / 'scene_pose_raw.csv')

    quality_cfg = dict(cfg.get('scene_quality_control', {}))
    enabled = bool(quality_cfg.get('enabled', True))
    exclude_scene_ids = {str(x).strip() for x in (quality_cfg.get('exclude_scene_ids', []) or []) if str(x).strip()}
    reasons_cfg = quality_cfg.get('exclusion_reason', {}) or {}
    if isinstance(reasons_cfg, str):
        reasons_cfg = {}
    require_brdf = bool(quality_cfg.get('require_brdf_pose', True))
    require_hdr_data = bool(quality_cfg.get('require_hdr_data', True))
    require_solar = bool(quality_cfg.get('require_solar_geometry', True))
    fail_on_mismatch = bool(quality_cfg.get('fail_on_scene_count_mismatch', False))

    report = manifest.copy()
    report['raw_in_manifest'] = True
    report['has_hdr'] = report.get('hdr_path', pd.Series('', index=report.index)).astype(str).str.strip().ne('')
    report['has_data'] = report.get('data_path', pd.Series('', index=report.index)).astype(str).str.strip().ne('')

    pose_keep_cols = [c for c in pose.columns if c != 'raw_in_manifest']
    report = report.merge(pose[pose_keep_cols], on='scene_id', how='left', suffixes=('', '_pose'))

    if 'has_brdf' in report.columns:
        report['has_brdf_pose'] = _as_bool_series(report['has_brdf'], default=False)
    else:
        report['has_brdf_pose'] = report['cam_lon_wgs84'].notna() if 'cam_lon_wgs84' in report.columns else False
    if 'solar_geometry_valid' in report.columns:
        report['has_solar_geometry'] = _as_bool_series(report['solar_geometry_valid'], default=False)
    else:
        report['has_solar_geometry'] = False

    report['manual_excluded'] = report['scene_id'].astype(str).isin(exclude_scene_ids)
    report['manual_exclusion_reason'] = report['scene_id'].astype(str).map(lambda s: str(reasons_cfg.get(s, 'manual_excluded')) if s in exclude_scene_ids else '')

    reject_reasons = []
    final_use = []
    for _, r in report.iterrows():
        reasons = []
        if enabled:
            if bool(r.get('manual_excluded', False)):
                reasons.append(str(r.get('manual_exclusion_reason', 'manual_excluded')) or 'manual_excluded')
            if require_hdr_data and not (bool(r.get('has_hdr', False)) and bool(r.get('has_data', False))):
                reasons.append('missing_hdr_or_data')
            if require_brdf and not bool(r.get('has_brdf_pose', False)):
                reasons.append('missing_brdf_pose')
            if require_solar and not bool(r.get('has_solar_geometry', False)):
                reasons.append('missing_solar_geometry')
        reject_reasons.append(';'.join(reasons))
        final_use.append(len(reasons) == 0)

    report['initial_reject_reason'] = reject_reasons
    report['final_use_flag'] = final_use

    used_scene_ids = report.loc[report['final_use_flag'], 'scene_id'].astype(str).tolist()

    manifest_filtered = manifest[manifest['scene_id'].astype(str).isin(used_scene_ids)].copy()
    pose_filtered = pose[pose['scene_id'].astype(str).isin(used_scene_ids)].copy()

    write_table(report, out_dir / 'scene_quality_report.csv', index=False)
    write_table(manifest_filtered, out_dir / 'scene_manifest_filtered.csv', index=False)
    write_table(pose_filtered, out_dir / 'scene_pose_filtered.csv', index=False)


    write_table(manifest_filtered, inv_dir / 'scene_manifest_filtered.csv', index=False)
    write_table(pose_filtered, meta_dir / 'scene_pose_filtered.csv', index=False)

    expected_raw = quality_cfg.get('expected_raw_scene_count', None)
    expected_used = quality_cfg.get('expected_used_scene_count', None)
    warnings = []
    if expected_raw is not None and int(expected_raw) != int(len(manifest)):
        warnings.append(f'raw scene count {len(manifest)} != expected_raw_scene_count {expected_raw}')
    if expected_used is not None and int(expected_used) != int(len(manifest_filtered)):
        warnings.append(f'used scene count {len(manifest_filtered)} != expected_used_scene_count {expected_used}')
    if warnings:
        print('[WARN] ' + ' | '.join(warnings))
        if fail_on_mismatch:
            raise RuntimeError('; '.join(warnings))

    save_json({
        'raw_scene_count': int(len(manifest)),
        'used_scene_count': int(len(manifest_filtered)),
        'manual_excluded_count': int(report['manual_excluded'].sum()),
        'warnings': warnings,
        'exclude_scene_ids': sorted(exclude_scene_ids),
    }, out_dir / 'scene_quality_summary.json')

    print(f'输出: {out_dir / "scene_quality_report.csv"}')
    print(f'原始场景数: {len(manifest)} | 初始保留场景数: {len(manifest_filtered)}')


if __name__ == '__main__':
    main()
