

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from hyperspectral_pointcloud_fusion.common import load_config, ensure_dir, load_json, save_json


def load_scene_meta(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_scene_meta(meta: dict, path: Path) -> None:

    bak = path.with_suffix(path.suffix + '.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def compute_intrinsics(cam_model: dict) -> tuple[float, float, float, float]:
    W = float(cam_model['image_width'])
    H = float(cam_model['image_height'])
    fov_h = float(cam_model['fov_h_deg'])
    fov_v = float(cam_model['fov_v_deg'])
    fx = ((W - 1.0) * 0.5) / np.tan(np.deg2rad(fov_h) * 0.5)
    fy = ((H - 1.0) * 0.5) / np.tan(np.deg2rad(fov_v) * 0.5)
    cx = (W - 1.0) * 0.5
    cy = (H - 1.0) * 0.5
    return fx, fy, cx, cy


def project_with_correction(
    points_world: np.ndarray,
    cam_xyz: np.ndarray,
    R_wc_original: np.ndarray,
    R_corr: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    flip_u: bool, flip_v: bool,
    W: int, H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    rel = np.atleast_2d(np.asarray(points_world, dtype=np.float64)) - np.asarray(cam_xyz, dtype=np.float64)[None, :]
    pc = (R_corr @ R_wc_original @ rel.T).T
    z = pc[:, 2]
    z_safe = np.where(np.abs(z) > 1e-8, z, 1e-8)
    u = fx * (pc[:, 0] / z_safe) + cx
    v = fy * (pc[:, 1] / z_safe) + cy
    if flip_u:
        u = (W - 1.0) - u
    if flip_v:
        v = (H - 1.0) - v
    return u, v, z


def residuals_full(
    rvec: np.ndarray,
    top_world: np.ndarray, base_world: np.ndarray,
    u_top_target: float, v_top_target: float,
    u_base_target: float, v_base_target: float,
    cam_xyz: np.ndarray, R_wc_original: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    flip_u: bool, flip_v: bool, W: int, H: int,
) -> np.ndarray:
    R_corr = Rotation.from_rotvec(rvec).as_matrix()
    pts = np.stack([top_world, base_world], axis=0)
    u, v, z = project_with_correction(pts, cam_xyz, R_wc_original, R_corr,
                                      fx, fy, cx, cy, flip_u, flip_v, W, H)

    penalty = np.where(z > 1e-6, 0.0, 1e6)
    return np.array([
        u[0] - u_top_target + penalty[0],
        v[0] - v_top_target + penalty[0],
        u[1] - u_base_target + penalty[1],
        v[1] - v_base_target + penalty[1],
    ], dtype=np.float64)


def residuals_single(
    rvec_xy: np.ndarray,
    pt_world: np.ndarray,
    u_target: float, v_target: float,
    cam_xyz: np.ndarray, R_wc_original: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    flip_u: bool, flip_v: bool, W: int, H: int,
) -> np.ndarray:

    rvec = np.array([float(rvec_xy[0]), float(rvec_xy[1]), 0.0], dtype=np.float64)
    R_corr = Rotation.from_rotvec(rvec).as_matrix()
    u, v, z = project_with_correction(pt_world[None, :], cam_xyz, R_wc_original, R_corr,
                                      fx, fy, cx, cy, flip_u, flip_v, W, H)
    penalty = 0.0 if z[0] > 1e-6 else 1e6
    return np.array([u[0] - u_target + penalty, v[0] - v_target + penalty], dtype=np.float64)


def clip_rotation_angle(rvec: np.ndarray, max_angle_deg: float) -> tuple[np.ndarray, bool]:
    angle_rad = float(np.linalg.norm(rvec))
    max_rad = np.deg2rad(float(max_angle_deg))
    if angle_rad > max_rad and angle_rad > 1e-12:
        return rvec * (max_rad / angle_rad), True
    return rvec, False


SOLVED_STATUSES = {'full', 'top_only', 'base_only'}
DEFAULT_R_CORR_TOTAL_ANGLE_THRESHOLDS = {'min': 5.0, 'median': 5.0, 'mean': 15.0}


def get_original_world_to_camera_R(scene_meta: dict) -> np.ndarray:

    source = scene_meta.get('world_to_camera_R_original', scene_meta['world_to_camera_R'])
    return np.asarray(source, dtype=np.float64)


def get_quality_gate_config(target_cfg: dict) -> tuple[bool, dict[str, float]]:
    gate_cfg = dict(target_cfg.get('quality_gate', {}) or {})
    enabled = bool(gate_cfg.get('enabled', True))
    raw = (
        gate_cfg.get('r_corr_total_angle_deg')
        or target_cfg.get('r_corr_total_angle_deg_thresholds')
        or {}
    )
    thresholds = dict(DEFAULT_R_CORR_TOTAL_ANGLE_THRESHOLDS)
    if isinstance(raw, dict):
        for key in ('min', 'median', 'mean'):
            if raw.get(key) is not None:
                thresholds[key] = float(raw[key])
    return enabled, thresholds


def evaluate_quality_gate(df: pd.DataFrame, thresholds: dict[str, float]) -> tuple[bool, list[str], dict[str, float]]:
    messages: list[str] = []
    stats = {'min': np.nan, 'median': np.nan, 'mean': np.nan, 'max': np.nan}
    if df.empty or 'annotation_status' not in df.columns:
        return False, ['06 did not produce any correction rows; check annotation CSV and scene_meta paths.'], stats

    solved = df[df['annotation_status'].isin(SOLVED_STATUSES)].copy()
    if solved.empty:
        return False, ['No scenes are marked full/top_only/base_only; run 05 and annotate at least one usable scene.'], stats

    if 'success' in solved.columns:
        success = solved['success'].fillna(False).astype(bool)
        if not bool(success.all()):
            bad = solved.loc[~success, 'scene_id'].astype(str).head(10).tolist()
            messages.append(f'Solver failed for {int((~success).sum())} scene(s): {", ".join(bad)}')

    if 'clipped_by_max_angle' in solved.columns:
        clipped = solved['clipped_by_max_angle'].fillna(False).astype(bool)
        if bool(clipped.any()):
            bad = solved.loc[clipped, 'scene_id'].astype(str).head(10).tolist()
            messages.append(f'{int(clipped.sum())} scene(s) were clipped by --max-angle-deg: {", ".join(bad)}')

    angles = pd.to_numeric(solved['rotation_total_deg'], errors='coerce').dropna()
    if angles.empty:
        messages.append('No finite rotation_total_deg values were produced.')
    else:
        stats = {
            'min': float(angles.min()),
            'median': float(angles.median()),
            'mean': float(angles.mean()),
            'max': float(angles.max()),
        }
        for key in ('min', 'median', 'mean'):
            limit = float(thresholds[key])
            if stats[key] > limit:
                messages.append(
                    f'R_corr total angle {key}={stats[key]:.3f} deg exceeds threshold {limit:.3f} deg.'
                )

    return len(messages) == 0, messages, stats


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors='coerce')


def add_scene_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['rotation_total_deg'] = numeric_series(out, 'rotation_total_deg')
    for prefix in ('pre', 'post'):
        target_cols = []
        for target in ('top', 'base'):
            u = numeric_series(out, f'{prefix}_err_u_{target}_px')
            v = numeric_series(out, f'{prefix}_err_v_{target}_px')
            col = f'{prefix}_{target}_pixel_error_px'
            out[col] = np.sqrt(u * u + v * v)
            target_cols.append(col)
        out[f'{prefix}_pixel_error_max_px'] = out[target_cols].max(axis=1, skipna=True)
        out[f'{prefix}_pixel_error_mean_px'] = out[target_cols].mean(axis=1, skipna=True)
    return out


def print_top_error_scenes(df: pd.DataFrame, top_n: int = 5) -> None:
    if df.empty:
        return

    view = add_scene_error_columns(df)

    def _fmt(value: object) -> str:
        try:
            if pd.isna(value):
                return ''
            return f'{float(value):.3f}'
        except Exception:
            return str(value)

    def _print(title: str, sort_col: str, cols: list[str]) -> None:
        if sort_col not in view.columns:
            return
        top = view.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False).head(top_n)
        if top.empty:
            print(f'\n=== {title} ===')
            print('  No finite values.')
            return
        display_cols = [c for c in cols if c in top.columns]
        print(f'\n=== {title} ===')
        print(top[display_cols].to_string(index=False, formatters={
            c: _fmt for c in display_cols
            if c not in {'scene_id', 'annotation_status', 'success', 'clipped_by_max_angle'}
        }))

    _print(
        f'Top {top_n} scenes by correction angle',
        'rotation_total_deg',
        [
            'scene_id',
            'annotation_status',
            'rotation_total_deg',
            'post_pixel_error_max_px',
            'pre_pixel_error_max_px',
            'clipped_by_max_angle',
            'success',
        ],
    )
    _print(
        f'Top {top_n} scenes by post-correction pixel error',
        'post_pixel_error_max_px',
        [
            'scene_id',
            'annotation_status',
            'post_pixel_error_max_px',
            'post_pixel_error_mean_px',
            'pre_pixel_error_max_px',
            'rotation_total_deg',
            'clipped_by_max_angle',
            'success',
        ],
    )


def apply_corrections_to_scene_meta(results: list[dict], scene_root: Path) -> int:
    applied = 0
    for res in results:
        if not bool(res.get('success', False)) or str(res.get('annotation_status')) == 'none':
            continue

        sid = str(res['scene_id'])
        meta_path = scene_root / sid / 'scene_meta.json'
        if not meta_path.exists():
            print(f'[skip apply] {sid}: scene_meta.json not found')
            continue

        scene_meta = load_scene_meta(meta_path)
        R_corr = np.asarray(res['R_corr'], dtype=np.float64)
        R_orig = get_original_world_to_camera_R(scene_meta)
        scene_meta['world_to_camera_R_original'] = R_orig.tolist()
        scene_meta['world_to_camera_R'] = (R_corr @ R_orig).tolist()
        scene_meta['target_calibration'] = {
            'annotation_status': res['annotation_status'],
            'solver_dof': res['solver_dof'],
            'rx_deg': res['rx_deg'], 'ry_deg': res['ry_deg'], 'rz_deg': res['rz_deg'],
            'rotation_total_deg': res['rotation_total_deg'],
            'clipped_by_max_angle': res['clipped_by_max_angle'],
            'pre_err_u_top_px': res['pre_err_u_top_px'],
            'pre_err_v_top_px': res['pre_err_v_top_px'],
            'pre_err_u_base_px': res['pre_err_u_base_px'],
            'pre_err_v_base_px': res['pre_err_v_base_px'],
            'post_err_u_top_px': res['post_err_u_top_px'],
            'post_err_v_top_px': res['post_err_v_top_px'],
            'post_err_u_base_px': res['post_err_u_base_px'],
            'post_err_v_base_px': res['post_err_v_base_px'],
        }
        write_scene_meta(scene_meta, meta_path)
        applied += 1
    return applied


def solve_one(ann_row: pd.Series, scene_meta: dict, max_angle_deg: float) -> dict:
    status = str(ann_row['annotation_status'])
    cam_xyz = np.asarray(scene_meta['camera_xyz'], dtype=np.float64)
    R_wc_original = get_original_world_to_camera_R(scene_meta)
    cm = scene_meta['camera_model']
    W = int(cm['image_width'])
    H = int(cm['image_height'])
    fx, fy, cx, cy = compute_intrinsics(cm)
    flip_u = bool(cm.get('flip_u', False))
    flip_v = bool(cm.get('flip_v', False))

    top_w = np.array([ann_row['top_anchor_x'], ann_row['top_anchor_y'], ann_row['top_anchor_z']], dtype=np.float64)
    base_w = np.array([ann_row['base_anchor_x'], ann_row['base_anchor_y'], ann_row['base_anchor_z']], dtype=np.float64)

    result = {
        'scene_id': str(ann_row['scene_id']),
        'annotation_status': status,
        'pre_err_u_top_px': np.nan, 'pre_err_v_top_px': np.nan,
        'pre_err_u_base_px': np.nan, 'pre_err_v_base_px': np.nan,
        'post_err_u_top_px': np.nan, 'post_err_v_top_px': np.nan,
        'post_err_u_base_px': np.nan, 'post_err_v_base_px': np.nan,
        'rx_deg': 0.0, 'ry_deg': 0.0, 'rz_deg': 0.0,
        'rotation_total_deg': 0.0,
        'solver_dof': 0,
        'clipped_by_max_angle': False,
        'success': False,
        'note': '',
        'R_corr': np.eye(3, dtype=np.float64).tolist(),
    }

    if status == 'none':
        result['success'] = True
        result['note'] = 'status=none, no correction applied'
        return result


    def _project_plain(pw):
        u, v, _ = project_with_correction(pw[None, :], cam_xyz, R_wc_original, np.eye(3),
                                          fx, fy, cx, cy, flip_u, flip_v, W, H)
        return float(u[0]), float(v[0])
    if status in ('full', 'top_only'):
        u_t_pred, v_t_pred = _project_plain(top_w)
        result['pre_err_u_top_px'] = u_t_pred - float(ann_row['u_top'])
        result['pre_err_v_top_px'] = v_t_pred - float(ann_row['v_top'])
    if status in ('full', 'base_only'):
        u_b_pred, v_b_pred = _project_plain(base_w)
        result['pre_err_u_base_px'] = u_b_pred - float(ann_row['u_base'])
        result['pre_err_v_base_px'] = v_b_pred - float(ann_row['v_base'])

    max_rad = np.deg2rad(float(max_angle_deg))

    if status == 'full':
        x0 = np.zeros(3, dtype=np.float64)
        try:
            sol = least_squares(
                residuals_full, x0,
                args=(top_w, base_w,
                      float(ann_row['u_top']), float(ann_row['v_top']),
                      float(ann_row['u_base']), float(ann_row['v_base']),
                      cam_xyz, R_wc_original, fx, fy, cx, cy, flip_u, flip_v, W, H),
                bounds=(-max_rad * 1.5, max_rad * 1.5),
                method='trf', xtol=1e-10, ftol=1e-10,
            )
            rvec = sol.x.astype(np.float64)
            result['success'] = bool(sol.success)
            result['solver_dof'] = 3
        except Exception as e:
            result['note'] = f'solver failed: {e}'
            return result
    elif status in ('top_only', 'base_only'):
        x0 = np.zeros(2, dtype=np.float64)
        if status == 'top_only':
            pw = top_w; uu = float(ann_row['u_top']); vv = float(ann_row['v_top'])
        else:
            pw = base_w; uu = float(ann_row['u_base']); vv = float(ann_row['v_base'])
        try:
            sol = least_squares(
                residuals_single, x0,
                args=(pw, uu, vv, cam_xyz, R_wc_original,
                      fx, fy, cx, cy, flip_u, flip_v, W, H),
                bounds=(-max_rad * 1.5, max_rad * 1.5),
                method='trf', xtol=1e-10, ftol=1e-10,
            )
            rvec = np.array([sol.x[0], sol.x[1], 0.0], dtype=np.float64)
            result['success'] = bool(sol.success)
            result['solver_dof'] = 2
        except Exception as e:
            result['note'] = f'solver failed: {e}'
            return result
    else:
        result['note'] = f'unknown annotation_status: {status}'
        return result

    rvec_clipped, clipped = clip_rotation_angle(rvec, max_angle_deg)
    result['clipped_by_max_angle'] = bool(clipped)
    R_corr = Rotation.from_rotvec(rvec_clipped).as_matrix()
    result['R_corr'] = R_corr.tolist()


    result['rx_deg'] = float(np.rad2deg(rvec_clipped[0]))
    result['ry_deg'] = float(np.rad2deg(rvec_clipped[1]))
    result['rz_deg'] = float(np.rad2deg(rvec_clipped[2]))
    result['rotation_total_deg'] = float(np.rad2deg(np.linalg.norm(rvec_clipped)))


    def _project_corr(pw):
        u, v, _ = project_with_correction(pw[None, :], cam_xyz, R_wc_original, R_corr,
                                          fx, fy, cx, cy, flip_u, flip_v, W, H)
        return float(u[0]), float(v[0])
    if status in ('full', 'top_only'):
        u_t_post, v_t_post = _project_corr(top_w)
        result['post_err_u_top_px'] = u_t_post - float(ann_row['u_top'])
        result['post_err_v_top_px'] = v_t_post - float(ann_row['v_top'])
    if status in ('full', 'base_only'):
        u_b_post, v_b_post = _project_corr(base_w)
        result['post_err_u_base_px'] = u_b_post - float(ann_row['u_base'])
        result['post_err_v_base_px'] = v_b_post - float(ann_row['v_base'])

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='主工程配置文件')
    ap.add_argument('--apply', action='store_true',
                    help='真正把修正矩阵写回 scene_meta.json（默认 dry-run）')
    ap.add_argument('--max-angle-deg', type=float, default=None,
                    help='R_corr 的旋转向量总角度上限（度），默认读取 target_calibration.max_angle_deg 或 10')
    ap.add_argument('--restore-original', action='store_true',
                    help='不解算，直接把每景 scene_meta.json 从 *.bak 里还原回未修正的状态')
    ap.add_argument('--skip-quality-gate', action='store_true',
                    help='跳过配置中的 R_corr 总角度门禁，主要用于人工调试')
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_cfg = cfg.get('target_calibration', {})
    if not target_cfg:
        raise SystemExit('主配置文件里没有 target_calibration 段')

    max_angle_deg = float(args.max_angle_deg if args.max_angle_deg is not None else target_cfg.get('max_angle_deg', 10.0))
    quality_gate_enabled, quality_thresholds = get_quality_gate_config(target_cfg)
    if args.skip_quality_gate:
        quality_gate_enabled = False

    scene_root = Path(cfg['outputs']['scene_db_dir']) / 'scenes'
    calibration_dir = Path(cfg['outputs'].get(
        'target_calibration_dir',
        Path(cfg['outputs']['scene_db_dir']).parent / '05_target_calibration',
    ))
    ann_csv = Path(target_cfg.get('annotation_csv', calibration_dir / 'target_annotations.csv'))
    log_csv = Path(target_cfg.get('correction_log_csv', calibration_dir / 'per_scene_correction.csv'))


    if args.restore_original:
        n_restored = 0
        for scene_dir in scene_root.iterdir():
            if not scene_dir.is_dir():
                continue
            meta_path = scene_dir / 'scene_meta.json'
            bak = meta_path.with_suffix(meta_path.suffix + '.bak')
            if bak.exists():
                shutil.copy2(bak, meta_path)
                n_restored += 1
        print(f'已从 .bak 还原 {n_restored} 个 scene_meta.json')
        return


    if not ann_csv.exists():
        raise SystemExit(f'找不到标注 CSV：{ann_csv}，请先跑 05。')
    ann = pd.read_csv(ann_csv, encoding='utf-8-sig')
    if ann.empty:
        raise SystemExit('标注 CSV 是空的，没有可解算的场景。')

    results = []
    for _, row in ann.iterrows():
        sid = str(row['scene_id'])
        meta_path = scene_root / sid / 'scene_meta.json'
        if not meta_path.exists():
            print(f'[skip] {sid}: 找不到 scene_meta.json')
            continue
        scene_meta = load_scene_meta(meta_path)
        res = solve_one(row, scene_meta, max_angle_deg=max_angle_deg)
        results.append(res)


    ensure_dir(log_csv.parent)
    rows_for_csv = []
    mats = {}
    for r in results:
        row = {k: v for k, v in r.items() if k != 'R_corr'}
        rows_for_csv.append(row)
        mats[r['scene_id']] = r['R_corr']
    df = pd.DataFrame(rows_for_csv)
    df.to_csv(log_csv, index=False, encoding='utf-8-sig')
    save_json({'correction_matrices': mats},
              log_csv.with_name(log_csv.stem + '_matrices.json'))


    solved = df[df['annotation_status'].isin(SOLVED_STATUSES)] if 'annotation_status' in df.columns else pd.DataFrame()
    if len(solved) > 0:
        print('\n=== 解算结果摘要 ===')
        print(f'  参与解算的场景数 : {len(solved)}')
        if len(solved) > 0:
            angles = solved['rotation_total_deg'].astype(float)
            print(f'  R_corr 总角度（deg）: '
                  f'min={float(angles.min()):.3f}  '
                  f'median={float(angles.median()):.3f}  '
                  f'mean={float(angles.mean()):.3f}  '
                  f'max={float(angles.max()):.3f}')

            pre = solved[['pre_err_u_top_px', 'pre_err_v_top_px',
                          'pre_err_u_base_px', 'pre_err_v_base_px']].abs().stack().dropna()
            post = solved[['post_err_u_top_px', 'post_err_v_top_px',
                           'post_err_u_base_px', 'post_err_v_base_px']].abs().stack().dropna()
            if len(pre) > 0 and len(post) > 0:
                print(f'  修正前像素绝对残差: mean={float(pre.mean()):.2f} '
                      f'median={float(pre.median()):.2f} max={float(pre.max()):.2f}')
                print(f'  修正后像素绝对残差: mean={float(post.mean()):.2f} '
                      f'median={float(post.median()):.2f} max={float(post.max()):.2f}')

            n_clip = int(solved['clipped_by_max_angle'].sum())
            if n_clip > 0:
                print(f'  ⚠ 有 {n_clip} 景超过 --max-angle-deg={max_angle_deg}° 被裁剪，'
                      f'这些景建议人工复查标注是否点错')
    print(f'\n  解算汇总 CSV: {log_csv}')
    print(f'  修正矩阵 JSON: {log_csv.with_name(log_csv.stem + "_matrices.json")}')

    if len(solved) > 0:
        print_top_error_scenes(solved, top_n=5)

    gate_passed = True
    gate_messages: list[str] = []
    gate_stats: dict[str, float] = {'min': np.nan, 'median': np.nan, 'mean': np.nan, 'max': np.nan}
    if quality_gate_enabled:
        gate_passed, gate_messages, gate_stats = evaluate_quality_gate(df, quality_thresholds)
        print('\n=== R_corr 总角度质量门禁 ===')
        print('  阈值（deg）: '
              f'min<={quality_thresholds["min"]:.3f}  '
              f'median<={quality_thresholds["median"]:.3f}  '
              f'mean<={quality_thresholds["mean"]:.3f}')
        if np.isfinite(gate_stats['min']):
            print('  当前（deg）: '
                  f'min={gate_stats["min"]:.3f}  '
                  f'median={gate_stats["median"]:.3f}  '
                  f'mean={gate_stats["mean"]:.3f}  '
                  f'max={gate_stats["max"]:.3f}')
        if not gate_passed:
            print('\n  质量门禁失败，未写回 scene_meta.json，流程已停止。')
            for msg in gate_messages:
                print(f'  - {msg}')
            print(f'  请回到 05 检查/修正标注，然后重新运行 run_all 或 06。')
            raise SystemExit(2)
        print('  质量门禁通过。')
    else:
        print('\n  已跳过 R_corr 总角度质量门禁。')

    if args.apply:
        applied = apply_corrections_to_scene_meta(results, scene_root)
        print(f'\n  ✓ 已把修正矩阵写回 {applied} 个 scene_meta.json（原始矩阵存放在 *.bak 文件和'
              ' world_to_camera_R_original 字段）')
    else:
        print('\n  这是 dry-run。确认汇报 OK 后，加 --apply 再跑一次正式写回。')
        print('  若结果不理想可以用 --restore-original 从 .bak 一键回滚。')


if __name__ == '__main__':
    main()
