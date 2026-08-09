
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


def _read_yaml_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f'config must be a YAML mapping: {config_path}')
    return cfg


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(current, value)
        else:
            merged[key] = value
    return merged


def _safe_windows_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', str(value)).strip(' .')
    if not name:
        raise ValueError('pointcloud filename must contain a usable stem')
    return name


def _resolve_runtime_output_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    project = cfg.setdefault('project', {})
    previous_root_value = project.get('output_root')
    previous_root = Path(str(previous_root_value)).resolve() if previous_root_value else None
    paths = cfg.get('paths', {})
    pointcloud = Path(str(paths.get('pointcloud_ply', '')))
    pointcloud_stem = _safe_windows_name(pointcloud.stem)

    band_mode = str((cfg.get('bands', {}) or {}).get('mode', 'subset')).strip().lower()
    if band_mode not in {'subset', 'all'}:
        raise ValueError(f"bands.mode must be 'subset' or 'all', got: {band_mode}")
    all_bands = band_mode == 'all'

    output_base = project.get('output_base')
    if output_base:
        base = Path(str(output_base)).resolve()
        run_name = f'{pointcloud_stem}_allbands' if all_bands else pointcloud_stem
        root = base / run_name
        analysis = base / ('result_allbands' if all_bands else 'result')
        figures = base / ('picture_allbands' if all_bands else 'picture')
    else:
        template = str(project.get('output_root', ''))
        if not template:
            raise ValueError('project.output_base or project.output_root is required')
        root = Path(template.format(
            pointcloud_stem=pointcloud_stem,
            band_suffix='_allbands' if all_bands else '',
        ))
        base = root.parent
        analysis = base / ('result_allbands' if all_bands else 'result')
        figures = base / ('picture_allbands' if all_bands else 'picture')

    root = root.resolve()
    project['output_root'] = str(root)
    project['analysis_root'] = str(analysis.resolve())
    project['figure_root'] = str(figures.resolve())
    project['pointcloud_stem'] = pointcloud_stem
    project['band_mode'] = band_mode
    project['all_bands'] = all_bands

    resolved_outputs: Dict[str, Any] = {}
    for key, value in (cfg.get('outputs', {}) or {}).items():
        path = Path(str(value))
        if path.is_absolute() and previous_root is not None:
            try:
                path = root / path.resolve().relative_to(previous_root)
            except ValueError:
                pass
        elif not path.is_absolute():
            path = root / path
        resolved_outputs[key] = str(path)
    cfg['outputs'] = resolved_outputs

    target_calibration = cfg.get('target_calibration', {}) or {}
    for key in ('annotation_csv', 'correction_log_csv'):
        value = target_calibration.get(key)
        if value:
            path = Path(str(value))
            if path.is_absolute() and previous_root is not None:
                try:
                    path = root / path.resolve().relative_to(previous_root)
                except ValueError:
                    pass
                target_calibration[key] = str(path)
            elif not path.is_absolute():
                target_calibration[key] = str(root / path)
    cfg['target_calibration'] = target_calibration
    return cfg


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path).resolve()
    cfg = _read_yaml_config(config_file)

    if 'base_config' in cfg:
        base_path = Path(str(cfg['base_config']))
        if not base_path.is_absolute():
            base_path = (config_file.parent / base_path).resolve()

        base_cfg = load_config(str(base_path))
        base_cfg = {k: v for k, v in base_cfg.items() if not str(k).startswith('_')}

        overrides = cfg.get('overrides', {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f'overrides must be a YAML mapping: {config_file}')

        direct_values = {k: v for k, v in cfg.items() if k not in {'base_config', 'overrides'}}
        cfg = _deep_merge_dict(base_cfg, overrides)
        cfg = _deep_merge_dict(cfg, direct_values)
        cfg['_base_config_path'] = str(base_path)

    cfg = _resolve_runtime_output_paths(cfg)
    cfg['_config_path'] = str(config_file)
    cfg['_project_dir'] = str(config_file.parent.parent)
    return cfg


def output_root(cfg: Dict[str, Any]) -> Path:
    return Path(str(cfg['project']['output_root'])).resolve()


def analysis_root(cfg: Dict[str, Any]) -> Path:
    return Path(str(cfg['project']['analysis_root'])).resolve()


def figure_root(cfg: Dict[str, Any]) -> Path:
    return Path(str(cfg['project']['figure_root'])).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.parent


def save_json(data: Any, path: str | Path) -> None:
    ensure_parent(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'表格不存在: {p}')
    if p.suffix.lower() in ['.csv', '.txt']:
        return pd.read_csv(p)
    if p.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(p, engine='openpyxl')
    if p.suffix.lower() in ['.pkl', '.pickle']:
        return pd.read_pickle(p)
    raise ValueError(f'不支持的表格格式: {p}')


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    p = Path(path)
    ensure_parent(p)
    if p.suffix.lower() in ['.csv', '.txt']:
        df.to_csv(p, index=index, encoding='utf-8-sig')
    elif p.suffix.lower() in ['.pkl', '.pickle']:
        df.to_pickle(p)
    else:
        raise ValueError(f'不支持的输出表格格式: {p}')


def append_table(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    header: bool | None = None,
) -> None:
    p = Path(path)
    ensure_parent(p)
    if p.suffix.lower() not in ['.csv', '.txt']:
        raise ValueError(f'append_table 仅支持 csv/txt: {p}')
    write_header = (not p.exists()) if header is None else bool(header)
    mode = 'w' if write_header else 'a'
    encoding = 'utf-8-sig' if write_header else 'utf-8'
    df.to_csv(p, mode=mode, header=write_header, index=index, encoding=encoding)


def normalize_text(s: object) -> str:
    if s is None:
        return ''
    return str(s).strip().lower().replace(' ', '').replace('(', '').replace(')', '')


def extract_scene_key_from_ref_name(name: str) -> str:
    base = Path(name).stem
    m = re.match(r'^(.*?)(_ref)$', base, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return base


def extract_scene_key_from_excel_label(label: object) -> str:
    s = os.path.basename(str(label).strip())
    s = re.sub(r'\.tif{1,2}$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'_pan$', '', s, flags=re.IGNORECASE)
    return s


def list_hdr_files(ref_root: str | Path) -> List[Path]:
    ref_root = Path(ref_root)
    files = list(ref_root.glob('*_REF.hdr')) + list(ref_root.glob('*_ref.hdr'))
    if not files:
        files = list(ref_root.glob('*.hdr')) + list(ref_root.glob('*.HDR'))
    unique: dict[str, Path] = {}
    for fp in files:
        key = str(fp.resolve()).lower()
        if key not in unique:
            unique[key] = fp.resolve()
    return sorted(unique.values(), key=lambda x: x.name.lower())


def derive_data_path_from_hdr(hdr_path: str | Path) -> Path:
    hdr_path = Path(hdr_path)
    stem = hdr_path.with_suffix('')
    for suffix in ['.cue', '.CUE', '.bin', '.BIN', '.dat', '.DAT']:
        p = Path(str(stem) + suffix)
        if p.exists():
            return p
    raise FileNotFoundError(f'未找到与 hdr 配套的数据文件: {hdr_path}')


def get_scene_workers(cfg: Dict[str, Any]) -> int:
    return int(cfg.get('processing', {}).get('scene_workers', max(1, os.cpu_count() or 1)))


def get_task_workers(cfg: Dict[str, Any]) -> int:
    return int(cfg.get('processing', {}).get('task_workers', max(1, os.cpu_count() or 1)))


def get_io_workers(cfg: Dict[str, Any]) -> int:
    return int(cfg.get('processing', {}).get('io_workers', 4))


def is_quiet(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get('processing', {}).get('quiet_logs', False))


def project_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg['_project_dir'])


def copy_self_to_output(cfg: Dict[str, Any], dst_dir: str | Path) -> None:
    script_path = Path(sys.argv[0]).resolve()
    dst = ensure_dir(dst_dir) / script_path.name
    try:
        shutil.copy2(script_path, dst)
    except Exception:
        pass


def chunk_ranges(n: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    chunk_size = max(1, int(chunk_size))
    for i in range(0, int(n), chunk_size):
        yield i, min(int(n), i + chunk_size)


def percentile_stretch(
    values: np.ndarray,
    valid_mask: np.ndarray,
    q_low: float = 0.0005,
    q_high: float = 0.9995,
) -> tuple[np.ndarray, float, float]:
    out = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(valid_mask):
        return out, 0.0, 1.0
    vals = np.asarray(values, dtype=np.float64)[valid_mask]
    lo = float(np.quantile(vals, q_low))
    hi = float(np.quantile(vals, q_high))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi <= lo:
            hi = lo + 1e-6
    tmp = np.zeros(values.shape, dtype=np.float64)
    tmp[valid_mask] = np.clip(
        (np.asarray(values, dtype=np.float64)[valid_mask] - lo) / (hi - lo),
        0.0,
        1.0,
    )
    out[valid_mask] = np.round(tmp[valid_mask] * 255.0).astype(np.uint8)
    return out, lo, hi


def resolve_center_window(
    image_width: int,
    image_height: int,
    window_width_px: int | None = None,
    window_height_px: int | None = None,
) -> Dict[str, int]:
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError(f'invalid image size: width={width}, height={height}')

    window_width = width if window_width_px is None else int(window_width_px)
    window_height = height if window_height_px is None else int(window_height_px)
    if window_width <= 0 or window_height <= 0:
        raise ValueError(
            f'center window must be positive, got width={window_width}, height={window_height}'
        )

    window_width = min(window_width, width)
    window_height = min(window_height, height)
    x_min = (width - window_width) // 2
    y_min = (height - window_height) // 2
    x_max = x_min + window_width - 1
    y_max = y_min + window_height - 1
    return {
        'width': width,
        'height': height,
        'window_width': window_width,
        'window_height': window_height,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
    }


def get_band_selection(
    cfg: Dict[str, Any],
    total_bands: int,
    wavelengths: Sequence[float] | None,
) -> tuple[list[int], dict[str, Any]]:
    bands_cfg = cfg.get('bands', {})
    mode = str(bands_cfg.get('mode', 'subset')).lower().strip()
    if mode == 'all':
        indices = list(range(total_bands))
    else:
        subset_indices = bands_cfg.get('subset_indices', []) or []
        subset_wavelengths = bands_cfg.get('subset_wavelengths_nm', []) or []
        indices: List[int] = []
        if subset_indices:
            for band_no in subset_indices:
                idx = int(band_no) - 1
                if idx < 0 or idx >= total_bands:
                    raise ValueError(f'band number out of range: {band_no}, valid range=1..{total_bands}')
                indices.append(idx)
        elif subset_wavelengths:
            if wavelengths is None or len(wavelengths) != total_bands:
                raise ValueError('subset_wavelengths_nm is configured, but hdr wavelength metadata is incomplete')
            wav = np.asarray(wavelengths, dtype=np.float64)
            for target in subset_wavelengths:
                indices.append(int(np.argmin(np.abs(wav - float(target)))))
        else:
            raise ValueError('when bands.mode=subset, subset_indices or subset_wavelengths_nm must be configured')
    indices = sorted(dict.fromkeys(indices))
    return indices, {
        'count': len(indices),
        'indices_zero_based': indices,
        'band_numbers_one_based': [int(i) + 1 for i in indices],
    }
