
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import ensure_dir, load_config, list_hdr_files, derive_data_path_from_hdr, write_table, save_json, extract_scene_key_from_ref_name, get_scene_workers, is_quiet
from hyperspectral_pointcloud_fusion.envi import parse_envi_hdr, wavelength_list_from_hdr


def process_one_hdr(hdr_path_str: str, wav_dir_str: str):
    hdr_path = Path(hdr_path_str)
    wav_dir = Path(wav_dir_str)
    try:
        hdr = parse_envi_hdr(str(hdr_path))
        data_path = derive_data_path_from_hdr(hdr_path)
        scene_id = extract_scene_key_from_ref_name(hdr_path.name)
        wavelengths = wavelength_list_from_hdr(hdr)
        wav_path = wav_dir / f'{scene_id}_wavelengths.npy'
        np.save(wav_path, np.asarray(wavelengths, dtype=np.float64))
        row = {
            'scene_id': scene_id,
            'scene_key': scene_id.lower(),
            'hdr_path': str(hdr_path),
            'data_path': str(data_path),
            'samples': int(hdr.get('samples')),
            'lines': int(hdr.get('lines')),
            'bands': int(hdr.get('bands')),
            'data_type': int(hdr.get('data type')),
            'interleave': str(hdr.get('interleave')).strip().lower(),
            'byte_order': int(hdr.get('byte order', 0)),
            'header_offset': int(hdr.get('header offset', 0)),
            'reflectance_scale_factor': float(hdr.get('reflectance scale factor', 1.0)),
            'wavelength_count': int(len(wavelengths)),
            'wavelength_min_nm': float(wavelengths[0]) if wavelengths else np.nan,
            'wavelength_max_nm': float(wavelengths[-1]) if wavelengths else np.nan,
            'wavelengths_path': str(wav_path),
        }
        return row, None
    except Exception as e:
        return None, {'hdr_path': str(hdr_path), 'error': str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['outputs']['inventory_dir'])
    wav_dir = ensure_dir(out_dir / 'wavelengths')

    hdr_files = list_hdr_files(cfg['paths']['ref_root'])
    print(f'发现 hdr 文件数量: {len(hdr_files)}')
    workers = max(1, min(get_scene_workers(cfg), len(hdr_files) if hdr_files else 1))
    quiet = is_quiet(cfg)
    if not quiet:
        print(f'scene_workers={workers}')
    rows = []
    errors = []
    if workers <= 1:
        for hdr_path in hdr_files:
            row, err = process_one_hdr(str(hdr_path), str(wav_dir))
            if row is not None:
                rows.append(row)
            if err is not None:
                errors.append(err)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_one_hdr, str(h), str(wav_dir)) for h in hdr_files]
            for fu in as_completed(futures):
                row, err = fu.result()
                if row is not None:
                    rows.append(row)
                if err is not None:
                    errors.append(err)

    manifest = pd.DataFrame(rows).sort_values('scene_id').reset_index(drop=True) if rows else pd.DataFrame()
    write_table(manifest, out_dir / 'scene_manifest.csv', index=False)
    write_table(pd.DataFrame(errors), out_dir / 'scene_errors.csv', index=False)
    save_json({'scene_count': int(len(manifest)), 'error_count': int(len(errors))}, out_dir / 'inventory_summary.json')
    print(f'有效场景: {len(manifest)}')
    print(f'错误场景: {len(errors)}')
    print(f'输出: {out_dir / "scene_manifest.csv"}')


if __name__ == '__main__':
    main()
