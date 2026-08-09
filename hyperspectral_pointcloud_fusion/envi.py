
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


def parse_envi_hdr(hdr_path: str) -> Dict[str, object]:
    import re

    with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    text = text.replace('\r\n', '\n')
    pattern = re.compile(r'^\s*([^=]+?)\s*=\s*(\{.*?\}|[^\n]+)\s*$', re.M | re.S)
    items: Dict[str, object] = {}
    for m in pattern.finditer(text):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if val.startswith('{') and val.endswith('}'):
            inner = val[1:-1].strip()
            parts = [p.strip() for p in inner.replace('\n', ' ').split(',') if p.strip()]
            parsed = []
            for p in parts:
                try:
                    if '.' in p or 'e' in p.lower():
                        parsed.append(float(p))
                    else:
                        parsed.append(int(p))
                except Exception:
                    parsed.append(p)
            items[key] = parsed
        else:
            try:
                if '.' in val or 'e' in val.lower():
                    items[key] = float(val)
                else:
                    items[key] = int(val)
            except Exception:
                items[key] = val
    return items


def wavelength_list_from_hdr(hdr: Dict[str, object]) -> list[float]:
    wav = hdr.get('wavelength', [])
    if isinstance(wav, list):
        out = []
        for v in wav:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out
    return []


def envi_dtype_to_numpy(data_type: int) -> np.dtype:
    table = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    if int(data_type) not in table:
        raise ValueError(f'暂不支持 ENVI data type = {data_type}')
    return np.dtype(table[int(data_type)])


def get_file_element_count(data_path: str, data_type: int, header_offset: int = 0) -> int:
    dtype = envi_dtype_to_numpy(data_type)
    file_size = Path(data_path).stat().st_size
    payload = file_size - int(header_offset)
    if payload < 0:
        raise ValueError('header offset 大于文件大小')
    if payload % dtype.itemsize != 0:
        raise ValueError('数据文件大小不是 dtype 的整数倍，请检查 hdr 与数据文件')
    return payload // dtype.itemsize


def read_envi_bsq_memmap(data_path: str, samples: int, lines: int, bands: int, data_type: int, byte_order: int = 0, header_offset: int = 0) -> np.memmap:
    base_dtype = envi_dtype_to_numpy(data_type)
    if int(byte_order) == 0:
        dtype = base_dtype.newbyteorder('<')
    elif int(byte_order) == 1:
        dtype = base_dtype.newbyteorder('>')
    else:
        raise ValueError(f'不支持 byte order = {byte_order}')
    actual = get_file_element_count(data_path, data_type=data_type, header_offset=header_offset)
    expected = int(samples) * int(lines) * int(bands)
    if actual != expected:
        raise ValueError(f'数据大小不匹配: 实际元素数 {actual}, 期望 {expected}')
    return np.memmap(data_path, dtype=dtype, mode='r', offset=header_offset, shape=(int(bands), int(lines), int(samples)))


def bilinear_sample_selected_bands_bsq(src_cube_bsq: np.ndarray, x: np.ndarray, y: np.ndarray, band_indices: Sequence[int], scale_factor: float = 1.0) -> np.ndarray:
    band_indices = np.asarray(list(band_indices), dtype=np.int64)
    _, H, W = src_cube_bsq.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    wx = (x - x0).astype(np.float64)
    wy = (y - y0).astype(np.float64)

    sub = src_cube_bsq[band_indices, :, :]
    p00 = sub[:, y0, x0].astype(np.float64)
    p10 = sub[:, y0, x1].astype(np.float64)
    p01 = sub[:, y1, x0].astype(np.float64)
    p11 = sub[:, y1, x1].astype(np.float64)

    p00 = np.moveaxis(p00, 0, -1)
    p10 = np.moveaxis(p10, 0, -1)
    p01 = np.moveaxis(p01, 0, -1)
    p11 = np.moveaxis(p11, 0, -1)

    val0 = (1.0 - wx)[..., None] * p00 + wx[..., None] * p10
    val1 = (1.0 - wx)[..., None] * p01 + wx[..., None] * p11
    out = (1.0 - wy)[..., None] * val0 + wy[..., None] * val1
    if scale_factor not in (0, 1, 1.0):
        out /= float(scale_factor)
    return out

