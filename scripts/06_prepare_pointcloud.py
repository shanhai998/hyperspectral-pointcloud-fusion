
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hyperspectral_pointcloud_fusion.common import load_config, ensure_dir, write_table, save_json, load_json
from hyperspectral_pointcloud_fusion.plyio import read_ply_xyz


def _normalize_vectors(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=1)
    ok = np.isfinite(n) & (n > 1e-12)
    out = np.zeros(v.shape, dtype=np.float32)
    out[ok] = (v[ok] / n[ok, None]).astype(np.float32)
    return out, ok


def _estimate_normals_pca(
    xyz: np.ndarray,
    k_neighbors: int,
    chunk_size: int,
    leafsize: int,
    workers: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    n_points = int(xyz.shape[0])
    k = max(4, min(int(k_neighbors), n_points))
    chunk_size = max(1, int(chunk_size))
    tree = cKDTree(xyz, leafsize=max(1, int(leafsize)))
    normals = np.zeros((n_points, 3), dtype=np.float32)
    confidence = np.zeros(n_points, dtype=np.float32)

    for i0 in range(0, n_points, chunk_size):
        i1 = min(n_points, i0 + chunk_size)
        _, nn_idx = tree.query(xyz[i0:i1], k=k, workers=max(1, int(workers)))
        if nn_idx.ndim == 1:
            nn_idx = nn_idx[:, None]

        neigh = xyz[nn_idx]
        centered = neigh - np.mean(neigh, axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', centered, centered) / max(1, k - 1)
        vals, vecs = np.linalg.eigh(cov)
        normal = vecs[:, :, 0]
        normal, ok = _normalize_vectors(normal)
        normals[i0:i1] = normal

        vals = np.maximum(vals, 0.0)
        denom = np.sum(vals, axis=1)
        conf = np.zeros(vals.shape[0], dtype=np.float64)
        good = np.isfinite(denom) & (denom > 1e-12)
        conf[good] = 1.0 - vals[good, 0] / denom[good]
        conf[~ok] = 0.0
        confidence[i0:i1] = np.clip(conf, 0.0, 1.0).astype(np.float32)

    return normals, confidence


def _load_or_estimate_normals(p: dict, xyz: np.ndarray, proc: dict) -> tuple[np.ndarray, np.ndarray, str]:
    if all(k in p for k in ['nx', 'ny', 'nz']):
        raw = np.column_stack([
            np.asarray(p['nx'], dtype=np.float64),
            np.asarray(p['ny'], dtype=np.float64),
            np.asarray(p['nz'], dtype=np.float64),
        ])
        normals, ok = _normalize_vectors(raw)
        if int(np.count_nonzero(ok)) >= max(1, int(0.95 * xyz.shape[0])):
            return normals, ok.astype(np.float32), 'ply'

    normals, confidence = _estimate_normals_pca(
        xyz=xyz,
        k_neighbors=int(proc.get('normal_estimation_k_neighbors', 48)),
        chunk_size=int(proc.get('normal_estimation_chunk_size', 20000)),
        leafsize=int(proc.get('kd_leafsize', 32)),
        workers=int(proc.get('normal_estimation_workers', proc.get('task_workers', 20))),
    )
    return normals, confidence, 'pca'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['outputs']['pointcloud_dir'])
    solution = load_json(Path(cfg['outputs']['crs_dir']) / 'hemi_solution.json')
    center_xyz = np.asarray(solution['hemisphere_center_xyz'], dtype=np.float64)
    proc = dict(cfg.get('processing', {}))

    p = read_ply_xyz(cfg['paths']['pointcloud_ply'])
    xyz = np.column_stack([
        np.asarray(p['x'], dtype=np.float64),
        np.asarray(p['y'], dtype=np.float64),
        np.asarray(p['z'], dtype=np.float64),
    ])
    xyz_local = xyz - center_xyz[None, :]
    np.save(out_dir / 'point_xyz.npy', xyz)

    normals, normal_confidence, normal_source = _load_or_estimate_normals(p, xyz, proc)
    surface_verticality = (1.0 - np.abs(np.asarray(normals[:, 2], dtype=np.float64))).clip(0.0, 1.0).astype(np.float32)
    np.save(out_dir / 'point_normals.npy', normals)
    np.save(out_dir / 'point_surface_verticality.npy', surface_verticality)
    np.save(out_dir / 'point_normal_confidence.npy', normal_confidence.astype(np.float32))

    attrs = {
        'point_id': np.arange(xyz.shape[0], dtype=np.int64),
        'x': xyz[:, 0],
        'y': xyz[:, 1],
        'z': xyz[:, 2],
        'x_local': xyz_local[:, 0],
        'y_local': xyz_local[:, 1],
        'z_local': xyz_local[:, 2],
        'normal_x': normals[:, 0],
        'normal_y': normals[:, 1],
        'normal_z': normals[:, 2],
        'surface_verticality': surface_verticality,
        'normal_confidence': normal_confidence,
    }
    if all(k in p for k in ['red', 'green', 'blue']):
        rgb = np.column_stack([
            np.asarray(p['red'], dtype=np.uint8),
            np.asarray(p['green'], dtype=np.uint8),
            np.asarray(p['blue'], dtype=np.uint8),
        ])
        attrs['red'] = rgb[:, 0]
        attrs['green'] = rgb[:, 1]
        attrs['blue'] = rgb[:, 2]

    point_index = pd.DataFrame(attrs)
    write_table(point_index, out_dir / 'point_index.csv', index=False)

    save_json({
        'point_count': int(xyz.shape[0]),
        'hemisphere_center_xyz': center_xyz.tolist(),
        'has_rgb': bool(all(k in p for k in ['red', 'green', 'blue'])),
        'normal_source': normal_source,
        'mean_surface_verticality': float(np.nanmean(surface_verticality)),
        'mean_normal_confidence': float(np.nanmean(normal_confidence)),
    }, out_dir / 'pointcloud_summary.json')
    print(f'points: {xyz.shape[0]}')
    print(f'normal_source={normal_source}, mean_surface_verticality={float(np.nanmean(surface_verticality)):.4f}')


if __name__ == '__main__':
    main()
