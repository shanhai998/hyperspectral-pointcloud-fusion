from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BAND_NUMBERS = np.asarray([26, 51, 76, 110, 164, 165], dtype=np.int16)
BAND_NAMES = ["band_26", "band_51", "band_76", "band_110", "band_164", "NDVI"]
EPS = 1.0e-12

OBS_COLUMNS = [
    "observation_id", "point_id", "scene_id", "scene_code", "u", "v", "is_clear",
    "brf_quality_level", "view_weight_raw", "view_zenith_deg", "view_azimuth_deg",
    "solar_zenith_deg", "solar_azimuth_deg", "relative_azimuth_deg",
    "local_view_angle_deg", "local_solar_incidence_deg", "offaxis_deg",
    "border_dist_px", "local_empty_cone_deg", "surface_view_cos",
    "surface_verticality", "normal_confidence", "normal_z", "range_m", "x", "y", "z",
]


@dataclass(frozen=True)
class V3Paths:
    result_root: Path
    observation_table: Path
    spectra: Path
    point_index: Path

    @classmethod
    def from_root(cls, result_root: str | Path) -> "V3Paths":
        root = Path(result_root)
        observation_table = root / "brf" / "point_directional_brf_observations.parquet"
        spectra = root / "brf" / "directional_brf_spectra.npy"
        point_index = root / "source" / "pointcloud" / "point_index.csv"
        return cls(
            result_root=root,
            observation_table=observation_table,
            spectra=spectra,
            point_index=point_index,
        )


def representative_spectra_columns(paths: V3Paths, spectra_column_count: int) -> np.ndarray:


    metadata_candidates = [
        paths.result_root / "brf" / "directional_brf_band_metadata.csv",
    ]
    metadata_path = next((path for path in metadata_candidates if path.exists()), None)
    if metadata_path is None:
        if int(spectra_column_count) == len(BAND_NUMBERS):
            return np.arange(len(BAND_NUMBERS), dtype=np.int64)
        raise FileNotFoundError(
            "Cannot map representative bands in an all-band matrix because "
            f"directional_brf_band_metadata.csv is missing under {paths.result_root}"
        )

    metadata = pd.read_csv(metadata_path)
    if "band_number" not in metadata.columns:
        raise ValueError(f"band_number is missing from {metadata_path}")
    if len(metadata) != int(spectra_column_count):
        raise ValueError(
            f"Band metadata rows ({len(metadata)}) do not match spectra columns "
            f"({spectra_column_count}) in {paths.spectra}"
        )
    band_values = pd.to_numeric(metadata["band_number"], errors="raise").astype(int)
    lookup: dict[int, int] = {}
    for local_index, band_number in enumerate(band_values.tolist()):
        if band_number in lookup:
            raise ValueError(f"Duplicate band_number={band_number} in {metadata_path}")
        lookup[int(band_number)] = int(local_index)
    missing = [int(band) for band in BAND_NUMBERS if int(band) not in lookup]
    if missing:
        raise ValueError(f"Representative bands are missing from {metadata_path}: {missing}")
    return np.asarray([lookup[int(band)] for band in BAND_NUMBERS], dtype=np.int64)


def read_representative_response(
    spectra: np.ndarray,
    observation_ids: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:


    ids = np.asarray(observation_ids, dtype=np.int64)
    selected = np.asarray(columns, dtype=np.int64)
    response = np.empty((len(ids), len(selected)), dtype=np.float32)
    for output_column, source_column in enumerate(selected.tolist()):
        response[:, output_column] = np.asarray(
            spectra[ids, int(source_column)], dtype=np.float32
        )
    return response


def scene_suffix(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else 0


def stable_point_sample(point_ids: np.ndarray, modulus: int, remainder: int = 0) -> np.ndarray:

    if modulus <= 1:
        return np.ones(len(point_ids), dtype=bool)
    x = np.asarray(point_ids, dtype=np.uint64)
    x ^= x >> np.uint64(16)
    x *= np.uint64(0x7FEB352D)
    x ^= x >> np.uint64(15)
    x *= np.uint64(0x846CA68B)
    x ^= x >> np.uint64(16)
    return (x % np.uint64(modulus)) == np.uint64(remainder)


def load_observations(
    paths: V3Paths,
    *,
    point_sample_modulus: int = 1,
    point_sample_remainder: int = 0,
    clear_only: bool = True,
    positive_reflectance_only: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    if not paths.observation_table.exists():
        raise FileNotFoundError(paths.observation_table)
    if not paths.spectra.exists():
        raise FileNotFoundError(paths.spectra)
    filters = [("is_clear", "==", True)] if clear_only else None
    obs = pd.read_parquet(paths.observation_table, columns=OBS_COLUMNS, filters=filters)
    if clear_only:
        obs = obs.loc[obs["is_clear"].fillna(False).astype(bool)].copy()
    sample_mask = stable_point_sample(
        pd.to_numeric(obs["point_id"], errors="raise").to_numpy(np.int64),
        point_sample_modulus,
        point_sample_remainder,
    )
    obs = obs.loc[sample_mask].reset_index(drop=True)
    finite_geometry = np.isfinite(
        obs[[
            "u", "v", "view_zenith_deg", "view_azimuth_deg", "solar_zenith_deg",
            "solar_azimuth_deg", "relative_azimuth_deg", "local_view_angle_deg",
            "local_solar_incidence_deg", "offaxis_deg", "surface_view_cos",
        ]].to_numpy(np.float64)
    ).all(axis=1)
    obs = obs.loc[finite_geometry].reset_index(drop=True)

    spectra = np.load(paths.spectra, mmap_mode="r")
    if spectra.ndim != 2:
        raise ValueError(f"Expected a two-dimensional spectra matrix, got {spectra.shape}")
    obs_ids = pd.to_numeric(obs["observation_id"], errors="raise").to_numpy(np.int64)
    if len(obs_ids) and (obs_ids.min() < 0 or obs_ids.max() >= spectra.shape[0]):
        raise IndexError("observation_id is outside directional_brf_spectra.npy")
    response_columns = representative_spectra_columns(paths, spectra.shape[1])
    response = read_representative_response(spectra, obs_ids, response_columns)
    valid_response = np.isfinite(response).all(axis=1)
    if positive_reflectance_only:
        valid_response &= (response[:, :5] > 0.0).all(axis=1)
        valid_response &= (response[:, 5] >= -1.0) & (response[:, 5] <= 1.0)
    obs = obs.loc[valid_response].reset_index(drop=True)
    response = response[valid_response]

    points = pd.read_csv(paths.point_index)
    required_point_cols = {
        "point_id", "x_local", "y_local", "z_local", "red", "green", "blue",
        "normal_z", "surface_verticality", "normal_confidence",
    }
    missing = sorted(required_point_cols - set(points.columns))
    if missing:
        raise ValueError(f"point_index.csv lacks columns: {missing}")
    points = points.sort_values("point_id", kind="mergesort").reset_index(drop=True)
    expected = np.arange(len(points), dtype=np.int64)
    actual = pd.to_numeric(points["point_id"], errors="raise").to_numpy(np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError("point_id must be contiguous and zero based")

    inventory = {
        "rows": int(len(obs)),
        "points": int(obs["point_id"].nunique()),
        "scenes": int(obs["scene_id"].nunique()),
        "clear_only": bool(clear_only),
        "positive_reflectance_only": bool(positive_reflectance_only),
        "point_sample_modulus": int(point_sample_modulus),
        "point_sample_remainder": int(point_sample_remainder),
        "band_numbers": BAND_NUMBERS.tolist(),
        "band_names": list(BAND_NAMES),
        "source_spectra_columns": response_columns.tolist(),
    }
    return obs, response, points, inventory


def _numeric(obs: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in obs:
        return np.full(len(obs), default, dtype=np.float64)
    values = pd.to_numeric(obs[name], errors="coerce").to_numpy(np.float64)
    return np.where(np.isfinite(values), values, default)


def ross_li_kernels(
    view_zenith_deg: np.ndarray,
    solar_zenith_deg: np.ndarray,
    relative_azimuth_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    tv = np.deg2rad(np.clip(view_zenith_deg, 0.0, 89.0))
    ts = np.deg2rad(np.clip(solar_zenith_deg, 0.0, 89.0))
    phi = np.deg2rad(np.mod(relative_azimuth_deg, 360.0))
    cos_xi = np.clip(
        np.cos(ts) * np.cos(tv) + np.sin(ts) * np.sin(tv) * np.cos(phi),
        -1.0,
        1.0,
    )
    xi = np.arccos(cos_xi)
    ross = ((np.pi / 2.0 - xi) * cos_xi + np.sin(xi)) / np.maximum(
        np.cos(ts) + np.cos(tv), 1.0e-6
    ) - np.pi / 4.0

    tan_s = np.tan(ts)
    tan_v = np.tan(tv)
    sec_s = 1.0 / np.maximum(np.cos(ts), 1.0e-6)
    sec_v = 1.0 / np.maximum(np.cos(tv), 1.0e-6)
    distance = np.sqrt(np.maximum(tan_s**2 + tan_v**2 - 2.0 * tan_s * tan_v * np.cos(phi), 0.0))
    temp = np.sqrt(distance**2 + (tan_s * tan_v * np.sin(phi)) ** 2)
    cos_t = np.clip(2.0 * temp / np.maximum(sec_s + sec_v, 1.0e-6), -1.0, 1.0)
    t = np.arccos(cos_t)
    overlap = (t - np.sin(t) * np.cos(t)) * (sec_s + sec_v) / np.pi
    li = overlap - sec_s - sec_v + 0.5 * (1.0 + cos_xi) * sec_s * sec_v
    return ross, li


def build_features(
    obs: pd.DataFrame,
    points: pd.DataFrame,
    requested: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], pd.DataFrame, np.ndarray]:


    requested_sets = {"ross_li", "extended"} if requested is None else set(requested)
    unknown = requested_sets - {"ross_li", "extended"}
    if unknown or not requested_sets:
        raise ValueError(f"Unknown or empty feature-set request: {sorted(unknown)}")
    point_ids = pd.to_numeric(obs["point_id"], errors="raise").to_numpy(np.int64)
    vz = _numeric(obs, "view_zenith_deg")
    va = _numeric(obs, "view_azimuth_deg")
    sz = _numeric(obs, "solar_zenith_deg")
    sa = _numeric(obs, "solar_azimuth_deg")
    raz = _numeric(obs, "relative_azimuth_deg")
    lv = _numeric(obs, "local_view_angle_deg")
    ls = _numeric(obs, "local_solar_incidence_deg")
    ross, li = ross_li_kernels(vz, sz, raz)
    vzr, var, szr, sar, razr = map(np.deg2rad, [vz, va, sz, sa, raz])
    local_phase = np.clip(
        np.cos(np.deg2rad(lv)) * np.cos(np.deg2rad(ls))
        + np.sin(np.deg2rad(lv)) * np.sin(np.deg2rad(ls)) * np.cos(razr),
        -1.0,
        1.0,
    )
    scene_num = obs["scene_id"].astype(str).map(scene_suffix).to_numpy(np.float64)
    time_center = float(np.mean(scene_num))
    time_scale = max(float(np.std(scene_num)), 1.0)
    time = (scene_num - time_center) / time_scale

    red = points["red"].to_numpy(np.float64)[point_ids] / 255.0
    green = points["green"].to_numpy(np.float64)[point_ids] / 255.0
    blue = points["blue"].to_numpy(np.float64)[point_ids] / 255.0
    vertical = points["surface_verticality"].to_numpy(np.float64)[point_ids]
    nz = points["normal_z"].to_numpy(np.float64)[point_ids]
    zlocal = points["z_local"].to_numpy(np.float64)[point_ids]
    zlocal = (zlocal - float(np.nanmedian(zlocal))) / max(float(np.nanstd(zlocal)), 1.0e-6)

    kernel_names = [
        "ross_thick", "li_sparse", "cos_view", "cos_sun", "cos_local_view",
        "cos_local_sun", "cos_local_phase", "sin_raa", "cos_raa",
    ]
    kernel = np.column_stack([
        ross,
        li,
        np.cos(vzr),
        np.cos(szr),
        np.cos(np.deg2rad(lv)),
        np.cos(np.deg2rad(ls)),
        local_phase,
        np.sin(razr),
        np.cos(razr),
    ])
    pose_names = [
        "sin_view_az", "cos_view_az", "sin_solar_az", "cos_solar_az",
        "time", "time2", "offaxis", "border_log", "empty_cone_log",
        "surface_view_cos", "normal_confidence", "range", "u_center", "v_center",
    ]
    pose = np.column_stack([
        np.sin(var),
        np.cos(var),
        np.sin(sar),
        np.cos(sar),
        time,
        time * time,
        _numeric(obs, "offaxis_deg") / 30.0,
        np.log1p(np.maximum(_numeric(obs, "border_dist_px"), 0.0)) / np.log(1887.0),
        np.log1p(np.maximum(_numeric(obs, "local_empty_cone_deg"), 0.0)) / np.log(91.0),
        _numeric(obs, "surface_view_cos", 0.5),
        _numeric(obs, "normal_confidence", 1.0),
        _numeric(obs, "range_m") / 300.0,
        (_numeric(obs, "u") - 943.0) / 943.0,
        (_numeric(obs, "v") - 943.0) / 943.0,
    ])
    interaction_names = [
        "ross_red", "ross_green", "ross_blue", "ross_verticality", "ross_normal_z", "ross_z",
        "li_red", "li_green", "li_blue", "li_verticality", "li_normal_z", "li_z",
        "phase_verticality", "phase_normal_z",
    ]
    interactions = np.column_stack([
        ross * red,
        ross * green,
        ross * blue,
        ross * vertical,
        ross * nz,
        ross * zlocal,
        li * red,
        li * green,
        li * blue,
        li * vertical,
        li * nz,
        li * zlocal,
        local_phase * vertical,
        local_phase * nz,
    ])
    feature_sets: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    if "extended" in requested_sets:
        extended = np.column_stack([kernel, pose, interactions])
        if "ross_li" not in requested_sets:


            del kernel, pose, interactions
        np.nan_to_num(extended, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        feature_sets["extended"] = extended.astype(np.float32)
        feature_names["extended"] = kernel_names + pose_names + interaction_names
    if "ross_li" in requested_sets:
        np.nan_to_num(kernel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        feature_sets["ross_li"] = kernel.astype(np.float32)
        feature_names["ross_li"] = kernel_names
    feature_sets = {
        name: feature_sets[name] for name in ("ross_li", "extended")
        if name in feature_sets
    }
    feature_names = {
        name: feature_names[name] for name in ("ross_li", "extended")
        if name in feature_names
    }

    scene_table = obs.groupby("scene_id", sort=True, observed=False).agg(
        scene_num=("scene_id", lambda x: scene_suffix(str(x.iloc[0]))),
        view_zenith=("view_zenith_deg", "median"),
        view_azimuth=("view_azimuth_deg", "median"),
        solar_zenith=("solar_zenith_deg", "median"),
        solar_azimuth=("solar_azimuth_deg", "median"),
    )
    sn = scene_table["scene_num"].to_numpy(np.float64)
    st = (sn - time_center) / time_scale
    sva = np.deg2rad(scene_table["view_azimuth"].to_numpy(np.float64))
    ssa = np.deg2rad(scene_table["solar_azimuth"].to_numpy(np.float64))
    scene_x = np.column_stack([
        st,
        st * st,
        scene_table["view_zenith"].to_numpy(np.float64) / 60.0,
        np.sin(sva),
        np.cos(sva),
        scene_table["solar_zenith"].to_numpy(np.float64) / 90.0,
        np.sin(ssa),
        np.cos(ssa),
    ]).astype(np.float64)
    return feature_sets, feature_names, scene_table, scene_x


def select_development_scenes(scene_table: pd.DataFrame) -> tuple[list[str], list[str]]:

    table = scene_table.copy()
    table["zenith_bin"] = pd.qcut(table["view_zenith"], q=5, labels=False, duplicates="drop")
    development: list[str] = []
    confirmation: list[str] = []
    for _, group in table.groupby("zenith_bin", sort=True):
        group = group.sort_values(["view_azimuth", "scene_num"], kind="mergesort")
        for position, scene in enumerate(group.index.astype(str)):
            (development if position % 2 == 0 else confirmation).append(scene)
    development.sort(key=scene_suffix)
    confirmation.sort(key=scene_suffix)
    return development, confirmation


def point_stats(
    point_ids: np.ndarray,
    response: np.ndarray,
    mask: np.ndarray,
    point_count: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ids = point_ids[mask]
    y = response[mask].astype(np.float64, copy=False)
    if weights is None:
        w = np.ones(len(ids), dtype=np.float64)
    else:
        w = np.asarray(weights[mask], dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    count = np.bincount(ids, weights=w, minlength=point_count).astype(np.float64)
    sums = np.empty((point_count, response.shape[1]), dtype=np.float64)
    for band in range(response.shape[1]):
        sums[:, band] = np.bincount(ids, weights=w * y[:, band], minlength=point_count)
    mean = np.divide(sums, count[:, None], out=np.zeros_like(sums), where=count[:, None] > 0.0)
    return count, mean


def quality_weights(obs: pd.DataFrame) -> np.ndarray:
    surface = np.clip(_numeric(obs, "surface_view_cos", 0.5), 0.05, 1.0)
    normal = np.clip(_numeric(obs, "normal_confidence", 0.8), 0.25, 1.0)
    offaxis = np.exp(-np.maximum(_numeric(obs, "offaxis_deg"), 0.0) / 60.0)
    cone = np.clip(_numeric(obs, "local_empty_cone_deg", 4.0) / 20.0, 0.25, 1.0)
    return np.asarray(surface * normal * offaxis * cone, dtype=np.float64)


def point_mean_prediction(
    point_ids: np.ndarray,
    response: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    point_count: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    count, mean = point_stats(point_ids, response, train, point_count, weights=weights)
    ids = point_ids[test]
    prediction = np.full((len(ids), response.shape[1]), np.nan, dtype=np.float64)
    supported = count[ids] > 0.0
    prediction[supported] = mean[ids[supported]]
    return prediction


def ridge_angular_prediction(
    features: np.ndarray,
    point_ids: np.ndarray,
    response: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    point_count: int,
    *,
    alpha: float,
    max_fit_rows: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    count, point_mean_y = point_stats(point_ids, response, train, point_count)
    ids_train = point_ids[train]
    ids_test = point_ids[test]
    x_train = np.asarray(features[train], dtype=np.float64)
    x_test = np.asarray(features[test], dtype=np.float64)
    y_train = np.asarray(response[train], dtype=np.float64)

    feature_sum = np.empty((point_count, x_train.shape[1]), dtype=np.float64)
    for column in range(x_train.shape[1]):
        feature_sum[:, column] = np.bincount(
            ids_train,
            weights=x_train[:, column],
            minlength=point_count,
        )
    point_mean_x = np.divide(
        feature_sum,
        count[:, None],
        out=np.zeros_like(feature_sum),
        where=count[:, None] > 0.0,
    )
    eligible = count[ids_train] >= 2.0
    fit_indices = np.flatnonzero(eligible)
    if max_fit_rows is not None and len(fit_indices) > max_fit_rows:
        rng = np.random.default_rng(seed)
        fit_indices = np.sort(rng.choice(fit_indices, size=max_fit_rows, replace=False))
    fit_ids = ids_train[fit_indices]
    fit_x = x_train[fit_indices] - point_mean_x[fit_ids]
    fit_y = y_train[fit_indices] - point_mean_y[fit_ids]
    scale = np.std(fit_x, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    fit_x /= scale
    test_x = (x_test - point_mean_x[ids_test]) / scale
    regression = np.eye(fit_x.shape[1], dtype=np.float64) * float(alpha)
    beta = np.linalg.solve(fit_x.T @ fit_x + regression, fit_x.T @ fit_y)
    angular = test_x @ beta
    prediction = np.full((len(ids_test), response.shape[1]), np.nan, dtype=np.float64)
    supported = count[ids_test] > 0.0
    prediction[supported] = point_mean_y[ids_test[supported]] + angular[supported]
    return prediction, angular


def voxel_codes_for_points(points: pd.DataFrame, shape_m: Iterable[float]) -> np.ndarray:
    shape = np.asarray(list(shape_m), dtype=np.float64)
    if shape.shape != (3,) or np.any(shape <= 0.0):
        raise ValueError("shape_m must contain three positive values")
    xyz = points[["x_local", "y_local", "z_local"]].to_numpy(np.float64)
    key = np.floor(xyz / shape).astype(np.int32)
    codes, _ = pd.factorize(pd.MultiIndex.from_arrays(key.T), sort=False)
    return codes.astype(np.int64, copy=False)


def spatial_shrink_prediction(
    point_ids: np.ndarray,
    voxel_by_point: np.ndarray,
    response: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    point_count: int,
    shrink: float,
) -> np.ndarray:
    count, point_mean = point_stats(point_ids, response, train, point_count)
    train_voxels = voxel_by_point[point_ids[train]]
    voxel_count = int(voxel_by_point.max()) + 1
    vcount = np.bincount(train_voxels, minlength=voxel_count).astype(np.float64)
    vsum = np.empty((voxel_count, response.shape[1]), dtype=np.float64)
    train_y = response[train].astype(np.float64, copy=False)
    for band in range(response.shape[1]):
        vsum[:, band] = np.bincount(train_voxels, weights=train_y[:, band], minlength=voxel_count)
    vmean = np.divide(vsum, vcount[:, None], out=np.zeros_like(vsum), where=vcount[:, None] > 0.0)
    ids = point_ids[test]
    voxels = voxel_by_point[ids]
    supported = (count[ids] > 0.0) & (vcount[voxels] > 0.0)
    weight = count[ids] / (count[ids] + float(shrink))
    prediction = np.full((len(ids), response.shape[1]), np.nan, dtype=np.float64)
    prediction[supported] = (
        weight[supported, None] * point_mean[ids[supported]]
        + (1.0 - weight[supported, None]) * vmean[voxels[supported]]
    )
    return prediction


def scene_metadata_bias(
    point_ids: np.ndarray,
    response: np.ndarray,
    scene_ids: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    point_count: int,
    scene_table: pd.DataFrame,
    scene_x: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    count, mean = point_stats(point_ids, response, train, point_count)
    table_order = scene_table.index.astype(str).to_numpy()
    lookup = {scene: idx for idx, scene in enumerate(table_order)}
    train_scenes = np.unique(scene_ids[train])
    metadata: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for scene in train_scenes:
        scene_mask = train & (scene_ids == scene)
        ids = point_ids[scene_mask]
        other_count = count[ids] - 1.0
        good = other_count > 0.0
        if int(good.sum()) < 100:
            continue
        yy = response[scene_mask].astype(np.float64, copy=False)
        other_mean = (mean[ids[good]] * count[ids[good], None] - yy[good]) / other_count[good, None]
        residual = yy[good] - other_mean
        biases.append(np.nanmedian(residual, axis=0))
        metadata.append(scene_x[lookup[str(scene)]])
    base = point_mean_prediction(point_ids, response, train, test, point_count)
    if len(metadata) < scene_x.shape[1] + 2:
        return base
    x = np.asarray(metadata, dtype=np.float64)
    y = np.asarray(biases, dtype=np.float64)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    design = np.column_stack([np.ones(len(x)), (x - center) / scale])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    heldout = str(np.unique(scene_ids[test])[0])
    tx = (scene_x[lookup[heldout]] - center) / scale
    bias = np.r_[1.0, tx] @ beta
    return base + bias


def metric_row(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y_true) & np.isfinite(prediction)
    yy = np.asarray(y_true[valid], dtype=np.float64)
    pp = np.asarray(prediction[valid], dtype=np.float64)
    if not len(yy):
        return {"n": 0, "coverage": 0.0, "rmse": math.nan, "mae": math.nan, "bias": math.nan, "r2": math.nan}
    error = pp - yy
    denominator = float(np.sum((yy - yy.mean()) ** 2))
    return {
        "n": int(len(yy)),
        "coverage": float(valid.mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error * error) / denominator) if denominator > 0.0 else math.nan,
    }


class MetricAccumulator:
    def __init__(self, bands: int) -> None:
        self.n = np.zeros(bands, dtype=np.int64)
        self.total = np.zeros(bands, dtype=np.int64)
        self.sum_y = np.zeros(bands, dtype=np.float64)
        self.sum_y2 = np.zeros(bands, dtype=np.float64)
        self.sse = np.zeros(bands, dtype=np.float64)
        self.sae = np.zeros(bands, dtype=np.float64)
        self.sum_error = np.zeros(bands, dtype=np.float64)

    def update(self, y_true: np.ndarray, prediction: np.ndarray) -> None:
        for band in range(y_true.shape[1]):
            valid = np.isfinite(y_true[:, band]) & np.isfinite(prediction[:, band])
            yy = y_true[valid, band].astype(np.float64, copy=False)
            pp = prediction[valid, band].astype(np.float64, copy=False)
            error = pp - yy
            self.total[band] += len(y_true)
            self.n[band] += len(yy)
            self.sum_y[band] += yy.sum()
            self.sum_y2[band] += np.dot(yy, yy)
            self.sse[band] += np.dot(error, error)
            self.sae[band] += np.abs(error).sum()
            self.sum_error[band] += error.sum()

    def rows(self, candidate: str, subset: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, band in enumerate(BAND_NUMBERS):
            n = int(self.n[index])
            denominator = self.sum_y2[index] - (self.sum_y[index] ** 2 / n) if n else math.nan
            rows.append({
                "candidate": candidate,
                "subset": subset,
                "band": int(band),
                "variable": BAND_NAMES[index],
                "n": n,
                "coverage": float(n / self.total[index]) if self.total[index] else 0.0,
                "rmse": float(math.sqrt(self.sse[index] / n)) if n else math.nan,
                "mae": float(self.sae[index] / n) if n else math.nan,
                "bias": float(self.sum_error[index] / n) if n else math.nan,
                "r2": float(1.0 - self.sse[index] / denominator) if n and denominator > 0.0 else math.nan,
            })
        return rows


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
