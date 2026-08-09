
import math

import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def evaluate_candidate_direction(
    neighbor_offsets: np.ndarray,
    direction: np.ndarray,
    clearance_m: float,
    min_blocker_along_m: float = 0.0,
) -> dict:
    offsets = np.asarray(neighbor_offsets, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    dn = float(np.linalg.norm(d))
    if dn <= 1e-12:
        d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        d = d / dn
    if offsets.size == 0:
        return {
            'is_clear': True,
            'blocker_count': 0,
            'front_count': 0,
            'empty_cone_deg': 180.0,
            'nearest_blocker_along_m': np.inf,
            'nearest_blocker_perp_m': np.inf,
            'max_front_cos': -1.0,
        }
    dist2 = np.sum(offsets * offsets, axis=1)
    t = offsets @ d
    front = t > max(1e-9, float(min_blocker_along_m))
    front_count = int(np.count_nonzero(front))
    if front_count == 0:
        return {
            'is_clear': True,
            'blocker_count': 0,
            'front_count': 0,
            'empty_cone_deg': 180.0,
            'nearest_blocker_along_m': np.inf,
            'nearest_blocker_perp_m': np.inf,
            'max_front_cos': -1.0,
        }
    perp2 = dist2[front] - t[front] * t[front]
    blockers = perp2 <= float(clearance_m) ** 2
    blocker_count = int(np.count_nonzero(blockers))
    is_clear = blocker_count == 0
    nearest_blocker_along = float(np.min(t[front][blockers])) if blocker_count else np.inf
    nearest_blocker_perp = float(np.sqrt(max(0.0, float(np.min(perp2[blockers]))))) if blocker_count else np.inf
    front_offsets = offsets[front]
    front_unit = normalize_rows(front_offsets)
    cosines = front_unit @ d
    max_front_cos = float(np.max(cosines)) if cosines.size else -1.0
    max_front_cos = min(1.0, max(-1.0, max_front_cos))
    empty_cone_deg = float(np.degrees(np.arccos(max_front_cos))) if cosines.size else 180.0
    return {
        'is_clear': bool(is_clear),
        'blocker_count': blocker_count,
        'front_count': front_count,
        'empty_cone_deg': empty_cone_deg,
        'nearest_blocker_along_m': nearest_blocker_along,
        'nearest_blocker_perp_m': nearest_blocker_perp,
        'max_front_cos': max_front_cos,
    }


def _unit_interval(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def build_selection_score(
    coarse_score: float,
    empty_cone_deg: float,
    blocker_count: int,
    offaxis_deg: float,
    range_m: float,
    weights: dict,
    view_zenith_deg: float = 0.0,
    surface_view_cos: float = 0.0,
    surface_verticality: float = 0.0,
    normal_confidence: float = 0.0,
) -> float:
    normal_reliability = float(weights.get('normal_confidence_floor', 0.25)) + (1.0 - float(weights.get('normal_confidence_floor', 0.25))) * _unit_interval(normal_confidence)
    surface_alignment_bonus = (
        float(weights.get('surface_alignment_weight', 0.0))
        * _unit_interval(surface_view_cos)
        * normal_reliability
    )
    vertical_view_bonus = (
        float(weights.get('vertical_view_zenith_weight', 0.0))
        * max(0.0, float(view_zenith_deg))
        * _unit_interval(surface_verticality)
        * normal_reliability
    )
    return (
        float(weights.get('coarse_score_weight', 1.0)) * float(coarse_score)
        + float(weights.get('empty_cone_weight', 0.5)) * float(empty_cone_deg)
        - float(weights.get('blocker_count_weight', 2.0)) * float(blocker_count)
        - float(weights.get('offaxis_weight', 0.0)) * float(offaxis_deg)
        - float(weights.get('range_weight', 0.0)) * float(range_m)
        + surface_alignment_bonus
        + vertical_view_bonus
    )


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def build_multiview_weight(
    coarse_score: float,
    empty_cone_deg: float,
    blocker_count: int,
    offaxis_deg: float,
    range_m: float,
    border_dist_px: float,
    is_clear: bool,
    params: dict,
    view_zenith_deg: float = 0.0,
    surface_view_cos: float = 0.0,
    surface_verticality: float = 0.0,
    normal_confidence: float = 0.0,
) -> float:
    coarse_scale = max(1e-6, float(params.get('coarse_score_scale', 8.0)))
    offaxis_scale = max(1e-6, float(params.get('offaxis_scale_deg', 12.0)))
    range_scale = max(1e-6, float(params.get('range_scale_m', 40.0)))
    border_scale = max(1e-6, float(params.get('border_scale_px', 80.0)))
    cone_scale = max(1e-6, float(params.get('cone_scale_deg', 45.0)))
    blocker_penalty = max(0.0, float(params.get('blocker_penalty', 0.7)))
    clear_boost = max(1e-6, float(params.get('clear_boost', 1.0)))
    unclear_penalty = max(1e-6, float(params.get('unclear_penalty', 0.25)))
    min_weight = max(0.0, float(params.get('min_weight', 1e-6)))
    normal_floor = _unit_interval(float(params.get('normal_confidence_floor', 0.25)))
    alignment_floor = _unit_interval(float(params.get('surface_alignment_floor', 0.05)))
    alignment_power = max(1e-6, float(params.get('surface_alignment_power', 1.0)))
    vertical_view_boost = max(0.0, float(params.get('vertical_view_boost', 0.0)))
    vertical_view_power = max(1e-6, float(params.get('vertical_view_power', 1.0)))

    coarse_term = _sigmoid(float(coarse_score) / coarse_scale)
    offaxis_term = math.exp(-max(0.0, float(offaxis_deg)) / offaxis_scale)
    range_term = math.exp(-max(0.0, float(range_m)) / range_scale)
    border_term = 1.0 - math.exp(-max(0.0, float(border_dist_px)) / border_scale)
    cone_term = 1.0 - math.exp(-max(0.0, float(empty_cone_deg)) / cone_scale)
    blocker_term = math.exp(-blocker_penalty * max(0, int(blocker_count)))
    visibility_term = clear_boost if bool(is_clear) else unclear_penalty
    normal_reliability = normal_floor + (1.0 - normal_floor) * _unit_interval(normal_confidence)
    alignment = _unit_interval(surface_view_cos)
    surface_term = alignment_floor + (1.0 - alignment_floor) * (alignment ** alignment_power) * normal_reliability
    zenith_fraction = _unit_interval(max(0.0, float(view_zenith_deg)) / 90.0)
    vertical_term = 1.0 + vertical_view_boost * _unit_interval(surface_verticality) * (zenith_fraction ** vertical_view_power) * normal_reliability

    raw = (
        coarse_term
        * offaxis_term
        * range_term
        * max(border_term, 1e-6)
        * max(cone_term, 1e-6)
        * blocker_term
        * visibility_term
        * surface_term
        * vertical_term
    )
    return float(max(raw, min_weight))


def _clip01_arr(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def build_selection_score_vec(
    coarse_score: np.ndarray,
    empty_cone_deg: np.ndarray,
    blocker_count: np.ndarray,
    offaxis_deg: np.ndarray,
    range_m: np.ndarray,
    weights: dict,
    view_zenith_deg: np.ndarray,
    surface_view_cos: np.ndarray,
    surface_verticality: np.ndarray,
    normal_confidence: np.ndarray,
) -> np.ndarray:


    nf = float(weights.get('normal_confidence_floor', 0.25))
    normal_reliability = nf + (1.0 - nf) * _clip01_arr(normal_confidence)
    surface_alignment_bonus = (
        float(weights.get('surface_alignment_weight', 0.0))
        * _clip01_arr(surface_view_cos)
        * normal_reliability
    )
    vertical_view_bonus = (
        float(weights.get('vertical_view_zenith_weight', 0.0))
        * np.maximum(0.0, np.asarray(view_zenith_deg, dtype=np.float64))
        * _clip01_arr(surface_verticality)
        * normal_reliability
    )
    return (
        float(weights.get('coarse_score_weight', 1.0)) * np.asarray(coarse_score, dtype=np.float64)
        + float(weights.get('empty_cone_weight', 0.5)) * np.asarray(empty_cone_deg, dtype=np.float64)
        - float(weights.get('blocker_count_weight', 2.0)) * np.asarray(blocker_count, dtype=np.float64)
        - float(weights.get('offaxis_weight', 0.0)) * np.asarray(offaxis_deg, dtype=np.float64)
        - float(weights.get('range_weight', 0.0)) * np.asarray(range_m, dtype=np.float64)
        + surface_alignment_bonus
        + vertical_view_bonus
    )


def _sigmoid_arr(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def build_multiview_weight_vec(
    coarse_score: np.ndarray,
    empty_cone_deg: np.ndarray,
    blocker_count: np.ndarray,
    offaxis_deg: np.ndarray,
    range_m: np.ndarray,
    border_dist_px: np.ndarray,
    is_clear: np.ndarray,
    params: dict,
    view_zenith_deg: np.ndarray,
    surface_view_cos: np.ndarray,
    surface_verticality: np.ndarray,
    normal_confidence: np.ndarray,
) -> np.ndarray:

    coarse_scale = max(1e-6, float(params.get('coarse_score_scale', 8.0)))
    offaxis_scale = max(1e-6, float(params.get('offaxis_scale_deg', 12.0)))
    range_scale = max(1e-6, float(params.get('range_scale_m', 40.0)))
    border_scale = max(1e-6, float(params.get('border_scale_px', 80.0)))
    cone_scale = max(1e-6, float(params.get('cone_scale_deg', 45.0)))
    blocker_penalty = max(0.0, float(params.get('blocker_penalty', 0.7)))
    clear_boost = max(1e-6, float(params.get('clear_boost', 1.0)))
    unclear_penalty = max(1e-6, float(params.get('unclear_penalty', 0.25)))
    min_weight = max(0.0, float(params.get('min_weight', 1e-6)))
    normal_floor = float(params.get('normal_confidence_floor', 0.25))
    alignment_floor = float(params.get('surface_alignment_floor', 0.05))
    alignment_power = max(1e-6, float(params.get('surface_alignment_power', 1.0)))
    vertical_view_boost = max(0.0, float(params.get('vertical_view_boost', 0.0)))
    vertical_view_power = max(1e-6, float(params.get('vertical_view_power', 1.0)))

    cs = np.asarray(coarse_score, dtype=np.float64)
    cone = np.asarray(empty_cone_deg, dtype=np.float64)
    bc = np.asarray(blocker_count, dtype=np.float64)
    oa = np.maximum(0.0, np.asarray(offaxis_deg, dtype=np.float64))
    rg = np.maximum(0.0, np.asarray(range_m, dtype=np.float64))
    bd = np.maximum(0.0, np.asarray(border_dist_px, dtype=np.float64))
    cone_pos = np.maximum(0.0, cone)
    svc = _clip01_arr(surface_view_cos)
    svv = _clip01_arr(surface_verticality)
    nc = _clip01_arr(normal_confidence)
    vz = np.maximum(0.0, np.asarray(view_zenith_deg, dtype=np.float64))
    clear_mask = np.asarray(is_clear, dtype=bool)

    coarse_term = _sigmoid_arr(cs / coarse_scale)
    offaxis_term = np.exp(-oa / offaxis_scale)
    range_term = np.exp(-rg / range_scale)
    border_term = 1.0 - np.exp(-bd / border_scale)
    cone_term = 1.0 - np.exp(-cone_pos / cone_scale)
    blocker_term = np.exp(-blocker_penalty * np.maximum(0.0, bc))
    visibility_term = np.where(clear_mask, clear_boost, unclear_penalty)
    normal_reliability = normal_floor + (1.0 - normal_floor) * nc
    surface_term = alignment_floor + (1.0 - alignment_floor) * (svc ** alignment_power) * normal_reliability
    zenith_fraction = np.clip(vz / 90.0, 0.0, 1.0)
    vertical_term = 1.0 + vertical_view_boost * svv * (zenith_fraction ** vertical_view_power) * normal_reliability

    raw = (
        coarse_term
        * offaxis_term
        * range_term
        * np.maximum(border_term, 1e-6)
        * np.maximum(cone_term, 1e-6)
        * blocker_term
        * visibility_term
        * surface_term
        * vertical_term
    )
    return np.maximum(raw, min_weight).astype(np.float64)


def evaluate_candidate_directions_batch(
    neighbor_offsets: np.ndarray,
    directions: np.ndarray,
    clearance_m: float,
    min_blocker_along_m: float = 0.0,
) -> dict:


    offsets = np.asarray(neighbor_offsets, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    S = int(dirs.shape[0])
    if S == 0:
        return {
            'is_clear': np.zeros((0,), dtype=bool),
            'blocker_count': np.zeros((0,), dtype=np.int32),
            'front_count': np.zeros((0,), dtype=np.int32),
            'empty_cone_deg': np.zeros((0,), dtype=np.float32),
        }


    dn = np.linalg.norm(dirs, axis=1)
    valid = dn > 1e-12
    dirs_unit = np.zeros_like(dirs)
    dirs_unit[valid] = dirs[valid] / dn[valid, None]
    dirs_unit[~valid] = np.array([0.0, 0.0, 1.0])

    if offsets.size == 0:
        return {
            'is_clear': np.ones((S,), dtype=bool),
            'blocker_count': np.zeros((S,), dtype=np.int32),
            'front_count': np.zeros((S,), dtype=np.int32),
            'empty_cone_deg': np.full((S,), 180.0, dtype=np.float32),
        }


    t = offsets @ dirs_unit.T
    dist2 = np.einsum('ij,ij->i', offsets, offsets)
    front_thresh = max(1e-9, float(min_blocker_along_m))
    front_mask = t > front_thresh

    perp2 = dist2[:, None] - t * t
    np.maximum(perp2, 0.0, out=perp2)
    blocker_thresh2 = float(clearance_m) ** 2

    front_count = np.count_nonzero(front_mask, axis=0).astype(np.int32)
    blocker_mask = front_mask & (perp2 <= blocker_thresh2)
    blocker_count = np.count_nonzero(blocker_mask, axis=0).astype(np.int32)
    is_clear = blocker_count == 0


    offset_len = np.sqrt(np.maximum(dist2, 1e-24))
    cos_all = t / offset_len[:, None]

    cos_masked = np.where(front_mask, cos_all, -np.inf)
    max_cos = np.max(cos_masked, axis=0)

    no_front = front_count == 0
    max_cos_clipped = np.clip(np.where(no_front, -1.0, max_cos), -1.0, 1.0)
    empty_cone_deg = np.degrees(np.arccos(max_cos_clipped)).astype(np.float32)
    empty_cone_deg[no_front] = 180.0

    return {
        'is_clear': is_clear,
        'blocker_count': blocker_count,
        'front_count': front_count,
        'empty_cone_deg': empty_cone_deg,
    }
