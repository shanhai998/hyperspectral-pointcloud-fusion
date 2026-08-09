
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from hyperspectral_pointcloud_fusion.common import ensure_dir, load_config


def safe_rmtree(path: Path, anchor: Path) -> None:
    target = path.resolve()
    root = anchor.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refuse to remove unsafe path: {target}")
    if target.exists():
        shutil.rmtree(target)


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce")


def point_error(df: pd.DataFrame, prefix: str, target: str) -> pd.Series:
    u = numeric(df, f"{prefix}_err_u_{target}_px")
    v = numeric(df, f"{prefix}_err_v_{target}_px")
    return np.sqrt(u * u + v * v)


def rmse_from_targets(df: pd.DataFrame, prefix: str) -> pd.Series:
    top = point_error(df, prefix, "top")
    base = point_error(df, prefix, "base")
    values = pd.concat([top.rename("top"), base.rename("base")], axis=1)
    return np.sqrt((values * values).mean(axis=1, skipna=True))


def write_registration_report(cfg: dict, result_dir: Path) -> Path:
    correction_csv = Path(cfg["outputs"]["target_calibration_dir"]) / "per_scene_correction.csv"
    if not correction_csv.exists():
        raise FileNotFoundError(f"配准修正报告不存在: {correction_csv}")

    df = pd.read_csv(correction_csv, encoding="utf-8-sig")
    out = pd.DataFrame()
    for col in ["scene_id", "annotation_status", "success", "solver_dof", "rotation_total_deg"]:
        if col in df.columns:
            out[col] = df[col]

    out["pre_top_error_px"] = point_error(df, "pre", "top")
    out["pre_base_error_px"] = point_error(df, "pre", "base")
    out["pre_registration_rmse_px"] = rmse_from_targets(df, "pre")
    out["post_top_error_px"] = point_error(df, "post", "top")
    out["post_base_error_px"] = point_error(df, "post", "base")
    out["post_registration_rmse_px"] = rmse_from_targets(df, "post")

    for col in [
        "pre_err_u_top_px",
        "pre_err_v_top_px",
        "pre_err_u_base_px",
        "pre_err_v_base_px",
        "post_err_u_top_px",
        "post_err_v_top_px",
        "post_err_u_base_px",
        "post_err_v_base_px",
        "rx_deg",
        "ry_deg",
        "rz_deg",
        "clipped_by_max_angle",
        "note",
    ]:
        if col in df.columns:
            out[col] = df[col]

    report_path = result_dir / "hyperspectral_pointcloud_fusion_registration_accuracy_report.csv"
    out.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"配准精度报告: {report_path}")
    return report_path


def remove_stale_projection_outputs(result_dir: Path) -> None:
    for name in [
        "hyperspectral_pointcloud_fusion_projection_before_registration",
        "hyperspectral_pointcloud_fusion_projection_after_registration",
    ]:
        stale_dir = result_dir / name
        if stale_dir.exists():
            safe_rmtree(stale_dir, result_dir)
            print(f"已移除旧投影结果目录: {stale_dir}")


def copy_fused_pointcloud(cfg: dict, result_dir: Path) -> Path:
    stem = str(cfg["project"]["pointcloud_stem"])
    all_bands = bool(cfg["project"].get("all_bands", False))
    name = f"hpf_{stem}_allbands.ply" if all_bands else f"hpf_{stem}.ply"
    source = Path(cfg["outputs"]["product_dir"]) / name
    if not source.exists():
        raise FileNotFoundError(f"融合点云不存在: {source}")
    target = result_dir / name
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)
    print(f"融合点云: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--cleanup-work-root", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_dir = ensure_dir(args.result_dir)

    remove_stale_projection_outputs(result_dir)
    write_registration_report(cfg, result_dir)
    copy_fused_pointcloud(cfg, result_dir)

    if args.cleanup_work_root:
        work_root = Path(cfg["project"]["output_root"]).resolve()
        try:
            safe_rmtree(work_root, result_dir)
            print(f"已清理临时工作目录: {work_root}")
        except Exception as exc:
            print(f"[WARN] 临时工作目录清理失败，不影响最终结果: {exc}")


if __name__ == "__main__":
    main()
