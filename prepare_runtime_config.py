"""
Prepare a runtime config adapted to the actual airport mesh.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy

import trimesh
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config_path: str, cfg: dict) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def load_reference_mesh(mesh_dir: str) -> trimesh.Trimesh:
    mesh_files = [
        os.path.join(mesh_dir, name)
        for name in sorted(os.listdir(mesh_dir))
        if name.endswith(".obj") or name.endswith(".ply")
    ]
    meshes = []
    for path in mesh_files:
        mesh = trimesh.load(path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
            meshes.append(mesh)

    if not meshes:
        raise RuntimeError(f"No mesh found in {mesh_dir}. Run data_ingestion.py first.")

    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def odd_grid_size(span: float, target_resolution: float, min_value: int, max_value: int) -> int:
    n = int(math.ceil(span / max(target_resolution, 1e-6))) + 1
    n = max(min_value, min(max_value, n))
    if n % 2 == 0:
        n += 1 if n < max_value else -1
    return max(min_value, min(max_value, n))


def prepare_runtime_config(config_path: str, output_path: str) -> dict:
    cfg = load_config(config_path)
    runtime_cfg = deepcopy(cfg)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    mesh_dir = os.path.join(base_dir, cfg["paths"]["meshes"])
    mesh = load_reference_mesh(mesh_dir)

    bbox_min, bbox_max = mesh.bounds
    span = bbox_max - bbox_min
    max_xy_span = float(max(span[0], span[1]))
    max_z = float(max(bbox_max[2], 1.0))

    prep_cfg = cfg.get("runtime_preparation", {})
    xy_margin = float(prep_cfg.get("xy_margin_m", max(80.0, min(220.0, max_xy_span * 0.18))))
    z_margin = float(prep_cfg.get("z_margin_m", max(15.0, min(80.0, max_z * 0.35))))
    target_xy_resolution = float(prep_cfg.get("target_xy_resolution_m", 6.0))
    target_z_resolution = float(prep_cfg.get("target_z_resolution_m", 2.5))

    x_min = float(bbox_min[0] - xy_margin)
    x_max = float(bbox_max[0] + xy_margin)
    y_min = float(bbox_min[1] - xy_margin)
    y_max = float(bbox_max[1] + xy_margin)
    z_max = float(bbox_max[2] + z_margin)

    grid_nx = odd_grid_size(x_max - x_min, target_xy_resolution, min_value=161, max_value=321)
    grid_ny = odd_grid_size(y_max - y_min, target_xy_resolution, min_value=161, max_value=321)
    grid_nz = odd_grid_size(z_max, target_z_resolution, min_value=64, max_value=129)

    runtime_cfg["domain"] = {
        "x_min": round(x_min, 3),
        "x_max": round(x_max, 3),
        "y_min": round(y_min, 3),
        "y_max": round(y_max, 3),
        "z_max": round(z_max, 3),
        "grid_nx": int(grid_nx),
        "grid_ny": int(grid_ny),
        "grid_nz": int(grid_nz),
    }

    sampling_cfg = runtime_cfg.setdefault("sampling", {})
    sampling_cfg["n_surface_points"] = int(prep_cfg.get("n_surface_points", 250000))
    sampling_cfg["n_near_surface_points"] = int(prep_cfg.get("n_near_surface_points", 1600000))
    sampling_cfg["n_far_points"] = int(prep_cfg.get("n_far_points", 250000))
    sampling_cfg["near_surface_sigma"] = float(prep_cfg.get("near_surface_sigma", sampling_cfg.get("near_surface_sigma", 0.00075)))
    sampling_cfg["clamp_distance"] = float(prep_cfg.get("clamp_distance", sampling_cfg.get("clamp_distance", 2.0)))

    training_cfg = runtime_cfg.setdefault("training", {})
    training_cfg["n_epochs"] = int(prep_cfg.get("n_epochs", 200))
    training_cfg["save_every_n_epochs"] = int(prep_cfg.get("save_every_n_epochs", 20))
    training_cfg["early_stopping_patience"] = int(prep_cfg.get("early_stopping_patience", 50))
    training_cfg["validation_fraction"] = float(prep_cfg.get("validation_fraction", training_cfg.get("validation_fraction", 0.03)))

    monitoring_cfg = runtime_cfg.setdefault("monitoring", {})
    monitoring_cfg.setdefault("enabled", True)
    monitoring_cfg.setdefault("every_n_epochs", 20)
    monitoring_cfg.setdefault("warmup_epochs", 20)
    monitoring_cfg.setdefault("fail_fast_after_epoch", 60)
    monitoring_cfg.setdefault("collapse_patience", 5)
    monitoring_cfg.setdefault("min_pred_std", 0.02)
    monitoring_cfg.setdefault("max_surface_ratio_to_baseline_after_warmup", 0.65)
    monitoring_cfg.setdefault("persistent_bad_probes", 2)
    monitoring_cfg.setdefault("min_iou_after_fail_fast", 0.04)
    monitoring_cfg.setdefault("min_precision_after_fail_fast", 0.08)
    monitoring_cfg.setdefault("min_recall_after_fail_fast", 0.08)
    monitoring_cfg.setdefault("max_boundary_fraction_after_fail_fast", 0.55)
    monitoring_cfg.setdefault("chamfer_samples", 10000)
    monitoring_cfg.setdefault("prefer_geometry_checkpoint_min_iou", 0.10)
    monitoring_cfg.setdefault("mirror_every_n_epochs", 5)
    monitoring_cfg["coarse_grid_nx"] = int(prep_cfg.get("coarse_grid_nx", max(81, min(161, grid_nx // 2 if grid_nx > 180 else grid_nx))))
    monitoring_cfg["coarse_grid_ny"] = int(prep_cfg.get("coarse_grid_ny", max(81, min(161, grid_ny // 2 if grid_ny > 180 else grid_ny))))
    monitoring_cfg["coarse_grid_nz"] = int(prep_cfg.get("coarse_grid_nz", max(41, min(97, grid_nz // 2 if grid_nz > 80 else grid_nz))))

    runtime_cfg.setdefault("mask", {})
    runtime_cfg["mask"].setdefault("surface_level", 0.0)
    runtime_cfg["mask"].setdefault("prefer_cleaned_reconstruction", True)

    summary = {
        "mesh_bbox_min": [float(v) for v in bbox_min],
        "mesh_bbox_max": [float(v) for v in bbox_max],
        "mesh_span_xyz_m": [float(v) for v in span],
        "xy_margin_m": xy_margin,
        "z_margin_m": z_margin,
        "target_xy_resolution_m": target_xy_resolution,
        "target_z_resolution_m": target_z_resolution,
        "runtime_domain": runtime_cfg["domain"],
        "runtime_sampling": runtime_cfg["sampling"],
        "runtime_training": runtime_cfg["training"],
        "runtime_monitoring": runtime_cfg["monitoring"],
    }

    save_config(output_path, runtime_cfg)
    summary_path = os.path.splitext(output_path)[0] + "_preparation.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "config_path": output_path,
        "summary_path": summary_path,
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a runtime config adapted to the generated airport mesh.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="config_runtime.yaml")
    args = parser.parse_args()

    info = prepare_runtime_config(args.config, args.output)
    print("Runtime configuration ready.")
    print(f"  Config  : {info['config_path']}")
    print(f"  Summary : {info['summary_path']}")
    print(f"  Domain  : {info['runtime_domain']}")


if __name__ == "__main__":
    main()
