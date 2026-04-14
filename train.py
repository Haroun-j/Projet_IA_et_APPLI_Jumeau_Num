"""
Training script for the airport DeepSDF model.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import math
import os
import random
import shutil
import time
import traceback
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split
import yaml

from deepsdf_model import build_model
from evaluate_reconstruction import (
    clean_predicted_occupancy,
    make_binary_field,
    occupancy_metrics_from_grid,
    predict_grid,
    reconstruct_mesh_from_sdf,
    symmetric_chamfer,
)
from sdf_sampling import FootprintSignHelper, load_meshes


print = partial(builtins.print, flush=True)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clamped_l1_per_sample(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    pred_c = torch.clamp(pred, -delta, delta)
    target_c = torch.clamp(target, -delta, delta)
    return torch.abs(pred_c - target_c).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device: str,
    num_workers: int,
    sample_weights: torch.Tensor | None = None,
):
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=max(0, num_workers),
        pin_memory=(device == "cuda"),
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
        **loader_kwargs,
    )


def build_sample_weights(
    sdf: torch.Tensor,
    delta: float,
    surface_zero_eps: float,
    surface_focus_band: float,
) -> torch.Tensor:
    abs_sdf = sdf.abs().squeeze(-1)
    weights = torch.ones_like(abs_sdf)

    exact_surface = abs_sdf <= surface_zero_eps
    focus_band = (abs_sdf > surface_zero_eps) & (abs_sdf <= surface_focus_band)
    mid_band = (abs_sdf > surface_focus_band) & (abs_sdf <= min(delta, surface_focus_band * 3.0))
    far_band = abs_sdf >= max(surface_focus_band * 4.0, delta * 0.8)

    weights[exact_surface] = 0.15
    weights[focus_band] = 6.0
    weights[mid_band] = 2.0
    weights[far_band] = 0.5
    weights = weights * torch.where(sdf.squeeze(-1) < 0.0, 1.25, 1.0)
    return weights


def describe_sdf_distribution(sdf: torch.Tensor, delta: float, surface_zero_eps: float, surface_focus_band: float) -> None:
    abs_sdf = sdf.abs().squeeze(-1)
    print("\nDataset diagnostics:")
    print(f"  mean(|sdf|)            : {float(abs_sdf.mean()):.6f}")
    print(f"  median(|sdf|)          : {float(abs_sdf.median()):.6f}")
    print(f"  exact surface zeros    : {float((abs_sdf <= surface_zero_eps).float().mean()) * 100:.2f}%")
    print(f"  near-surface non-zero  : {float(((abs_sdf > surface_zero_eps) & (abs_sdf <= surface_focus_band)).float().mean()) * 100:.2f}%")
    print(f"  clamped far samples    : {float((abs_sdf >= delta * 0.99).float().mean()) * 100:.2f}%")


def autocast_context(device: str, amp_enabled: bool, amp_dtype: str):
    if not amp_enabled or device != "cuda":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_epoch(
    model,
    loader,
    optimizer,
    device: str,
    delta: float,
    grad_clip: float | None,
    amp_enabled: bool,
    amp_dtype: str,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    model.train()
    running_loss = 0.0
    for coords_b, sdf_b, weight_b in loader:
        coords_b = coords_b.to(device, non_blocking=True)
        sdf_b = sdf_b.to(device, non_blocking=True)
        weight_b = weight_b.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled, amp_dtype):
            pred = model(coords_b)
            per_sample = clamped_l1_per_sample(pred, sdf_b, delta)
            loss = (per_sample * weight_b).sum() / weight_b.sum().clamp_min(1e-8)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str,
    delta: float,
    surface_zero_eps: float,
    surface_focus_band: float,
    amp_enabled: bool,
    amp_dtype: str,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    zero_baseline_loss = 0.0
    running_surface_loss = 0.0
    running_surface_baseline = 0.0
    surface_batches = 0
    pred_mean = 0.0
    pred_std = 0.0
    n_batches = 0

    for coords_b, sdf_b, _weight_b in loader:
        coords_b = coords_b.to(device, non_blocking=True)
        sdf_b = sdf_b.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled, amp_dtype):
            pred = model(coords_b)

        per_sample = clamped_l1_per_sample(pred, sdf_b, delta)
        running_loss += per_sample.mean().item()

        target_c = torch.clamp(sdf_b, -delta, delta)
        zero_baseline_loss += target_c.abs().mean().item()

        abs_target = target_c.abs().squeeze(-1)
        focus_mask = (abs_target > surface_zero_eps) & (abs_target <= surface_focus_band)
        if focus_mask.any():
            running_surface_loss += per_sample[focus_mask].mean().item()
            running_surface_baseline += abs_target[focus_mask].mean().item()
            surface_batches += 1

        pred_c = torch.clamp(pred, -delta, delta)
        pred_mean += pred_c.mean().item()
        pred_std += pred_c.std().item()
        n_batches += 1

    return {
        "val_loss": running_loss / max(1, n_batches),
        "zero_baseline_loss": zero_baseline_loss / max(1, n_batches),
        "val_surface_loss": running_surface_loss / max(1, surface_batches),
        "surface_zero_baseline_loss": running_surface_baseline / max(1, surface_batches),
        "pred_mean": pred_mean / max(1, n_batches),
        "pred_std": pred_std / max(1, n_batches),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=json_default) + "\n")


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rebuild_epoch_records_from_history(history: dict[str, list[Any]]) -> list[dict[str, Any]]:
    n = len(history.get("train_loss", []))
    rows = []
    for idx in range(n):
        row = {
            "epoch": idx + 1,
            "train_loss": history.get("train_loss", [None] * n)[idx],
            "val_loss": history.get("val_loss", [None] * n)[idx],
            "val_surface_loss": history.get("val_surface_loss", [None] * n)[idx],
            "zero_baseline_loss": history.get("zero_baseline_loss", [None] * n)[idx],
            "surface_zero_baseline_loss": history.get("surface_zero_baseline_loss", [None] * n)[idx],
            "pred_mean": history.get("pred_mean", [None] * n)[idx],
            "pred_std": history.get("pred_std", [None] * n)[idx],
            "lr": history.get("lr", [None] * n)[idx],
            "epoch_time_s": history.get("epoch_time_s", [None] * n)[idx],
        }
        row["selection_loss"] = row["val_surface_loss"] if row["val_surface_loss"] and row["val_surface_loss"] > 0 else row["val_loss"]
        rows.append(row)
    return rows


def make_monitor_domain(domain: dict[str, Any], monitor_cfg: dict[str, Any]) -> dict[str, Any]:
    monitor_domain = deepcopy(domain)
    monitor_domain["grid_nx"] = int(monitor_cfg.get("coarse_grid_nx", domain["grid_nx"]))
    monitor_domain["grid_ny"] = int(monitor_cfg.get("coarse_grid_ny", domain["grid_ny"]))
    monitor_domain["grid_nz"] = int(monitor_cfg.get("coarse_grid_nz", domain["grid_nz"]))
    return monitor_domain

def geometry_score_from_monitor(record: dict[str, Any], domain: dict[str, Any]) -> float:
    iou = float(record.get("occupancy_iou", 0.0))
    precision = float(record.get("occupancy_precision", 0.0))
    recall = float(record.get("occupancy_recall", 0.0))
    boundary_fraction = float(record.get("boundary_fraction", 1.0))
    fragmentation_penalty = min(1.0, float(record.get("fragmentation_ratio", 1.0)))
    xy_extent = max(domain["x_max"] - domain["x_min"], domain["y_max"] - domain["y_min"], 1.0)
    chamfer_norm = min(1.0, float(record.get("chamfer_l1_m", xy_extent)) / xy_extent)
    return 1.80 * iou + 0.45 * precision + 0.20 * recall - 0.35 * boundary_fraction - 0.20 * fragmentation_penalty - 0.20 * chamfer_norm


def run_geometry_probe(model, device: str, domain: dict[str, Any], monitor_cfg: dict[str, Any], sign_helper: FootprintSignHelper | None, reference_mesh) -> dict[str, Any]:
    if sign_helper is None or not sign_helper.available or reference_mesh is None:
        return {"monitor_status": "skipped", "monitor_reason": "reference_geometry_unavailable"}

    probe_domain = make_monitor_domain(domain, monitor_cfg)
    raw_sdf_grid, pts_world_grid = predict_grid(model, device, probe_domain)
    reference_occ = sign_helper.contains(pts_world_grid.reshape(-1, 3))
    reference_voxels = int(reference_occ.sum())

    cleaned_occ, cleaning_info = clean_predicted_occupancy(raw_sdf_grid, reference_voxels)
    cleaned_sdf_grid = make_binary_field(cleaned_occ)
    occ = occupancy_metrics_from_grid(cleaned_sdf_grid, pts_world_grid, sign_helper)

    boundary_fraction = float(cleaning_info.get("boundary_component_count", 0)) / max(1, int(cleaning_info.get("n_components", 0)))
    grounded_components = int(cleaning_info.get("grounded_component_count", 0))
    kept_components = len(cleaning_info.get("kept_component_labels", []))
    fragmentation_ratio = max(0.0, (grounded_components - max(1, kept_components)) / max(1, grounded_components))

    record = {
        "monitor_status": "ok",
        "reference_obstacle_voxels_monitor": reference_voxels,
        "boundary_fraction": boundary_fraction,
        "fragmentation_ratio": fragmentation_ratio,
        **cleaning_info,
        **occ,
        "monitor_grid_nx": int(probe_domain["grid_nx"]),
        "monitor_grid_ny": int(probe_domain["grid_ny"]),
        "monitor_grid_nz": int(probe_domain["grid_nz"]),
    }

    try:
        pred_mesh = reconstruct_mesh_from_sdf(cleaned_sdf_grid, probe_domain)
        chamfer = symmetric_chamfer(reference_mesh, pred_mesh, n_samples=int(monitor_cfg.get("chamfer_samples", 8000)))
        record.update(chamfer)
        record["predicted_vertices"] = int(len(pred_mesh.vertices))
        record["predicted_faces"] = int(len(pred_mesh.faces))
    except Exception as exc:
        record["monitor_status"] = "failed"
        record["monitor_reason"] = f"mesh_reconstruction_failed: {exc}"
        record["predicted_vertices"] = 0
        record["predicted_faces"] = 0
        record["chamfer_l1_m"] = float("inf")
        record["chamfer_l2_m2"] = float("inf")
        record["hausdorff_approx_m"] = float("inf")

    record["geometry_score"] = geometry_score_from_monitor(record, probe_domain)
    return record


def build_summary(history: dict[str, list[Any]], epoch_records: list[dict[str, Any]], monitor_records: list[dict[str, Any]], best_epoch: int, best_selection_loss: float, best_geometry: dict[str, Any] | None, n_epochs_requested: int, validation_fraction: float, status: str, failure_info: dict[str, Any] | None = None) -> dict[str, Any]:
    best_surface_epoch = None
    best_surface_value = None
    if history.get("val_surface_loss"):
        valid = [(i, value) for i, value in enumerate(history["val_surface_loss"]) if value > 0]
        if valid:
            idx, value = min(valid, key=lambda item: item[1])
            best_surface_epoch = idx + 1
            best_surface_value = value

    best_val_epoch = None
    best_val_value = None
    if history.get("val_loss"):
        idx = int(np.argmin(history["val_loss"]))
        best_val_epoch = idx + 1
        best_val_value = history["val_loss"][idx]

    summary = {
        "status": status,
        "best_epoch_selection": best_epoch + 1 if best_epoch >= 0 else None,
        "best_selection_loss": best_selection_loss if math.isfinite(best_selection_loss) else None,
        "best_val_epoch": best_val_epoch,
        "best_val_loss": best_val_value,
        "best_surface_epoch": best_surface_epoch,
        "best_surface_loss": best_surface_value,
        "latest_epoch": epoch_records[-1]["epoch"] if epoch_records else 0,
        "latest_val_loss": epoch_records[-1]["val_loss"] if epoch_records else None,
        "latest_surface_loss": epoch_records[-1]["val_surface_loss"] if epoch_records else None,
        "latest_zero_baseline_loss": epoch_records[-1]["zero_baseline_loss"] if epoch_records else None,
        "n_epochs_requested": n_epochs_requested,
        "n_epochs_run": len(epoch_records),
        "validation_fraction": validation_fraction,
        "monitor_probes_run": len(monitor_records),
        "best_geometry": best_geometry,
    }
    if failure_info is not None:
        summary["failure"] = failure_info
    return summary


def build_checkpoint_recommendation(best_geometry: dict[str, Any] | None, monitor_cfg: dict[str, Any]) -> dict[str, Any]:
    min_iou = float(monitor_cfg.get("prefer_geometry_checkpoint_min_iou", 0.10))
    if best_geometry and float(best_geometry.get("occupancy_iou", 0.0)) >= min_iou:
        return {
            "recommended_checkpoint": "checkpoints/best_geometry.pt",
            "reason": "Le meilleur checkpoint géométrique est plus fiable pour la reconstruction finale.",
            "source": "geometry_monitor",
            "best_geometry": best_geometry,
        }
    return {
        "recommended_checkpoint": "checkpoints/best.pt",
        "reason": "Le meilleur checkpoint de validation reste la référence principale.",
        "source": "validation_selection",
        "best_geometry": best_geometry,
    }


def persist_structured_logs(ckpt_dir: str, history: dict[str, list[Any]], epoch_records: list[dict[str, Any]], monitor_records: list[dict[str, Any]], summary: dict[str, Any], recommendation: dict[str, Any], failure_info: dict[str, Any] | None = None) -> None:
    write_json(os.path.join(ckpt_dir, "history.json"), history)
    write_json(os.path.join(ckpt_dir, "training_epochs.json"), epoch_records)
    write_jsonl(os.path.join(ckpt_dir, "training_epochs.jsonl"), epoch_records)
    write_csv(os.path.join(ckpt_dir, "training_epochs.csv"), epoch_records)
    if monitor_records:
        write_json(os.path.join(ckpt_dir, "monitor_history.json"), monitor_records)
        write_jsonl(os.path.join(ckpt_dir, "monitor_history.jsonl"), monitor_records)
        write_csv(os.path.join(ckpt_dir, "monitor_history.csv"), monitor_records)
    write_json(os.path.join(ckpt_dir, "training_summary.json"), summary)
    write_json(os.path.join(ckpt_dir, "checkpoint_recommendation.json"), recommendation)
    if failure_info is not None:
        write_json(os.path.join(ckpt_dir, "training_failure.json"), failure_info)


def mirror_training_artifacts(mirror_root: str | None, ckpt_dir: str) -> None:
    if not mirror_root:
        return
    mirror_ckpt_dir = os.path.join(mirror_root, "checkpoints")
    os.makedirs(mirror_ckpt_dir, exist_ok=True)

    filenames = [
        "best.pt",
        "best_geometry.pt",
        "last.pt",
        "history.json",
        "training_epochs.json",
        "training_epochs.jsonl",
        "training_epochs.csv",
        "training_summary.json",
        "checkpoint_recommendation.json",
        "monitor_history.json",
        "monitor_history.jsonl",
        "monitor_history.csv",
        "training_failure.json",
    ]
    for name in filenames:
        src = os.path.join(ckpt_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(mirror_ckpt_dir, name))

    for path in sorted(os.listdir(ckpt_dir)):
        if path.startswith("checkpoint_epoch_") and path.endswith(".pt"):
            src = os.path.join(ckpt_dir, path)
            dst = os.path.join(mirror_ckpt_dir, path)
            if not os.path.isfile(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                shutil.copy2(src, dst)


def failure_payload(code: str, epoch: int, message: str, details: dict[str, Any], likely_causes: list[str]) -> dict[str, Any]:
    return {
        "status": "failed",
        "code": code,
        "epoch": epoch,
        "message": message,
        "details": details,
        "likely_causes": likely_causes,
    }


def detect_failure(epoch_records: list[dict[str, Any]], monitor_records: list[dict[str, Any]], monitor_cfg: dict[str, Any]) -> dict[str, Any] | None:
    if not epoch_records:
        return None

    latest = epoch_records[-1]
    epoch = int(latest["epoch"])
    for key in ("train_loss", "val_loss", "val_surface_loss", "pred_std"):
        value = latest.get(key)
        if value is None or not math.isfinite(float(value)):
            return failure_payload("non_finite_metric", epoch, f"La métrique {key} est invalide à l'époque {epoch}.", {"metric": key, "value": value}, ["Explosition numérique ou NaN.", "Gradient instable."])

    collapse_patience = int(monitor_cfg.get("collapse_patience", 5))
    min_pred_std = float(monitor_cfg.get("min_pred_std", 0.02))
    if len(epoch_records) >= collapse_patience:
        recent = epoch_records[-collapse_patience:]
        if all(float(row.get("pred_std", 0.0)) < min_pred_std for row in recent):
            return failure_payload("collapsed_predictor", epoch, "Les prédictions se sont effondrées vers une valeur presque constante.", {"recent_pred_std": [row.get("pred_std") for row in recent], "threshold": min_pred_std}, ["Retour du prédicteur trivial.", "Signal de surface insuffisant."])

    warmup_epochs = int(monitor_cfg.get("warmup_epochs", 20))
    max_surface_ratio = float(monitor_cfg.get("max_surface_ratio_to_baseline_after_warmup", 0.65))
    if epoch >= warmup_epochs:
        ratios = []
        for row in epoch_records:
            denom = float(row.get("surface_zero_baseline_loss", 0.0) or 0.0)
            numer = float(row.get("val_surface_loss", 0.0) or 0.0)
            if denom > 0:
                ratios.append(numer / denom)
        if ratios and min(ratios) > max_surface_ratio:
            return failure_payload("no_learning_surface_signal", epoch, "Le modèle ne dépasse pas suffisamment la baseline sur la bande de surface.", {"best_surface_ratio": min(ratios), "threshold": max_surface_ratio}, ["Solution triviale persistante.", "Sampling insuffisamment informatif."])

    fail_fast_after_epoch = int(monitor_cfg.get("fail_fast_after_epoch", 60))
    persistent_bad_probes = int(monitor_cfg.get("persistent_bad_probes", 2))
    if epoch < fail_fast_after_epoch:
        return None

    ok_recent = [row for row in monitor_records if row.get("monitor_status") == "ok"][-persistent_bad_probes:]
    if len(ok_recent) >= persistent_bad_probes:
        min_iou = float(monitor_cfg.get("min_iou_after_fail_fast", 0.04))
        min_precision = float(monitor_cfg.get("min_precision_after_fail_fast", 0.08))
        min_recall = float(monitor_cfg.get("min_recall_after_fail_fast", 0.08))
        max_boundary_fraction = float(monitor_cfg.get("max_boundary_fraction_after_fail_fast", 0.55))
        bad = []
        for probe in ok_recent:
            iou = float(probe.get("occupancy_iou", 0.0))
            precision = float(probe.get("occupancy_precision", 0.0))
            recall = float(probe.get("occupancy_recall", 0.0))
            boundary_fraction = float(probe.get("boundary_fraction", 1.0))
            if iou < min_iou and (precision < min_precision or recall < min_recall or boundary_fraction > max_boundary_fraction):
                bad.append(probe)
        if len(bad) == persistent_bad_probes:
            precision_mean = float(np.mean([row.get("occupancy_precision", 0.0) for row in bad]))
            recall_mean = float(np.mean([row.get("occupancy_recall", 0.0) for row in bad]))
            boundary_mean = float(np.mean([row.get("boundary_fraction", 1.0) for row in bad]))
            if precision_mean < min_precision and recall_mean > 0.40:
                code, message, causes = "geometry_overprediction", "La reconstruction détecte beaucoup trop de faux obstacles.", ["Champ SDF bruité.", "Nettoyage trop permissif."]
            elif recall_mean < min_recall and precision_mean > 0.40:
                code, message, causes = "geometry_under_reconstruction", "La reconstruction est trop incomplète.", ["Domaine trop large ou résolution trop faible.", "Seuil trop strict."]
            elif boundary_mean > max_boundary_fraction:
                code, message, causes = "geometry_boundary_artifacts", "La reconstruction touche trop les bords du domaine.", ["Domaine mal ajusté.", "Surface parasite sur les limites du volume."]
            else:
                code, message, causes = "geometry_quality_too_low", "La qualité géométrique reste trop faible de manière persistante.", ["La loss point par point ne se traduit pas en géométrie exploitable.", "Compromis précision/rappel insuffisant."]
            return failure_payload(code, epoch, message, {"recent_bad_probes": bad}, causes)
    return None

def train(config_path: str, resume: str | None = None) -> None:
    cfg = load_config(config_path)
    tcfg = cfg["training"]
    paths = cfg["paths"]
    monitor_cfg = cfg.get("monitoring", {})
    perf_cfg = cfg.get("performance", {})
    base_dir = os.path.dirname(os.path.abspath(config_path))

    dataset_path = os.path.join(base_dir, paths["sdf_dataset"], "sdf_dataset.pt")
    mesh_dir = os.path.join(base_dir, paths["meshes"])
    ckpt_dir = os.path.join(base_dir, paths["checkpoints"])
    os.makedirs(ckpt_dir, exist_ok=True)

    mirror_root = os.environ.get("DEEPSDF_MIRROR_DIR")
    mirror_every = int(monitor_cfg.get("mirror_every_n_epochs", 5))

    seed = int(tcfg.get("seed", 42))
    val_fraction = float(tcfg.get("validation_fraction", 0.02))
    num_workers = int(tcfg.get("num_workers", min(4, os.cpu_count() or 1)))
    grad_clip = tcfg.get("gradient_clip", 1.0)
    early_stopping_patience = int(tcfg.get("early_stopping_patience", 0))
    surface_zero_eps = float(tcfg.get("surface_zero_eps", 1e-5))
    surface_focus_band = float(tcfg.get("surface_focus_band", min(0.02, float(tcfg["clamp_delta"]) * 0.5)))

    seed_everything(seed)

    matmul_precision = str(perf_cfg.get("matmul_precision", "high"))
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            pass

    device = tcfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(perf_cfg.get("tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(perf_cfg.get("tf32", True))
        torch.backends.cudnn.benchmark = bool(perf_cfg.get("cudnn_benchmark", True))

    amp_enabled = bool(perf_cfg.get("amp", device == "cuda"))
    amp_dtype = str(perf_cfg.get("amp_dtype", "float16")).lower()
    if amp_dtype not in {"float16", "bfloat16"}:
        amp_dtype = "float16"
    scaler = None
    if device == "cuda" and amp_enabled and amp_dtype == "float16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    print(f"  AMP enabled : {amp_enabled and device == 'cuda'}")
    if device == "cuda":
        print(f"  AMP dtype   : {amp_dtype}")

    print(f"\nLoading dataset: {dataset_path}")
    data = torch.load(dataset_path, map_location="cpu")
    coords = data[:, :3]
    sdfs = data[:, 3:4]
    weights = build_sample_weights(
        sdfs,
        delta=float(tcfg["clamp_delta"]),
        surface_zero_eps=surface_zero_eps,
        surface_focus_band=surface_focus_band,
    )
    describe_sdf_distribution(sdfs, delta=float(tcfg["clamp_delta"]), surface_zero_eps=surface_zero_eps, surface_focus_band=surface_focus_band)

    full_dataset = TensorDataset(coords, sdfs, weights)
    print(f"  {len(full_dataset):,} points loaded")

    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_weights = weights[train_ds.indices]
    train_loader = make_loader(train_ds, tcfg["batch_size"], True, device, num_workers, sample_weights=train_weights)
    val_loader = make_loader(val_ds, tcfg["batch_size"], False, device, num_workers)
    print(f"  Train points: {len(train_ds):,}")
    print(f"  Val points  : {len(val_ds):,}")
    print(f"  Batch size  : {tcfg['batch_size']:,}")
    print(f"  Surface band: {surface_focus_band:.5f}")

    model = build_model(cfg).to(device)
    n_pars = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {n_pars:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
    scheduler_name = tcfg.get("lr_scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(tcfg["n_epochs"]), eta_min=1e-6)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    else:
        scheduler = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_surface_loss": [],
        "zero_baseline_loss": [],
        "surface_zero_baseline_loss": [],
        "pred_mean": [],
        "pred_std": [],
        "lr": [],
        "epoch_time_s": [],
    }
    epoch_records: list[dict[str, Any]] = []
    monitor_records: list[dict[str, Any]] = []
    best_selection_loss = float("inf")
    best_epoch = -1
    best_geometry: dict[str, Any] | None = None
    best_geometry_score = -float("inf")
    start_epoch = 0
    epochs_without_improvement = 0

    if resume and os.path.isfile(resume):
        print(f"\nResuming from {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_selection_loss = float(ckpt.get("best_selection_loss", ckpt.get("best_val_loss", ckpt.get("best_loss", float("inf")))))
        best_epoch = int(ckpt.get("best_epoch", -1))
        history = ckpt.get("history", history)
        epoch_records = ckpt.get("epoch_records", rebuild_epoch_records_from_history(history))
        monitor_records = ckpt.get("monitor_records", [])
        best_geometry = ckpt.get("best_geometry")
        if best_geometry is not None:
            best_geometry_score = float(best_geometry.get("geometry_score", -float("inf")))
        print(f"  Resume epoch     : {start_epoch}")
        print(f"  Best selection   : {best_selection_loss:.6f}")

    delta = float(tcfg["clamp_delta"])
    save_every = int(tcfg["save_every_n_epochs"])
    n_epochs = int(tcfg["n_epochs"])

    geometry_monitor_enabled = bool(monitor_cfg.get("enabled", True))
    monitor_every = int(monitor_cfg.get("every_n_epochs", 20))
    geometry_sign_helper = None
    reference_mesh = None
    if geometry_monitor_enabled:
        try:
            geometry_sign_helper = FootprintSignHelper(mesh_dir)
            meshes = load_meshes(mesh_dir)
            if meshes:
                reference_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        except Exception:
            geometry_sign_helper = None
            reference_mesh = None
            print("Warning: geometry monitor unavailable; reference geometry could not be loaded.")

    print("\n" + "=" * 76)
    print(f"Training for {n_epochs} epochs  |  clamp delta = {delta}")
    print("Primary selection metric: validation near-surface loss")
    if geometry_monitor_enabled:
        print(f"Geometry probe every {monitor_every} epochs")
    print("=" * 76)

    failure_info = None
    try:
        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            train_loss = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                delta,
                grad_clip,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                delta,
                surface_zero_eps=surface_zero_eps,
                surface_focus_band=surface_focus_band,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if scheduler is not None:
                scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_surface_loss"].append(val_metrics["val_surface_loss"])
            history["zero_baseline_loss"].append(val_metrics["zero_baseline_loss"])
            history["surface_zero_baseline_loss"].append(val_metrics["surface_zero_baseline_loss"])
            history["pred_mean"].append(val_metrics["pred_mean"])
            history["pred_std"].append(val_metrics["pred_std"])
            history["lr"].append(lr_now)
            history["epoch_time_s"].append(epoch_time)

            selection_loss = val_metrics["val_surface_loss"] if val_metrics["val_surface_loss"] > 0 else val_metrics["val_loss"]
            is_best = selection_loss < best_selection_loss
            if is_best:
                best_selection_loss = selection_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_surface_loss": val_metrics["val_surface_loss"],
                "zero_baseline_loss": val_metrics["zero_baseline_loss"],
                "surface_zero_baseline_loss": val_metrics["surface_zero_baseline_loss"],
                "pred_mean": val_metrics["pred_mean"],
                "pred_std": val_metrics["pred_std"],
                "lr": lr_now,
                "epoch_time_s": epoch_time,
                "selection_loss": selection_loss,
                "is_best_selection": bool(is_best),
            }
            epoch_records.append(row)

            print(
                f"Epoch {epoch+1:4d}/{n_epochs}  "
                f"train={train_loss:.6f}  "
                f"val={val_metrics['val_loss']:.6f}  "
                f"surface={val_metrics['val_surface_loss']:.6f}  "
                f"zero={val_metrics['zero_baseline_loss']:.6f}  "
                f"pred_mean={val_metrics['pred_mean']:.5f}  "
                f"pred_std={val_metrics['pred_std']:.5f}  "
                f"lr={lr_now:.2e}  "
                f"{'best' if is_best else ''}  "
                f"{epoch_time:.1f}s"
            )

            run_monitor = geometry_monitor_enabled and ((epoch + 1) >= int(monitor_cfg.get("warmup_epochs", 20)) and (((epoch + 1) % monitor_every) == 0 or (epoch + 1) == n_epochs))
            if run_monitor:
                print(f"  -> Geometry probe at epoch {epoch + 1}")
                probe = run_geometry_probe(model, device, cfg["domain"], monitor_cfg, geometry_sign_helper, reference_mesh)
                probe["epoch"] = epoch + 1
                monitor_records.append(probe)
                if probe.get("monitor_status") == "ok":
                    print(f"     IoU={probe['occupancy_iou']:.4f}  P={probe['occupancy_precision']:.4f}  R={probe['occupancy_recall']:.4f}  boundary={probe['boundary_fraction']:.3f}  score={probe['geometry_score']:.4f}")
                    if probe["geometry_score"] > best_geometry_score:
                        best_geometry_score = float(probe["geometry_score"])
                        best_geometry = probe
                else:
                    print(f"     Monitor failed: {probe.get('monitor_reason', 'unknown reason')}")

            summary = build_summary(history, epoch_records, monitor_records, best_epoch, best_selection_loss, best_geometry, n_epochs, val_fraction, status="running")
            recommendation = build_checkpoint_recommendation(best_geometry, monitor_cfg)

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_selection_loss": best_selection_loss,
                "best_val_loss": val_metrics["val_loss"],
                "best_epoch": best_epoch,
                "history": history,
                "epoch_records": epoch_records,
                "monitor_records": monitor_records,
                "best_geometry": best_geometry,
                "config": cfg,
            }
            if scheduler is not None:
                ckpt["scheduler_state"] = scheduler.state_dict()

            if (epoch + 1) % save_every == 0:
                torch.save(ckpt, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1:04d}.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            if best_geometry is not None and best_geometry.get("epoch") == epoch + 1:
                torch.save(ckpt, os.path.join(ckpt_dir, "best_geometry.pt"))
            torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))

            failure_info = detect_failure(epoch_records, monitor_records, monitor_cfg)
            persist_structured_logs(ckpt_dir, history, epoch_records, monitor_records, summary, recommendation, failure_info=failure_info)

            should_mirror = mirror_root and (is_best or (epoch + 1) % max(1, mirror_every) == 0 or (epoch + 1) % save_every == 0 or failure_info is not None)
            if should_mirror:
                mirror_training_artifacts(mirror_root, ckpt_dir)

            if failure_info is not None:
                raise RuntimeError(failure_info["message"])

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epochs_without_improvement} epochs without validation improvement.")
                break

        summary = build_summary(history, epoch_records, monitor_records, best_epoch, best_selection_loss, best_geometry, n_epochs, val_fraction, status="completed")
        recommendation = build_checkpoint_recommendation(best_geometry, monitor_cfg)
        persist_structured_logs(ckpt_dir, history, epoch_records, monitor_records, summary, recommendation)
        mirror_training_artifacts(mirror_root, ckpt_dir)

        print(f"\nTraining complete. Best epoch: {summary['best_epoch_selection']}")
        print(f"Best selection loss: {best_selection_loss:.6f}")
        if best_geometry is not None:
            print(f"Best geometry probe: epoch {best_geometry['epoch']} | IoU={best_geometry.get('occupancy_iou', 0.0):.4f} | score={best_geometry.get('geometry_score', 0.0):.4f}")
        print(f"Recommended checkpoint: {recommendation['recommended_checkpoint']}")
        print(f"Best model saved to: {os.path.join(ckpt_dir, 'best.pt')}")

    except Exception as exc:
        if failure_info is None:
            failure_info = failure_payload("unexpected_exception", epoch_records[-1]["epoch"] if epoch_records else 0, f"Crash inattendu dans train.py: {exc}", {"traceback": traceback.format_exc()}, ["Erreur non prévue pendant l'entraînement ou le monitoring.", "Fichier intermédiaire absent ou dépendance cassée."])

        summary = build_summary(history, epoch_records, monitor_records, best_epoch, best_selection_loss, best_geometry, n_epochs, val_fraction, status="failed", failure_info=failure_info)
        recommendation = build_checkpoint_recommendation(best_geometry, monitor_cfg)
        persist_structured_logs(ckpt_dir, history, epoch_records, monitor_records, summary, recommendation, failure_info=failure_info)
        mirror_training_artifacts(mirror_root, ckpt_dir)
        print("\nTraining stopped with a diagnostic failure.")
        print(json.dumps(failure_info, indent=2, default=json_default))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSDF training with structured monitoring")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()
    train(args.config, resume=args.resume)
