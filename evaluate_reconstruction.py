"""
Quantitative validation for the airport DeepSDF pipeline.

Outputs:
- reconstructed_mesh.obj
- reconstruction_metrics.json
- reconstruction_comparison.png
- reconstruction_cleaning.json
- cleaned_occupancy_grid.npy
- occupancy_confusion_matrix.json
- occupancy_confusion_matrix.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure
import trimesh
import yaml

from deepsdf_model import build_model
from sdf_sampling import FootprintSignHelper, load_meshes


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, cfg: dict, device: str):
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_grid(model, device: str, domain: dict, batch: int = 16384):
    nx = int(domain["grid_nx"])
    ny = int(domain["grid_ny"])
    nz = int(domain["grid_nz"])

    xs = np.linspace(domain["x_min"], domain["x_max"], nx, dtype=np.float32)
    ys = np.linspace(domain["y_min"], domain["y_max"], ny, dtype=np.float32)
    zs = np.linspace(0.0, domain["z_max"], nz, dtype=np.float32)

    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    pts_world = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    cx = (domain["x_min"] + domain["x_max"]) / 2.0
    cy = (domain["y_min"] + domain["y_max"]) / 2.0
    cz = domain["z_max"] / 2.0
    scale = max(
        domain["x_max"] - domain["x_min"],
        domain["y_max"] - domain["y_min"],
        domain["z_max"],
    ) / 2.0
    center = np.array([cx, cy, cz], dtype=np.float32)
    pts_norm = ((pts_world - center) / scale).astype(np.float32)

    preds = []
    pts_t = torch.from_numpy(pts_norm)
    for i in range(0, len(pts_t), batch):
        pred = model(pts_t[i:i + batch].to(device)).squeeze(-1).cpu().numpy()
        preds.append(pred)
    sdf_flat = np.concatenate(preds)
    sdf_grid = sdf_flat.reshape(nz, ny, nx)
    return sdf_grid, pts_world.reshape(nz, ny, nx, 3)


def make_binary_field(occupancy_grid: np.ndarray) -> np.ndarray:
    return np.where(occupancy_grid, -1.0, 1.0).astype(np.float32)


def choose_occupancy_level(raw_sdf_grid: np.ndarray, reference_voxels: int) -> tuple[float, dict]:
    neg_values = np.sort(raw_sdf_grid[raw_sdf_grid < 0.0].reshape(-1))
    if neg_values.size == 0:
        return 0.0, {
            "occupancy_level": 0.0,
            "occupancy_level_source": "no_negative_values",
            "target_obstacle_voxels": int(reference_voxels),
            "predicted_obstacle_voxels_at_level": 0,
        }

    target_voxels = int(max(1, round(reference_voxels * 1.10)))
    target_voxels = min(target_voxels, int(neg_values.size))
    occupancy_level = float(neg_values[target_voxels - 1])
    occupancy_level = min(occupancy_level, -1e-6)
    predicted_voxels = int((raw_sdf_grid < occupancy_level).sum())

    return occupancy_level, {
        "occupancy_level": occupancy_level,
        "occupancy_level_source": "matched_reference_voxel_count",
        "target_obstacle_voxels": int(reference_voxels),
        "predicted_obstacle_voxels_at_level": predicted_voxels,
    }


def clean_predicted_occupancy(raw_sdf_grid: np.ndarray, reference_voxels: int) -> tuple[np.ndarray, dict]:
    occupancy_level, level_info = choose_occupancy_level(raw_sdf_grid, reference_voxels)
    raw_occ_base = raw_sdf_grid < 0.0
    raw_occ = raw_sdf_grid < occupancy_level
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, n_components = ndimage.label(raw_occ, structure=structure)

    if n_components == 0:
        return raw_occ, {
            **level_info,
            "raw_negative_fraction": float(raw_occ_base.mean()),
            "thresholded_negative_fraction": float(raw_occ.mean()),
            "cleaned_negative_fraction": float(raw_occ.mean()),
            "n_components": 0,
            "kept_component_labels": [],
            "kept_component_sizes": [],
            "grounded_component_count": 0,
            "fallback_used": True,
        }

    component_ids = np.arange(1, n_components + 1)
    sizes = ndimage.sum(raw_occ, labels, component_ids).astype(np.int64)

    boundary_labels = set(np.unique(labels[:, 0, :]).tolist())
    boundary_labels.update(np.unique(labels[:, -1, :]).tolist())
    boundary_labels.update(np.unique(labels[:, :, 0]).tolist())
    boundary_labels.update(np.unique(labels[:, :, -1]).tolist())
    boundary_labels.update(np.unique(labels[-1, :, :]).tolist())
    boundary_labels.discard(0)

    min_component_voxels = max(16, int(reference_voxels * 0.004))
    ground_touch_slices = max(1, raw_occ.shape[0] // 24)

    candidates = []
    for cid, size in zip(component_ids, sizes):
        cid = int(cid)
        size = int(size)
        if cid in boundary_labels or size < min_component_voxels:
            continue
        component_coords = np.argwhere(labels == cid)
        z_min = int(component_coords[:, 0].min())
        z_max = int(component_coords[:, 0].max())
        if z_min > ground_touch_slices:
            continue
        candidates.append(
            {
                "label": cid,
                "size": size,
                "z_min": z_min,
                "z_max": z_max,
                "z_span": z_max - z_min + 1,
            }
        )

    candidates.sort(key=lambda item: (item["size"], item["z_span"]), reverse=True)

    target_keep_voxels = int(max(1, round(reference_voxels * 1.15)))
    keep_ids = []
    cumulative_voxels = 0
    for item in candidates:
        keep_ids.append(item["label"])
        cumulative_voxels += item["size"]
        if cumulative_voxels >= target_keep_voxels:
            break

    fallback_used = False
    if not keep_ids:
        fallback_used = True
        keep_ids = [int(component_ids[int(np.argmax(sizes))])]

    cleaned_occ = np.isin(labels, keep_ids)
    cleaned_occ = ndimage.binary_closing(cleaned_occ, structure=np.ones((1, 3, 3), dtype=bool), iterations=1)
    cleaned_occ = ndimage.binary_opening(cleaned_occ, structure=np.ones((1, 2, 2), dtype=bool), iterations=1)

    kept_sizes = [int(sizes[int(cid) - 1]) for cid in keep_ids]
    cleaning_info = {
        **level_info,
        "raw_negative_fraction": float(raw_occ_base.mean()),
        "thresholded_negative_fraction": float(raw_occ.mean()),
        "cleaned_negative_fraction": float(cleaned_occ.mean()),
        "n_components": int(n_components),
        "boundary_component_count": int(sum(1 for cid in component_ids if int(cid) in boundary_labels)),
        "grounded_component_count": int(len(candidates)),
        "ground_touch_slices": int(ground_touch_slices),
        "min_component_voxels": int(min_component_voxels),
        "kept_component_labels": keep_ids,
        "kept_component_sizes": kept_sizes,
        "target_keep_voxels": int(target_keep_voxels),
        "kept_total_voxels_before_morphology": int(sum(kept_sizes)),
        "fallback_used": fallback_used,
    }
    return cleaned_occ, cleaning_info


def reconstruct_mesh_from_sdf(sdf_grid: np.ndarray, domain: dict):
    try:
        verts_zyx, faces, _normals, _values = measure.marching_cubes(sdf_grid, level=0.0)
    except Exception as exc:
        raise RuntimeError(f"Marching cubes failed: {exc}") from exc

    nx = int(domain["grid_nx"])
    ny = int(domain["grid_ny"])
    nz = int(domain["grid_nz"])
    dx = (domain["x_max"] - domain["x_min"]) / (nx - 1)
    dy = (domain["y_max"] - domain["y_min"]) / (ny - 1)
    dz = domain["z_max"] / (nz - 1)

    verts_xyz = np.column_stack(
        [
            domain["x_min"] + verts_zyx[:, 2] * dx,
            domain["y_min"] + verts_zyx[:, 1] * dy,
            verts_zyx[:, 0] * dz,
        ]
    ).astype(np.float32)

    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces.astype(np.int32), process=True)
    return mesh


def symmetric_chamfer(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_samples: int = 50000):
    pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples)
    pts_b, _ = trimesh.sample.sample_surface(mesh_b, n_samples)

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a_to_b = tree_b.query(pts_a, k=1)[0]
    dist_b_to_a = tree_a.query(pts_b, k=1)[0]

    return {
        "chamfer_l1_m": float(dist_a_to_b.mean() + dist_b_to_a.mean()),
        "chamfer_l2_m2": float((dist_a_to_b ** 2).mean() + (dist_b_to_a ** 2).mean()),
        "hausdorff_approx_m": float(max(dist_a_to_b.max(), dist_b_to_a.max())),
    }


def occupancy_metrics_from_grid(sdf_grid: np.ndarray, pts_world_grid: np.ndarray, sign_helper: FootprintSignHelper):
    pts_flat = pts_world_grid.reshape(-1, 3)
    pred_occ = (sdf_grid.reshape(-1) < 0.0)
    ref_occ = sign_helper.contains(pts_flat)

    tp = int(np.logical_and(pred_occ, ref_occ).sum())
    tn = int(np.logical_and(~pred_occ, ~ref_occ).sum())
    fp = int(np.logical_and(pred_occ, ~ref_occ).sum())
    fn = int(np.logical_and(~pred_occ, ref_occ).sum())

    inter = tp
    union = int(np.logical_or(pred_occ, ref_occ).sum())
    pred_sum = int(pred_occ.sum())
    ref_sum = int(ref_occ.sum())

    return {
        "occupancy_iou": float(inter / union) if union > 0 else 0.0,
        "occupancy_precision": float(inter / pred_sum) if pred_sum > 0 else 0.0,
        "occupancy_recall": float(inter / ref_sum) if ref_sum > 0 else 0.0,
        "occupancy_tp": tp,
        "occupancy_tn": tn,
        "occupancy_fp": fp,
        "occupancy_fn": fn,
        "predicted_obstacle_voxels": pred_sum,
        "reference_obstacle_voxels": ref_sum,
    }


def save_occupancy_confusion_figure(metrics: dict, out_png: str, out_json: str) -> None:
    matrix = np.array(
        [
            [metrics["occupancy_tn"], metrics["occupancy_fp"]],
            [metrics["occupancy_fn"], metrics["occupancy_tp"]],
        ],
        dtype=np.int64,
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Prédit libre", "Prédit obstacle"])
    ax.set_yticks([0, 1], labels=["Référence libre", "Référence obstacle"])
    ax.set_title("Matrice d'occupation voxelisée")

    total = max(int(matrix.sum()), 1)
    for i in range(2):
        for j in range(2):
            value = int(matrix[i, j])
            pct = 100.0 * value / total
            ax.text(j, i, f"{value}\n{pct:.1f}%", ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    payload = {
        "matrix": matrix.tolist(),
        "labels": {
            "x": ["pred_free", "pred_obstacle"],
            "y": ["ref_free", "ref_obstacle"],
        },
        "accuracy": float((metrics["occupancy_tp"] + metrics["occupancy_tn"]) / max(total, 1)),
        "precision": float(metrics["occupancy_precision"]),
        "recall": float(metrics["occupancy_recall"]),
        "iou": float(metrics["occupancy_iou"]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_comparison_figure(sdf_grid: np.ndarray, pts_world_grid: np.ndarray, sign_helper: FootprintSignHelper, out_path: str):
    nz = sdf_grid.shape[0]
    k_values = [max(1, nz // 8), max(1, nz // 4), max(1, nz // 2)]

    fig, axes = plt.subplots(len(k_values), 2, figsize=(10, 11))
    if len(k_values) == 1:
        axes = np.array([axes])

    for row, k in enumerate(k_values):
        world_slice = pts_world_grid[k].reshape(-1, 3)
        ref_occ = sign_helper.contains(world_slice).reshape(sdf_grid.shape[1], sdf_grid.shape[2])
        pred_occ = (sdf_grid[k] < 0.0)

        ax0, ax1 = axes[row]
        ax0.imshow(ref_occ, origin="lower", cmap="Greys")
        ax0.set_title(f"Reference z={k}")
        ax0.axis("off")

        ax1.imshow(pred_occ, origin="lower", cmap="Greys")
        ax1.set_title(f"Prediction z={k}")
        ax1.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_evaluation(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    paths = cfg["paths"]
    domain = cfg["domain"]
    base_dir = os.path.dirname(os.path.abspath(config_path))
    mesh_dir = os.path.join(base_dir, paths["meshes"])
    out_dir = os.path.join(base_dir, paths["outputs"])
    os.makedirs(out_dir, exist_ok=True)

    meshes = load_meshes(mesh_dir)
    if not meshes:
        raise RuntimeError("No reference meshes found. Run data_ingestion.py first.")
    ref_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    sign_helper = FootprintSignHelper(mesh_dir)
    if not sign_helper.available:
        raise RuntimeError("buildings_2d.json missing. Re-run data_ingestion.py before evaluation.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_model(checkpoint_path, cfg, device)

    print("Evaluating SDF on validation grid...")
    raw_sdf_grid, pts_world_grid = predict_grid(model, device, domain)

    reference_occ = sign_helper.contains(pts_world_grid.reshape(-1, 3))
    reference_voxels = int(reference_occ.sum())

    print("Cleaning predicted occupancy...")
    cleaned_occ, cleaning_info = clean_predicted_occupancy(raw_sdf_grid, reference_voxels)
    cleaned_sdf_grid = make_binary_field(cleaned_occ)
    np.save(os.path.join(out_dir, "cleaned_occupancy_grid.npy"), cleaned_occ.astype(np.uint8))

    print("Reconstructing cleaned mesh with marching cubes...")
    pred_mesh = reconstruct_mesh_from_sdf(cleaned_sdf_grid, domain)
    mesh_path = os.path.join(out_dir, "reconstructed_mesh.obj")
    pred_mesh.export(mesh_path)

    print("Computing Chamfer-style reconstruction metrics...")
    chamfer = symmetric_chamfer(ref_mesh, pred_mesh, n_samples=50000)

    print("Computing occupancy metrics on the simulation grid...")
    occ = occupancy_metrics_from_grid(cleaned_sdf_grid, pts_world_grid, sign_helper)

    comparison_path = os.path.join(out_dir, "reconstruction_comparison.png")
    save_comparison_figure(cleaned_sdf_grid, pts_world_grid, sign_helper, comparison_path)
    confusion_json = os.path.join(out_dir, "occupancy_confusion_matrix.json")
    confusion_png = os.path.join(out_dir, "occupancy_confusion_matrix.png")
    save_occupancy_confusion_figure(occ, confusion_png, confusion_json)

    metrics = {
        "reference_vertices": int(len(ref_mesh.vertices)),
        "reference_faces": int(len(ref_mesh.faces)),
        "predicted_vertices": int(len(pred_mesh.vertices)),
        "predicted_faces": int(len(pred_mesh.faces)),
        **chamfer,
        **occ,
        **cleaning_info,
    }

    metrics_path = os.path.join(out_dir, "reconstruction_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cleaning_path = os.path.join(out_dir, "reconstruction_cleaning.json")
    with open(cleaning_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_info, f, indent=2)

    print("\nReconstruction metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nPredicted mesh saved to: {mesh_path}")
    print(f"Metrics saved to      : {metrics_path}")
    print(f"Comparison figure     : {comparison_path}")
    print(f"Confusion matrix      : {confusion_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction validation for DeepSDF")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(args.config))
    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(base, ckpt)
    run_evaluation(args.config, ckpt)
