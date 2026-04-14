"""
Visualization for the airport DeepSDF pipeline.

Outputs:
- render_Front_X.png
- render_Front_Y.png
- render_Diagonal.png
- render_Top-down.png
- renders_grid.png
- sdf_slice_z005.png
- sdf_slice_z020.png
- sdf_slice_z050.png
- training_loss.png
- monitoring_curves.png
- flyover.mp4
"""

import argparse
import json
import os

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import yaml

from deepsdf_model import build_model


SKY_RGB = np.array([135, 206, 235], dtype=np.uint8)


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
def eval_model_batch(model, pts_t: torch.Tensor, device: str, chunk: int = 8192) -> np.ndarray:
    out = []
    for i in range(0, len(pts_t), chunk):
        out.append(model(pts_t[i:i + chunk].to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def render_sdf_slice(model, device: str, z_norm: float, n: int = 128, output_path: str | None = None):
    lin = np.linspace(-1.0, -1.0 + 2.0, n)
    xx, yy = np.meshgrid(lin, lin)
    zz = np.full_like(xx, z_norm)
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    sdf = eval_model_batch(model, torch.from_numpy(pts), device)
    grid = sdf.reshape(n, n)

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(lin, lin, grid, levels=64, cmap="RdBu_r")
    ax.contour(lin, lin, grid, levels=[0.0], colors="k", linewidths=1.5)
    plt.colorbar(cf, ax=ax, label="SDF")
    ax.set_title(f"SDF slice z_norm={z_norm:.2f}")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()
    return grid


def plot_loss(ckpt_dir: str, output_dir: str):
    history_path = os.path.join(ckpt_dir, "history.json")
    if not os.path.isfile(history_path):
        print("No history.json found - skipping loss plot.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(epochs, history.get("train_loss", []), label="train", linewidth=2)
    if history.get("val_loss"):
        axes[0, 0].plot(epochs, history.get("val_loss", []), label="val", linewidth=2)
    axes[0, 0].set_title("Training and validation loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    if history.get("val_surface_loss"):
        axes[0, 1].plot(epochs, history.get("val_surface_loss", []), label="surface", linewidth=2)
    if history.get("zero_baseline_loss"):
        axes[0, 1].plot(epochs, history.get("zero_baseline_loss", []), label="zero baseline", linewidth=2)
    axes[0, 1].set_title("Surface validation signal")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    if history.get("pred_mean"):
        axes[1, 0].plot(epochs, history.get("pred_mean", []), label="pred_mean", linewidth=2)
    if history.get("pred_std"):
        axes[1, 0].plot(epochs, history.get("pred_std", []), label="pred_std", linewidth=2)
    axes[1, 0].set_title("Prediction statistics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    if history.get("lr"):
        axes[1, 1].plot(epochs, history.get("lr", []), label="lr", linewidth=2)
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "training_loss.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Loss curves -> {out}")


def plot_monitoring(ckpt_dir: str, output_dir: str):
    monitor_path = os.path.join(ckpt_dir, "monitor_history.json")
    if not os.path.isfile(monitor_path):
        print("No monitor_history.json found - skipping monitoring plot.")
        return

    with open(monitor_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    ok_rows = [row for row in history if row.get("monitor_status") == "ok"]
    if not ok_rows:
        print("Monitoring history exists but no successful probe was recorded.")
        return

    epochs = np.array([row["epoch"] for row in ok_rows], dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(epochs, [row.get("occupancy_iou", 0.0) for row in ok_rows], label="IoU", linewidth=2)
    axes[0, 0].plot(epochs, [row.get("occupancy_precision", 0.0) for row in ok_rows], label="precision", linewidth=2)
    axes[0, 0].plot(epochs, [row.get("occupancy_recall", 0.0) for row in ok_rows], label="recall", linewidth=2)
    axes[0, 0].set_title("Monitoring géométrique")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [row.get("geometry_score", 0.0) for row in ok_rows], label="geometry_score", linewidth=2)
    axes[0, 1].plot(epochs, [row.get("boundary_fraction", 0.0) for row in ok_rows], label="boundary_fraction", linewidth=2)
    axes[0, 1].plot(epochs, [row.get("fragmentation_ratio", 0.0) for row in ok_rows], label="fragmentation_ratio", linewidth=2)
    axes[0, 1].set_title("Score et structure")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row.get("chamfer_l1_m", np.nan) for row in ok_rows], label="chamfer_l1_m", linewidth=2)
    axes[1, 0].plot(epochs, [row.get("hausdorff_approx_m", np.nan) for row in ok_rows], label="hausdorff_approx_m", linewidth=2)
    axes[1, 0].set_title("Distances de reconstruction")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [row.get("predicted_obstacle_voxels", 0) for row in ok_rows], label="pred voxels", linewidth=2)
    axes[1, 1].plot(epochs, [row.get("reference_obstacle_voxels", 0) for row in ok_rows], label="ref voxels", linewidth=2)
    axes[1, 1].set_title("Volume d'obstacles")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "monitoring_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Monitoring curves -> {out}")


def load_render_mesh(out_dir: str):
    mesh_path = os.path.join(out_dir, "reconstructed_mesh.obj")
    if not os.path.isfile(mesh_path):
        return None
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def simplify_mesh_for_render(mesh: trimesh.Trimesh, max_faces: int = 12000):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) > max_faces:
        step = max(1, len(faces) // max_faces)
        faces = faces[::step][:max_faces]
    return vertices, faces


def render_mesh_frame(mesh: trimesh.Trimesh, elev: float, azim: float, title: str | None = None, size=(6, 6)) -> np.ndarray:
    vertices, faces = simplify_mesh_for_render(mesh)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(tuple(SKY_RGB / 255.0))
    ax.set_facecolor(tuple(SKY_RGB / 255.0))

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color=(0.55, 0.65, 0.80),
        linewidth=0.0,
        antialiased=True,
        shade=True,
        alpha=1.0,
    )

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half_size = max(float(np.ptp(vertices[:, 0])), float(np.ptp(vertices[:, 1])), float(np.ptp(vertices[:, 2]))) * 0.55
    half_size = max(half_size, 1.0)

    ax.set_xlim(center[0] - half_size, center[0] + half_size)
    ax.set_ylim(center[1] - half_size, center[1] + half_size)
    ax.set_zlim(max(0.0, center[2] - half_size * 0.3), center[2] + half_size * 0.7)
    ax.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1e-6))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    plt.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return frame


def save_mesh_views(mesh: trimesh.Trimesh, out_dir: str):
    viewpoints = [
        {"elev": 20, "azim": -90, "label": "Front X"},
        {"elev": 20, "azim": 0, "label": "Front Y"},
        {"elev": 30, "azim": -45, "label": "Diagonal"},
        {"elev": 85, "azim": -90, "label": "Top-down"},
    ]

    images = []
    for vp in viewpoints:
        print(f"Rendering mesh view: {vp['label']}...")
        frame = render_mesh_frame(mesh, elev=vp["elev"], azim=vp["azim"], title=vp["label"])
        images.append((frame, vp["label"]))
        plt.imsave(os.path.join(out_dir, f"render_{vp['label'].replace(' ', '_')}.png"), frame)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (img, label) in zip(axes.flat, images):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    plt.suptitle("Airport reconstruction views", fontsize=13)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, "renders_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"Render grid -> {grid_path}")


def save_flyover_video(mesh: trimesh.Trimesh, out_dir: str):
    video_path = os.path.join(out_dir, "flyover.mp4")
    frames = []
    n_frames = 72
    print("Generating flyover video...")
    for i, azim in enumerate(np.linspace(-180.0, 180.0, n_frames, endpoint=False)):
        frame = render_mesh_frame(mesh, elev=28.0, azim=float(azim), size=(6, 6))
        frames.append(frame)
        if (i + 1) % 12 == 0:
            print(f"  {i+1}/{n_frames} frames done")

    writer = iio.get_writer(video_path, fps=24, codec="libx264", output_params=["-crf", "23"])
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Flyover video -> {video_path}")


def run_visualization(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    paths = cfg["paths"]
    base_dir = os.path.dirname(os.path.abspath(config_path))
    out_dir = os.path.join(base_dir, paths["outputs"])
    ckpt_dir = os.path.join(base_dir, paths["checkpoints"])
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(checkpoint_path, cfg, device)
    print(f"Model loaded from {checkpoint_path}")

    mesh = load_render_mesh(out_dir)
    if mesh is None:
        raise FileNotFoundError("reconstructed_mesh.obj not found. Run evaluate_reconstruction.py first.")

    save_mesh_views(mesh, out_dir)

    for z_norm in [0.05, 0.20, 0.50]:
        z_tag = f"z{int(z_norm * 100):03d}"
        spath = os.path.join(out_dir, f"sdf_slice_{z_tag}.png")
        render_sdf_slice(model, device, z_norm=z_norm, n=128, output_path=spath)
        print(f"SDF slice z={z_norm:.2f} -> {spath}")

    plot_loss(ckpt_dir, out_dir)
    plot_monitoring(ckpt_dir, out_dir)
    save_flyover_video(mesh, out_dir)

    print("\nVisualization complete")
    print(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization for DeepSDF")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(args.config))
    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(base, ckpt)

    run_visualization(args.config, ckpt)
