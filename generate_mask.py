"""
 Génération du obstacle_mask pour wind.py
=======================================================================
Évalue le modèle DeepSDF entraîné sur une grille 3D régulière et construit le
masque binaire d'obstacles attendu par wind.py / solver.py :

    mask = 1  →  air  (SDF ≥ 0)
    mask = 0  →  à l'intérieur d'un bâtiment  (SDF < 0)

Forme du tenseur de sortie : (1, 1, nz, ny, nx)  ←  le format requis par wind.py.

Utilisation :
    python generate_mask.py --config config.yaml --checkpoint checkpoints/best.pt
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from deepsdf_model import build_model


# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_mask(model, device: str, domain: dict,
                  norm_center: np.ndarray, norm_scale: float,
                  surface_level: float = 0.0,
                  batch: int = 16_384):
    """
    Évalue le modèle sur la grille (nz, ny, nx) et renvoie :
        mask_tensor  (1, 1, nz, ny, nx)  float32  — 1=air, 0=obstacle
        sdf_grid     (nz, ny, nx)         float32  — valeurs brutes de SDF
        dx, dy, dz   pas de grille en mètres
    """
    nx = domain["grid_nx"]
    ny = domain["grid_ny"]
    nz = domain["grid_nz"]

    xs = np.linspace(domain["x_min"], domain["x_max"], nx, dtype=np.float32)
    ys = np.linspace(domain["y_min"], domain["y_max"], ny, dtype=np.float32)
    zs = np.linspace(0.0, domain["z_max"],             nz, dtype=np.float32)

    # Construit la grille aplatie (nz*ny*nx, 3)  — indexation (k, j, i) = (z, y, x)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    pts_world = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Normalise
    pts_norm = ((pts_world - norm_center) / norm_scale).astype(np.float32)
    pts_t    = torch.from_numpy(pts_norm)

    print(f"  Grid: {nz}×{ny}×{nx} = {nz*ny*nx:,} cells")
    sdf_vals = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pts_t), batch):
            s = model(pts_t[i : i + batch].to(device)).squeeze(-1).cpu().numpy()
            sdf_vals.append(s)
    sdf_flat  = np.concatenate(sdf_vals)
    sdf_grid  = sdf_flat.reshape(nz, ny, nx)

    # 1 = air,  0 = obstacle
    mask_np = (sdf_grid >= surface_level).astype(np.float32)
    mask_t  = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1,1,nz,ny,nx)

    dx = (domain["x_max"] - domain["x_min"]) / (nx - 1)
    dy = (domain["y_max"] - domain["y_min"]) / (ny - 1)
    dz = domain["z_max"] / (nz - 1)

    n_obs   = int((mask_np < 0.5).sum())
    n_total = int(mask_np.size)
    print(f"  Obstacle cells: {n_obs:,} / {n_total:,}  ({100*n_obs/n_total:.1f}%)")
    print(f"  Grid spacing  : dx={dx:.2f} m, dy={dy:.2f} m, dz={dz:.2f} m")

    return mask_t, sdf_grid, dx, dy, dz


def visualize_mask_slice(sdf_grid: np.ndarray, domain: dict, out_dir: str):
    """Sauvegarde quelques coupes horizontales du masque d'obstacles en PNG."""
    nz, ny, nx = sdf_grid.shape
    xs = np.linspace(domain["x_min"], domain["x_max"], nx)
    ys = np.linspace(domain["y_min"], domain["y_max"], ny)

    for k, frac in [(2, "z5pct"), (nz // 4, "z25pct"), (nz // 2, "z50pct")]:
        mask_slice = (sdf_grid[k] < 0.0).astype(float)  # 1 = dans le bâtiment
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.pcolormesh(xs, ys, mask_slice, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Empreinte des obstacles — couche k={k}/{nz}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        plt.tight_layout()
        path = os.path.join(out_dir, f"mask_slice_{frac}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Mask slice → {path}")


# ─────────────────────────────────────────────────────────────────────────────

def run_generate_mask(config_path: str, checkpoint_path: str,
                      output_path: str | None = None):
    cfg      = load_config(config_path)
    paths    = cfg["paths"]
    domain   = cfg["domain"]
    surface_level = float(cfg.get("mask", {}).get("surface_level", 0.0))
    prefer_cleaned = bool(cfg.get("mask", {}).get("prefer_cleaned_reconstruction", True))
    base_dir = os.path.dirname(os.path.abspath(config_path))
    out_dir  = os.path.join(base_dir, paths["outputs"])
    os.makedirs(out_dir, exist_ok=True)

    # ── Paramètres de normalisation ──────────────────────────────────────────
    norm_path = os.path.join(base_dir, paths["sdf_dataset"], "normalization.json")
    with open(norm_path) as f:
        norm = json.load(f)
    norm_center = np.array(norm["center"], dtype=np.float32)
    norm_scale  = float(norm["scale"])

    # ── Charge le modèle ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = build_model(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ── Génère le masque ─────────────────────────────────────────────────────
    print("\n=== Generating obstacle mask ===")
    cleaned_occ_path = os.path.join(out_dir, "cleaned_occupancy_grid.npy")
    if prefer_cleaned and os.path.isfile(cleaned_occ_path):
        cleaned_occ = np.load(cleaned_occ_path)
        if cleaned_occ.shape == (domain["grid_nz"], domain["grid_ny"], domain["grid_nx"]):
            print("  Using cleaned occupancy grid from evaluation.")
            sdf_grid = np.where(cleaned_occ.astype(bool), -1.0, 1.0).astype(np.float32)
            mask_np = (~cleaned_occ.astype(bool)).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
            dx = (domain["x_max"] - domain["x_min"]) / (domain["grid_nx"] - 1)
            dy = (domain["y_max"] - domain["y_min"]) / (domain["grid_ny"] - 1)
            dz = domain["z_max"] / (domain["grid_nz"] - 1)
            n_obs = int(cleaned_occ.sum())
            n_total = int(cleaned_occ.size)
            print(f"  Obstacle cells: {n_obs:,} / {n_total:,}  ({100*n_obs/n_total:.1f}%)")
            print(f"  Grid spacing  : dx={dx:.2f} m, dy={dy:.2f} m, dz={dz:.2f} m")
        else:
            print("  Cleaned occupancy grid found but shape does not match domain. Falling back to model evaluation.")
            mask_t, sdf_grid, dx, dy, dz = generate_mask(
                model, device, domain, norm_center, norm_scale, surface_level=surface_level
            )
    else:
        mask_t, sdf_grid, dx, dy, dz = generate_mask(
            model, device, domain, norm_center, norm_scale, surface_level=surface_level
        )

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    if output_path is None:
        output_path = os.path.join(out_dir, "obstacle_mask.pt")
    torch.save(mask_t, output_path)
    print(f"  Mask saved → {output_path}  shape={tuple(mask_t.shape)}")

    sdf_path = os.path.join(out_dir, "sdf_grid.npy")
    np.save(sdf_path, sdf_grid)
    print(f"  SDF grid saved → {sdf_path}")

    meta_path = os.path.join(out_dir, "obstacle_mask_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "surface_level": surface_level,
                "shape": list(mask_t.shape),
                "dx_m": dx,
                "dy_m": dy,
                "dz_m": dz,
            },
            f,
            indent=2,
        )
    print(f"  Mask metadata saved → {meta_path}")

    # ── Visualise ────────────────────────────────────────────────────────────
    visualize_mask_slice(sdf_grid, domain, out_dir)

    # ── Affiche l'extrait d'intégration ──────────────────────────────────────
    print("\n" + "="*60)
    print("COPIER-COLLER : comment l'utiliser avec wind.py")
    print("="*60)
    print(f"""
import sys, torch
sys.path.insert(0, r"{os.path.join(base_dir, 'prof_files')}")
from wind import MassConsistentWindSolverNumerical

mask = torch.load(r"{output_path}").cuda()        # (1,1,{domain['grid_nz']},{domain['grid_ny']},{domain['grid_nx']})

dx, dy, dz = {dx:.2f}, {dy:.2f}, {dz:.2f}       # mètres
solver = MassConsistentWindSolverNumerical(dx=dx, dy=dy, dz=dz, device="cuda")

# Exemple : vent d'ouest à 5 m/s
u0 = torch.full_like(mask, 5.0)
v0 = torch.zeros_like(mask)
w0 = torch.zeros_like(mask)

u, v, w, sigma, K = solver.adjust_wind_field(u0, v0, w0, mask)
print("Forme du champ de vent :", u.shape)   # (1,1,{domain['grid_nz']},{domain['grid_ny']},{domain['grid_nx']})
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 6: Generate obstacle mask")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output",     default=None, help="Override output path for mask .pt")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(args.config))
    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(base, ckpt)

    run_generate_mask(args.config, ckpt, args.output)
