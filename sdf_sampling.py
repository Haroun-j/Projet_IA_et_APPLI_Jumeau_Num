"""
Échantillonnage des points SDF
============================================================
Charge des maillages 3D extrudés, échantillonne trois catégories de points,
calcule le champ de distance signée (SDF) pour chacun, normalise les
coordonnées et sauvegarde le jeu d'entraînement sous forme d'un tenseur
PyTorch (N, 4) : [x_norm, y_norm, z_norm, sdf].

Convention : SDF > 0  → extérieur (air)
             SDF = 0  → sur la surface
             SDF < 0  → intérieur (bâtiment)
"""

import argparse
import json
import os

import numpy as np
import torch
import trimesh
import yaml
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_meshes(mesh_dir: str) -> list:
    meshes = []
    for fname in sorted(os.listdir(mesh_dir)):
        if not (fname.endswith(".obj") or fname.endswith(".ply")):
            continue
        path = os.path.join(mesh_dir, fname)
        try:
            m = trimesh.load(path, force="mesh", process=False)
            if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0:
                meshes.append(m)
                print(f"  Loaded {fname}: {len(m.vertices):,} verts / {len(m.faces):,} faces")
        except Exception as exc:
            print(f"  WARNING: could not load {fname}: {exc}")
    return meshes


class FootprintSignHelper:
    """
    Reconstruit intérieur/extérieur à partir des empreintes 2D et des hauteurs
    d'extrusion.

    C'est plus fiable que d'appeler mesh.contains() sur la scène aéroportuaire
    concaténée, qui n'est généralement pas étanche une fois que de nombreux
    bâtiments sont fusionnés.
    """

    def __init__(self, mesh_dir: str):
        meta_path = os.path.join(mesh_dir, "buildings_2d.json")
        self.available = os.path.isfile(meta_path)
        self._tree = None
        self._geoms = []
        self._heights = np.zeros(0, dtype=np.float32)

        if not self.available:
            return

        with open(meta_path, "r") as f:
            data = json.load(f)

        geoms = []
        heights = []
        for item in data.get("buildings", []):
            coords = item.get("coords", [])
            h = float(item.get("height", 0.0))
            try:
                poly = Polygon(coords)
            except Exception:
                continue
            if poly.is_empty or not poly.is_valid or poly.area <= 0.0:
                continue
            geoms.append(poly)
            heights.append(h)

        if geoms:
            self._geoms = geoms
            self._heights = np.array(heights, dtype=np.float32)
            self._tree = STRtree(geoms)
            self.available = True
        else:
            self.available = False

    def contains(self, points_xyz: np.ndarray) -> np.ndarray:
        if not self.available or self._tree is None:
            return np.zeros(len(points_xyz), dtype=bool)

        inside = np.zeros(len(points_xyz), dtype=bool)
        # Premier filtre rapide sur z. Les bâtiments sont extrudés à partir de z=0 vers le haut.
        valid_idx = np.where(points_xyz[:, 2] >= 0.0)[0]
        if len(valid_idx) == 0:
            return inside

        try:
            import shapely

            pts = shapely.points(points_xyz[valid_idx, 0], points_xyz[valid_idx, 1])
            pairs = self._tree.query(pts, predicate="within")
            if len(pairs) == 2 and len(pairs[0]) > 0:
                pt_idx, poly_idx = pairs
                z_vals = points_xyz[valid_idx[pt_idx], 2]
                inside_hits = z_vals <= self._heights[poly_idx]
                inside[valid_idx[pt_idx[inside_hits]]] = True
                return inside
        except Exception:
            pass

        # Solution de repli prudente pour les anciennes versions de Shapely.
        for global_idx in valid_idx:
            px, py, pz = points_xyz[global_idx]
            point = Point(float(px), float(py))
            for poly_idx in self._tree.query(point):
                if point.within(self._geoms[poly_idx]) and pz <= self._heights[poly_idx]:
                    inside[global_idx] = True
                    break
        return inside


def compute_sdf_batch(mesh: trimesh.Trimesh, points: np.ndarray,
                      sign_helper: FootprintSignHelper | None = None,
                      chunk: int = 8_000) -> np.ndarray:
    results = []
    for i in tqdm(range(0, len(points), chunk), desc="  SDF", leave=False):
        batch = points[i : i + chunk]
        _, dist, _ = trimesh.proximity.closest_point(mesh, batch)
        if sign_helper is not None and sign_helper.available:
            inside = sign_helper.contains(batch)
        else:
            inside = mesh.contains(batch)
        sdf = np.where(inside, -dist.astype(np.float32), dist.astype(np.float32))
        results.append(sdf)
    return np.concatenate(results)


# ─────────────────────────────────────────────────────────────────────────────

def run_sampling(config_path: str):
    cfg = load_config(config_path)
    paths    = cfg["paths"]
    sampling = cfg["sampling"]
    domain   = cfg["domain"]

    base_dir    = os.path.dirname(os.path.abspath(config_path))
    mesh_dir    = os.path.join(base_dir, paths["meshes"])
    dataset_dir = os.path.join(base_dir, paths["sdf_dataset"])
    os.makedirs(dataset_dir, exist_ok=True)

    # ── Charge et fusionne les maillages ─────────────────────────────────────
    print("=== Loading meshes ===")
    meshes = load_meshes(mesh_dir)
    if not meshes:
        raise RuntimeError(f"No meshes in {mesh_dir}. Run data_ingestion.py first.")

    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    print(f"  Combined: {len(mesh.vertices):,} verts / {len(mesh.faces):,} faces")
    print(f"  Watertight: {mesh.is_watertight}  (False is normal for a multi-building scene)")
    sign_helper = FootprintSignHelper(mesh_dir)
    if sign_helper.available:
        print(f"  2D footprint metadata loaded for {len(sign_helper._geoms):,} buildings")
    else:
        print("  WARNING: buildings_2d.json missing -> falling back to mesh.contains()")

    # ── Paramètres de normalisation ──────────────────────────────────────────
    cx = (domain["x_min"] + domain["x_max"]) / 2.0
    cy = (domain["y_min"] + domain["y_max"]) / 2.0
    cz = domain["z_max"] / 2.0
    center = np.array([cx, cy, cz], dtype=np.float32)
    scale  = max(
        domain["x_max"] - domain["x_min"],
        domain["y_max"] - domain["y_min"],
        domain["z_max"],
    ) / 2.0

    norm_params = {"center": center.tolist(), "scale": float(scale)}
    with open(os.path.join(dataset_dir, "normalization.json"), "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"  Normalisation: center={center.tolist()}, scale={scale:.2f} m")

    rng = np.random.default_rng(seed=42)
    all_pts, all_sdf = [], []

    # ── Type 1 : points de surface (SDF ≈ 0) — 30 % ─────────────────────────
    n_surf = sampling["n_surface_points"]
    print(f"\n=== Sampling {n_surf:,} surface points ===")
    surf_pts, _ = trimesh.sample.sample_surface(mesh, n_surf)
    surf_pts = surf_pts.astype(np.float32)

    # Conserve uniquement les points dans le domaine de simulation. Des bâtiments téléchargés
    # avec un rayon plus grand que le domaine (radius_m > demi-largeur du domaine) produiraient
    # des points de surface avec x_norm > 1, ce qui perturberait les Fourier features.
    in_domain = (
        (surf_pts[:, 0] >= domain["x_min"]) & (surf_pts[:, 0] <= domain["x_max"]) &
        (surf_pts[:, 1] >= domain["y_min"]) & (surf_pts[:, 1] <= domain["y_max"]) &
        (surf_pts[:, 2] >= 0.0)             & (surf_pts[:, 2] <= domain["z_max"])
    )
    surf_pts = surf_pts[in_domain]
    surf_sdf = np.zeros(len(surf_pts), dtype=np.float32)
    print(f"  {len(surf_pts):,} surface points kept within domain "
          f"({in_domain.sum()/n_surf*100:.1f}%)")
    all_pts.append(surf_pts)
    all_sdf.append(surf_sdf)

    # ── Type 2 : points proches de la surface (faible bruit gaussien) — 50 % ─
    n_near = sampling["n_near_surface_points"]
    sigma  = float(sampling["near_surface_sigma"]) * scale
    print(f"\n=== Sampling {n_near:,} near-surface points (σ={sigma:.3f} m) ===")
    base_pts, _ = trimesh.sample.sample_surface(mesh, n_near)
    noise = rng.standard_normal(base_pts.shape).astype(np.float32) * sigma
    near_pts = (base_pts + noise).astype(np.float32)
    # Tronque à la boîte englobante du domaine
    near_pts[:, 0] = np.clip(near_pts[:, 0], domain["x_min"], domain["x_max"])
    near_pts[:, 1] = np.clip(near_pts[:, 1], domain["y_min"], domain["y_max"])
    near_pts[:, 2] = np.clip(near_pts[:, 2], 0.0, domain["z_max"])
    print("  Computing SDF for near-surface points…")
    near_sdf = compute_sdf_batch(mesh, near_pts, sign_helper=sign_helper)
    all_pts.append(near_pts)
    all_sdf.append(near_sdf)

    # ── Type 3 : points aléatoires dans le domaine — 20 % ───────────────────
    n_far = sampling["n_far_points"]
    print(f"\n=== Sampling {n_far:,} random domain points ===")
    far_pts = rng.uniform(
        low =[domain["x_min"], domain["y_min"], 0.0],
        high=[domain["x_max"], domain["y_max"], domain["z_max"]],
        size=(n_far, 3),
    ).astype(np.float32)
    print("  Computing SDF for far points…")
    far_sdf = compute_sdf_batch(mesh, far_pts, sign_helper=sign_helper)
    all_pts.append(far_pts)
    all_sdf.append(far_sdf)

    # ── Assemble le jeu de données ───────────────────────────────────────────
    all_pts = np.concatenate(all_pts, axis=0)
    all_sdf = np.concatenate(all_sdf, axis=0)

    # Tronque la SDF dans [-δ, δ]
    delta = float(sampling["clamp_distance"])
    all_sdf = np.clip(all_sdf, -delta, delta)

    # Normalise les coordonnées
    norm_pts = (all_pts - center) / scale

    # Construit le tenseur (N, 4)
    dataset = np.concatenate([norm_pts, all_sdf[:, np.newaxis]], axis=1)
    tensor  = torch.from_numpy(dataset)

    # Mélange
    perm   = torch.randperm(len(tensor))
    tensor = tensor[perm]

    out_path = os.path.join(dataset_dir, "sdf_dataset.pt")
    torch.save(tensor, out_path)

    print(f"\n=== Dataset saved ===")
    print(f"  Shape  : {tuple(tensor.shape)}")
    print(f"  Path   : {out_path}")
    print(f"  Coords : [{norm_pts.min():.3f}, {norm_pts.max():.3f}]  (normalised)")
    print(f"  SDF    : [{all_sdf.min():.4f}, {all_sdf.max():.4f}]")
    inside_frac = (all_sdf < 0).mean() * 100
    print(f"  Inside : {inside_frac:.1f}% of points")
    abs_sdf = np.abs(all_sdf)
    print(f"  |SDF| mean       : {abs_sdf.mean():.4f}")
    print(f"  |SDF| median     : {np.median(abs_sdf):.4f}")
    print(f"  Exact surface    : {(abs_sdf <= 1e-6).mean() * 100:.1f}%")
    print(f"  Near surface <0.5: {(abs_sdf <= 0.5).mean() * 100:.1f}%")
    print(f"  Near surface <1.0: {(abs_sdf <= 1.0).mean() * 100:.1f}%")
    print(f"  Clamped to delta : {(abs_sdf >= delta * 0.99).mean() * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 2: SDF point sampling")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_sampling(args.config)
