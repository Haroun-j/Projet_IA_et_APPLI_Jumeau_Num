"""
losses.py — Loss composite Physics-Informed pour la dispersion atmosphérique de NOx.

Composantes :
    - MSE pondérée spatialement (near-wall)
    - Conservation de masse (voxels fluides)
    - Herméticité des obstacles (voxels solides)
    - Total Variation 3D anisotrope (lissage physique)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Résolution physique de la grille (en mètres par voxel)
# Utilisées pour la pondération anisotrope de la TV loss
# ─────────────────────────────────────────────────────────────────────────────
_DZ: float = 3.0   # axe vertical
_DY: float = 5.0   # axe horizontal Y
_DX: float = 5.0   # axe horizontal X


def boundary_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    geom: torch.Tensor,
    boundary_band: float = 5.0,
    boundary_weight: float = 5.0,
) -> torch.Tensor:
    """
    MSE pondérée spatialement — amplifie la précision près des parois.

    Formule unifiée SDF / masque binaire :
        w(x) = 1 + (w_max − 1) · exp(−|g(x)| / boundary_band)

    Pour g = SDF normalisée :
        |g(x)| ≈ 0 près des parois  → w ≈ w_max  (poids maximal)
        |g(x)| grand loin des parois → w ≈ 1      (poids unitaire)

    Pour g = masque binaire ∈ {0, 1} :
        g = 0 (solide) → w = w_max
        g = 1 (fluide) → w ≈ 1 si boundary_band > 1

    Cette formule est identique pour les deux encodages géométriques,
    ce qui garantit une comparaison équitable entre voxelisation et SDF.

    Arguments :
        pred            : (B, 1, Nz, Ny, Nx) float32 — prédiction normalisée
        target          : (B, 1, Nz, Ny, Nx) float32 — cible normalisée
        geom            : (B, 1, Nz, Ny, Nx) float32 — premier canal géométrique (SDF ou masque)
        boundary_band   : échelle de décroissance exponentielle [mêmes unités que geom]
        boundary_weight : poids maximal appliqué près des parois (w_max)

    Retourne :
        Scalaire float32 — MSE pondérée moyennée sur le batch.
    """
    pred_f   = pred.float()
    target_f = target.float()
    geom_f   = geom.float()

    # Calcul du poids spatial : fort près des parois, unitaire au loin
    w = 1.0 + (boundary_weight - 1.0) * torch.exp(-geom_f.abs() / boundary_band)

    return (w * (pred_f - target_f).pow(2)).mean()


def mass_conservation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    geom: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conservation de masse et herméticité des obstacles.

    Formule unifiée SDF / masque binaire pour les masques fluide/solide :
        masque_fluide = σ(g · 20)      (sigmoïde appliquée à la valeur géométrique)

    Pour g = SDF normalisée :
        g > 0 → fluide    : σ(+) → 1
        g < 0 → solide    : σ(−) → 0
        g = 0 → frontière : σ(0) = 0.5

    Pour g = masque binaire ∈ {0, 1} :
        g = 0 (solide) → σ(0)  = 0.5
        g = 1 (fluide) → σ(20) ≈ 1.0

    Remarque : pour le masque binaire, g = 0 donne σ(0) = 0.5 plutôt que 0,
    ce qui introduit un léger biais identique pour les deux variantes.

    Stabilisations numériques :
        1. Dénominateur n_fluid — évite une explosion si la masse cible tend vers 0.
        2. Clamp de pred dans [0, NORM_MAX = 2] — borne la loss en début d'entraînement
           pour les modèles sans activation de sortie (la MSE continue de corriger
           les valeurs hors de cette plage grâce à son gradient non nul).

    Valeurs attendues en régime stable :
        l_mass_balance  ∈ [0, 0.3]
        l_solid_penalty ∈ [0, 0.1]

    Arguments :
        pred   : (B, 1, Nz, Ny, Nx) float32 — prédiction normalisée
        target : (B, 1, Nz, Ny, Nx) float32 — cible normalisée
        geom   : (B, 1, Nz, Ny, Nx) float32 — premier canal géométrique (SDF ou masque)

    Retourne :
        (l_mass_balance, l_solid_penalty) — deux scalaires float32
    """
    pred_f   = pred.float()
    target_f = target.float()
    geom_f   = geom.float()

    # Masques fluide et solide unifiés pour SDF et masque binaire
    fluid_mask = torch.sigmoid(geom_f * 20.0)
    solid_mask = 1.0 - fluid_mask

    # Borne défensive sur les prédictions pour éviter les explosions numériques
    # en début d'entraînement (la MSE continue de corriger via son gradient)
    NORM_MAX  = 2.0
    pred_pos  = pred_f.clamp(min=0.0, max=NORM_MAX)
    target_pos = target_f.clamp(min=0.0)

    # Masse totale intégrée sur les voxels fluides
    mass_pred   = (pred_pos * fluid_mask).sum(dim=(-3, -2, -1))   # (B,)
    mass_target = (target_pos * fluid_mask).sum(dim=(-3, -2, -1)) # (B,)

    # Erreur de conservation de masse normalisée par le nombre de voxels fluides
    n_fluid         = fluid_mask.sum(dim=(-3, -2, -1)).clamp(min=1.0)  # (B,)
    l_mass_balance  = ((mass_pred - mass_target).abs() / n_fluid).mean()

    # Pénalité d'herméticité : la concentration dans les solides doit être nulle
    l_solid_penalty = (pred_pos * solid_mask).mean()

    return l_mass_balance, l_solid_penalty


def tv_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Total Variation Loss 3D anisotrope — lissage spatial physique.

    Pénalise les variations spatiales abruptes du champ de concentration prédit.

    Justification physique :
        Un solveur Lagrangien (Thomson 1987) génère des panaches continus par
        accumulation de particules sur une grille discrète. Le champ de concentration
        réel est gouverné par la diffusion turbulente et la convection par vent moyen,
        deux processus qui produisent des champs spatialement lisses. Les discontinuités
        dans la prédiction (artefacts spectraux FNO, bruit de grille) sont non physiques
        et doivent être pénalisées.

    Formule anisotrope pondérée par la résolution physique :
        L_TV = mean( |ΔC_z| / DZ + |ΔC_y| / DY + |ΔC_x| / DX )

    La pondération 1/DZ > 1/DY = 1/DX (DZ = 3m < DX = 5m) amplifie la pénalité
    verticale, ce qui est cohérent avec les forts gradients de concentration
    observés dans la couche limite atmosphérique basse (< 50 m).

    Arguments :
        pred : (B, 1, Nz, Ny, Nx) float32 — prédiction normalisée

    Retourne :
        Scalaire float32 — TV loss moyennée sur le batch.
    """
    pred_f = pred.float()

    # Différences finies du premier ordre sur chaque axe spatial
    dz_diff = (pred_f[:, :, 1:, :, :] - pred_f[:, :, :-1, :, :]).abs()  # Nz-1 différences
    dy_diff = (pred_f[:, :, :, 1:, :] - pred_f[:, :, :, :-1, :]).abs()  # Ny-1 différences
    dx_diff = (pred_f[:, :, :, :, 1:] - pred_f[:, :, :, :, :-1]).abs()  # Nx-1 différences

    # Somme anisotrope pondérée par la résolution physique de chaque axe
    tv = dz_diff.mean() / _DZ + dy_diff.mean() / _DY + dx_diff.mean() / _DX

    return tv


def dispersion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    geom: torch.Tensor,
    lambda_mass: float = 0.1,
    lambda_tv: float = 0.01,
    boundary_band: float = 5.0,
    boundary_weight: float = 5.0,
) -> dict[str, torch.Tensor]:
    """
    Loss composite Physics-Informed pour la dispersion atmosphérique de NOx.

    Formule :
        L_total = L_mse + λ_mass · (L_mass + L_solid) + λ_tv · L_TV

    Correspondance entre chaque composante et la physique du solveur :
        L_mse   ← MSE sur le champ accumulé par le solveur Lagrangien,
                   avec pondération accrue près des parois (forts gradients)
        L_mass  ← conservation de la masse totale des émissions LTO / GSE
        L_solid ← condition limite de non-pénétration (parois imperméables)
        L_TV    ← diffusion turbulente continue (Thomson 1987 / VDI 3783)

    Arguments :
        pred            : (B, 1, Nz, Ny, Nx) float32 — prédiction normalisée
        target          : (B, 1, Nz, Ny, Nx) float32 — cible normalisée
        geom            : (B, 1, Nz, Ny, Nx) float32 — premier canal géométrique (SDF ou masque)
        lambda_mass     : poids de la loss de conservation de masse (défaut : 0.1)
        lambda_tv       : poids de la Total Variation loss (défaut : 0.01)
        boundary_band   : échelle de décroissance de la pondération near-wall
        boundary_weight : poids maximal appliqué près des parois

    Retourne :
        Dictionnaire contenant :
            "loss"        : loss totale (à utiliser pour backward())
            "loss_mse"    : MSE pondérée spatialement
            "loss_mass"   : conservation de masse dans les voxels fluides
            "loss_solid"  : pénalité d'herméticité des obstacles
            "loss_tv"     : Total Variation (lissage spatial)
    """
    l_mse                   = boundary_weighted_mse(pred, target, geom, boundary_band, boundary_weight)
    l_mass_balance, l_solid = mass_conservation_loss(pred, target, geom)
    l_tv                    = tv_loss(pred)

    total_loss = l_mse + lambda_mass * (l_mass_balance + l_solid) + lambda_tv * l_tv

    return {
        "loss":       total_loss,
        "loss_mse":   l_mse,
        "loss_mass":  l_mass_balance,
        "loss_solid": l_solid,
        "loss_tv":    l_tv,
    }
