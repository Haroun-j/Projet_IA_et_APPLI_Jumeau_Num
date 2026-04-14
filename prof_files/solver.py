"""
solver.py — Lagrangian Dispersion Engine (LASAT 3.4 / AUSTAL 3.3 / VDI 3945 Blatt 3)
=======================================================================================
Canonical references
────────────────────
  [LASAT]   LASAT 3.4 Reference Book (Janicke Consulting, 2018)
            Section 5.3.1 "The dispersion algorithm", Eq. 5.3.164–5.3.206
  [AUSTAL]  AUSTAL 3.3, Programmbeschreibung (2024-03-22), §3.4 (PLURIS)
  [VDI]     VDI 3945 Blatt 3 — Partikelmodell

MARKOV ALGORITHM (LASAT 5.3.1)
────────────────────────────────
LASAT implements a discrete Markov process (Eq. 5.3.164–5.3.165) :

    u_new = Ψ · u_old + W + Λ · R               (5.3.164)
    x_new = x_old + τ · [V_eff + u_new]         (5.3.165)

with :
    V_eff = 0.5·[V(x̂) + V(x̂ + τ·V(x̂))]       (5.3.166)  — effective 3D wind
    w     = W + Λ·R                              (5.3.167)  — stochastic velocity
    Λ·Λᵀ = Ω                                    (5.3.168)  — Cholesky decomposition

This is NOT the differential Langevin form (da = a·dt + b·dW), but the
exact (finite) discretization valid for τ up to 2·T_L (Eq. 5.3.204-205).

TENSOR COMPUTATION (LASAT 5.3.1, Eq. 5.3.185–5.3.202)
────────────────────────────────────────────────────────
The velocity covariance tensor in the wind frame (x̂ = downwind) :

    Σ = | σ_u²   0    −u★²  |    (5.3.185) — term −u★² = Σ_uw (shear)
        |  0    σ_v²   0    |
        | −u★²   0    σ_w²  |

    K = diag(K_u, K_v, K_w)       (5.3.186)

The v component is independent (off-diagonal zero in the u/w plane) :

    Φ_vv = σ_v²/K_v = 1/T_Lv
    p_v  = τ/2 · Φ_vv
    Ψ_vv = (1−p_v)/(1+p_v)                     (5.3.188)
    Ω_vv = 4·σ_v²·p_v·(1+p_v)⁻²               (5.3.190)

In the u/w plane (with coupling ρ = −u★²) :

    ρ_u = Σ_uw/σ_u²,  ρ_w = Σ_uw/σ_w²,  q = ρ_u·ρ_w
    p_u = τ/2·Φ_uu,   p_w = τ/2·Φ_ww
    Δ₊  = (1+p_u)(1+p_w) − q·p_u·p_w          (5.3.196)

    Ψ_uu = [(1−p_u)(1+p_w) + q·p_u·p_w] / Δ₊  (5.3.197)
    Ψ_ww = [(1+p_u)(1−p_w) + q·p_u·p_w] / Δ₊
    Ψ_uw = −2·p_w·ρ_w / Δ₊
    Ψ_wu = −2·p_u·ρ_u / Δ₊

    Ω_uu = 4·σ_u²·{q·p_w + p_u·[1+(1−q)·p_w]²} / Δ₊²   (5.3.198)
    Ω_ww = 4·σ_w²·{q·p_u + p_w·[1+(1−q)·p_u]²} / Δ₊²   (5.3.199)
    Ω_uw = 4·(−u★²)·[p_u+p_w+2(1−q)·p_u·p_w] / Δ₊²      (5.3.200)

DRIFT W (LASAT 5.3.1, Eq. 5.3.176 — spatially constant τ)
────────────────────────────────────────────────────────────
    W = (τ/2)·(I + Ψ) · (∇·Θ)    with  Θ = Σ  (5.3.176, 5.3.179)

In 3D diagonal + u/w coupling, only the z component is non-trivial :
    W_z = (τ/2) · (1 + Ψ_ww) · ∂σ_w²/∂z
         + (τ/2) · Ψ_wu       · ∂σ_u²/∂z
    W_u = (τ/2) · Ψ_uw        · ∂σ_w²/∂z
         + (τ/2) · (1 + Ψ_uu)  · ∂σ_u²/∂z

EFFECTIVE WIND Veff (LASAT 5.3.1, Eq. 5.3.166)
────────────────────────────────────────────────
For an inhomogeneous 3D wind field, the effective wind over step τ is :
    V_eff = 0.5·[V(x̂) + V(x̂ + τ·V(x̂))]

CONSTRAINT ON τ (LASAT 5.3.2, Eq. 5.3.204-205)
─────────────────────────────────────────────────
For Ω to remain positive definite : p_u ≤ 1 and p_w ≤ 1
    ⟺  τ ≤ 2·T_Lu  and  τ ≤ 2·T_Lw
The effective timestep is clamped to min(dt, 2·T_Lu, 2·T_Lw, 2·T_Lv).

ADDITIONAL WAKE TURBULENCE (AUSTAL 3.3, Annex D.2.3)
──────────────────────────────────────────────────────
The σ̂ and K̂ fields produced by wind.py are added linearly :
    σ²_tot = σ²_bg + σ̂²       (additive variances)
    K_tot  = K_bg  + K̂        (additive diffusivities)
The effective T_L is recomputed : T_L_eff = K_tot / σ²_tot.

The Σ and K tensors are extended accordingly before computing Ψ/Ω/W.
Note : σ̂ is isotropic (AUSTAL D.34 : σ̂_{u,v,w} identical for all three components)
and spatially homogeneous in the wake zone → its gradient is zero → no
contribution to W.

PLUME (AUSTAL 3.3, §3.4 + LASAT 3.4 §3.4.2.9)
────────────────────────────────────────────────
AUSTAL §3.4 : "Die fahneninduzierte Anfangsturbulenz wird intern auf 10% der
resultierenden, effektiven Austrittsgeschwindigkeit gesetzt."
The parameter vs_exit [m/s] (initial exit velocity) decays with a
characteristic time ts_plume [s]. The initial plume turbulence is σ_plume = 0.1·vs_exit.
This turbulence is added quadratically to the total variances.
The rise Δz = 0.5·(vs + vs_new)·dt (trapezoidal) is added to the vertical displacement.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class AustalLagrangianSolver:
    """
    Lagrangian dispersion engine compliant with LASAT 3.4 / AUSTAL 3.3.

    Implements the discrete Markov scheme (VDI 3945 Blatt 3 / LASAT 5.3.1) :
        u_new = Ψ·u_old + W + Λ·R
        x_new = x_old + τ·(V_eff + u_new)

    The tensors Ψ, Ω, W are computed exactly from (σ, K, τ)
    according to Eq. 5.3.185–5.3.206 of LASAT 3.4.
    """

    def __init__(
        self,
        num_particles: int,
        domain_bounds: Dict[str, float],
        profile,
        dmk_fields: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        obstacle_mask: torch.Tensor,
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        num_particles  : total number of particles (active + inactive)
        domain_bounds  : dict with x_min, x_max, y_min, y_max, z_max [m]
        profile        : AustalAtmosphericProfile instance (turbulence.py)
        dmk_fields     : tuple (u, v, w, σ̂, K̂) each with shape (1,1,nz,ny,nx)
                         from wind.py (MassConsistentWindSolverNumerical)
        obstacle_mask  : (1,1,nz,ny,nx) — 1=air, 0=obstacle
        device         : 'cuda' or 'cpu'
        """
        self.num_particles = num_particles
        self.bounds = domain_bounds
        self.profile = profile
        self.device = device
        self.obstacle_mask = obstacle_mask

        # ?? Champs DMK empil?s pour l'interpolation GPU (5 canaux) ???????????
        # Canal 0-2 : vent moyen (u, v, w) [m/s]
        # Canal 3   : ?? (turbulence de sillage) [m/s]   ? AUSTAL D.34
        # Canal 4   : K? (diffusivit? de sillage)  [m?/s]  ? AUSTAL D.35
        u_dmk, v_dmk, w_dmk, wake_turb_dmk, wake_K_dmk = dmk_fields
        self.dmk_5d = torch.cat(
            [u_dmk, v_dmk, w_dmk, wake_turb_dmk, wake_K_dmk], dim=1
        )  # (1, 5, nz, ny, nx)

        _, _, self.nz, self.ny, self.nx = self.dmk_5d.shape
        self.dx = (domain_bounds["x_max"] - domain_bounds["x_min"]) / self.nx
        self.dy = (domain_bounds["y_max"] - domain_bounds["y_min"]) / self.ny
        self.dz = domain_bounds["z_max"] / self.nz

        # ?? ?tat des particules ???????????????????????????????????????????????
        self.pos = torch.zeros((num_particles, 3), device=device, dtype=torch.float32)
        self.vel_turb = torch.zeros((num_particles, 3), device=device, dtype=torch.float32)
        self.is_alive = torch.zeros(num_particles, device=device, dtype=torch.bool)
        self.mass = torch.zeros(num_particles, device=device, dtype=torch.float32)

        # Panache : vitesse initiale de sortie [m/s] et temps caract?ristique [s]
        # AUSTAL ?3.4 : dq, vq, tq ? ? d?finition du diam?tre de la source, de la
        # vitesse de sortie et de la temp?rature de sortie ?
        self.vs_exit = torch.zeros(num_particles, device=device, dtype=torch.float32)
        self.ts_plume = torch.zeros(num_particles, device=device, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def _normalize_coords(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Physical coordinates [m] → normalized [−1, 1] for F.grid_sample.
        PyTorch 5D convention : grid[..., 0] → x axis (W), [1] → y (H), [2] → z (D).
        """
        norm = torch.empty_like(pos)
        Lx = self.bounds["x_max"] - self.bounds["x_min"]
        Ly = self.bounds["y_max"] - self.bounds["y_min"]
        norm[:, 0] = 2.0 * (pos[:, 0] - self.bounds["x_min"]) / Lx - 1.0
        norm[:, 1] = 2.0 * (pos[:, 1] - self.bounds["y_min"]) / Ly - 1.0
        norm[:, 2] = 2.0 * pos[:, 2] / self.bounds["z_max"] - 1.0
        return norm.view(1, -1, 1, 1, 3)

    # ──────────────────────────────────────────────────────────────────────────
    def _get_dmk_fields(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trilinear interpolation of the 5 DMK fields in a single GPU call.
        AUSTAL Annex D.2.3 : σ̂ and K̂ bilinearly interpolated.

        Returns
        ───────
        mean_wind  : (N, 3) [m/s]  interpolated mean wind
        extra_turb : (N,)   [m/s]  wake σ̂ (isotropic)
        extra_K    : (N,)   [m²/s] wake diffusivity K̂
        """
        grid = self._normalize_coords(pos)
        sampled = (
            F.grid_sample(
                self.dmk_5d,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .view(5, -1)
            .T
        )  # (N, 5)

        return sampled[:, :3], sampled[:, 3], sampled[:, 4]

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_markov_tensors(
        self,
        sig_u_sq: torch.Tensor,
        sig_v_sq: torch.Tensor,
        sig_w_sq: torch.Tensor,
        K_u: torch.Tensor,
        K_v: torch.Tensor,
        K_w: torch.Tensor,
        u_star_sq: float,
        tau: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,  # Psi_uu, Psi_ww, Psi_uw, Psi_wu
        torch.Tensor,  # Psi_vv
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,  # Omega_uu, Omega_ww, Omega_uw
        torch.Tensor,  # Omega_vv
    ]:
        """
        Computes the Ψ and Ω tensors of the Markov scheme (LASAT 5.3.1).

        LASAT Eq. 5.3.185 : Σ_uw = −u★²  (shear covariance)
        LASAT Eq. 5.3.186 : K diagonal

        Parameters
        ──────────
        sig_u_sq, sig_v_sq, sig_w_sq : total variances (background + wake) (N,) [m²/s²]
        K_u, K_v, K_w                : total diffusivities (N,) [m²/s]
        u_star_sq                    : Python scalar u★² [m²/s²]
        tau                          : effective timestep (N,) [s]

        Returns
        ───────
        Ψ and Ω tensors for the u (downwind), w (vertical) and v (crosswind) components.
        Ψ is in the wind frame (u = downwind = direction of V_eff).
        """
        eps = 1e-8

        # ?? Composante transverse v (d?coupl?e) ? LASAT Eq. 5.3.188-5.3.190 ??
        Phi_vv = sig_v_sq / K_v.clamp(min=eps)
        p_v = 0.5 * tau * Phi_vv
        p_v = p_v.clamp(max=1.0 - eps)  # condition Eq. 5.3.204 : p_v ? 1
        Psi_vv = (1.0 - p_v) / (1.0 + p_v)
        Omega_vv = 4.0 * sig_v_sq * p_v * (1.0 + p_v).pow(-2)

        # ?? Plan u/w (avec couplage ?_uw = ?u??) ?????????????????????????????
        # ?_uu = ?_uu/K_uu = ?_u?/K_u   (la contribution diagonale de ?_uw/K_ww
        # est ignor?e car K est diagonal ? LASAT Eq. 5.3.187)
        # Remarque : LASAT 5.3.187 donne ?_uw = ?_uw/K_ww, mais pour construire
        # les scalaires p_u, p_w on utilise les ?l?ments diagonaux de ?.
        Phi_uu = sig_u_sq / K_u.clamp(min=eps)
        Phi_ww = sig_w_sq / K_w.clamp(min=eps)
        p_u = (0.5 * tau * Phi_uu).clamp(max=1.0 - eps)
        p_w = (0.5 * tau * Phi_ww).clamp(max=1.0 - eps)

        # Covariance de cisaillement (LASAT Eq. 5.3.185)
        Sigma_uw = -u_star_sq  # scalaire Python < 0

        # Corr?lations sans dimension (LASAT Eq. 5.3.193-5.3.195)
        rho_u = Sigma_uw / sig_u_sq.clamp(min=eps)  # (N,) < 0
        rho_w = Sigma_uw / sig_w_sq.clamp(min=eps)  # (N,) < 0
        q = rho_u * rho_w  # (N,) > 0 (produit de deux valeurs n?gatives)
        q = q.clamp(max=1.0 - eps)  # condition Eq. 5.3.203 : q < 1

        # ?? (LASAT Eq. 5.3.196)
        Delta = (1.0 + p_u) * (1.0 + p_w) - q * p_u * p_w
        Delta = Delta.clamp(min=eps)

        # Tenseur ? dans le plan u/w (LASAT Eq. 5.3.197)
        Psi_uu = ((1.0 - p_u) * (1.0 + p_w) + q * p_u * p_w) / Delta
        Psi_ww = ((1.0 + p_u) * (1.0 - p_w) + q * p_u * p_w) / Delta
        Psi_uw = -2.0 * p_w * rho_w / Delta
        Psi_wu = -2.0 * p_u * rho_u / Delta

        # Tenseur ? dans le plan u/w (LASAT Eq. 5.3.198-5.3.200)
        inv_D2 = Delta.pow(-2)
        Omega_uu = 4.0 * sig_u_sq * (q * p_w + p_u * (1.0 + (1.0 - q) * p_w).pow(2)) * inv_D2
        Omega_ww = 4.0 * sig_w_sq * (q * p_u + p_w * (1.0 + (1.0 - q) * p_u).pow(2)) * inv_D2
        Omega_uw = 4.0 * Sigma_uw * (p_u + p_w + 2.0 * (1.0 - q) * p_u * p_w) * inv_D2

        # Garantit la d?finition positive (ne devrait pas ?tre n?cessaire si les
        # troncatures ci-dessus sont appliqu?es, mais s?curit? num?rique)
        Omega_uu = Omega_uu.clamp(min=0.0)
        Omega_ww = Omega_ww.clamp(min=0.0)
        Omega_vv = Omega_vv.clamp(min=0.0)

        return (
            Psi_uu,
            Psi_ww,
            Psi_uw,
            Psi_wu,
            Psi_vv,
            Omega_uu,
            Omega_ww,
            Omega_uw,
            Omega_vv,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_drift_W(
        self,
        Psi_uu: torch.Tensor,
        Psi_ww: torch.Tensor,
        Psi_uw: torch.Tensor,
        Psi_wu: torch.Tensor,
        d_sig_u_sq_dz: torch.Tensor,
        d_sig_w_sq_dz: torch.Tensor,
        tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Drift velocity W to maintain the well-mixed condition (WMC).

        LASAT 5.3.1, Eq. 5.3.176 (spatially constant τ) :
            W = (τ/2)·(I + Ψ)·(∇·Θ)   with Θ = Σ

        For atmospheric geometry (turbulence varying only with z),
        only the vertical gradient ∂/∂z is non-zero. In the u/w plane :

            W_u = (τ/2)·[(1+Ψ_uu)·∂σ_u²/∂z + Ψ_uw·∂σ_w²/∂z]
            W_w = (τ/2)·[Ψ_wu·∂σ_u²/∂z   + (1+Ψ_ww)·∂σ_w²/∂z]

        Note : the wake turbulence σ̂ is spatially homogeneous in the
        wake zone in the DMK model (AUSTAL D.34) → ∂σ̂²/∂z = 0.
        It therefore does not contribute to W.
        """
        half_tau = 0.5 * tau

        W_u = half_tau * ((1.0 + Psi_uu) * d_sig_u_sq_dz + Psi_uw * d_sig_w_sq_dz)
        W_w = half_tau * (Psi_wu * d_sig_u_sq_dz + (1.0 + Psi_ww) * d_sig_w_sq_dz)
        return W_u, W_w

    # ──────────────────────────────────────────────────────────────────────────
    def _cholesky_2x2(
        self,
        Omega_uu: torch.Tensor,
        Omega_ww: torch.Tensor,
        Omega_uw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2×2 Cholesky decomposition of Ω (LASAT Eq. 5.3.168 : Λ·Λᵀ = Ω).

        For Ω = | Ω_uu  Ω_uw |, the lower triangular decomposition is :
                | Ω_wu  Ω_ww |

        L = | L_uu   0   |     such that L·Lᵀ = Ω
            | L_wu  L_ww |

        L_uu  = √Ω_uu
        L_wu  = Ω_uw / L_uu                  (if Ω_uu > 0)
        L_ww  = √(Ω_ww − L_wu²)

        Returns : L_uu, L_wu, L_ww, L_vv = √Ω_vv (separate v component)
        """
        eps = 1e-12
        L_uu = torch.sqrt(Omega_uu.clamp(min=eps))
        L_wu = Omega_uw / L_uu.clamp(min=eps)
        L_ww = torch.sqrt((Omega_ww - L_wu**2).clamp(min=eps))
        return L_uu, L_wu, L_ww

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_boundaries(
        self,
        pos_before_step: torch.Tensor,
        active_mask: torch.Tensor,
    ):
        """
        Boundary conditions for active particles.

        1. Ground + mixing layer
        ────────────────────────
        Unstable / neutral (L_M ≤ 0) — well-mixed CBL :
            Specular reflection at the ground AND at roof h_m (exact O(1) GPU fold).
            LASAT verifies the well-mixed condition with this fold (Tests 11-14).

        Stable (L_M > 0) — decoupled layer :
            The fold is only applied below h_m.
            Particles numerically ejected above advect freely
            with σ ≈ 0 (VDI 3783 Blatt 8 profile : exp(−z/hm) → 0 for z ≫ hm).

        2. Buildings
        ────────────
        Revert to pre-step position + inversion of turbulent velocities.
        Single authority : no double obstacle detection (cf. AUSTAL D.2).
        """
        if not active_mask.any():
            return

        p = self.pos[active_mask].clone()
        v = self.vel_turb[active_mask].clone()
        hm = self.profile.h_m

        # ?? 1a. Rebond au sol (commun ? toutes les classes de stabilit?) ?????
        neg = p[:, 2] < 0.0
        pz = torch.where(neg, -p[:, 2], p[:, 2])
        vz = torch.where(neg, -v[:, 2], v[:, 2])

        if self.profile.L_M > 0.0:
            # ?? 1b. Stable : barri?re ? sens unique uniquement au sol ?????????
            # Pour z ? hm : pliage entre le sol et hm.
            # Pour z > hm : particule libre (? ? 0, advection pure).
            above_hm = pz > hm
            eps_hm = hm * (1.0 - 1e-6)
            pz_in = pz.clamp(max=eps_hm)
            n = torch.floor(pz_in / hm)
            z_mod = pz_in - n * hm
            n_odd = (n.long() % 2) == 1
            z_fold = torch.where(n_odd, hm - z_mod, z_mod)
            vz_fold = torch.where(n_odd, -vz, vz)
            p[:, 2] = torch.where(above_hm, pz, z_fold)
            v[:, 2] = torch.where(above_hm, vz, vz_fold)
        else:
            # ?? 1b. Instable / neutre : pliage classique de CBL ???????????????
            n = torch.floor(pz / hm)
            z_mod = pz - n * hm
            n_odd = (n.long() % 2) == 1
            p[:, 2] = torch.where(n_odd, hm - z_mod, z_mod)
            v[:, 2] = torch.where(n_odd, -vz, vz)

        # ?? 2. B?timents ??????????????????????????????????????????????????????
        grid = self._normalize_coords(p)
        in_obs = (
            F.grid_sample(
                self.obstacle_mask,
                grid,
                mode="nearest",
                padding_mode="border",
                align_corners=True,
            ).view(-1)
            < 0.5
        )
        if in_obs.any():
            p[in_obs] = pos_before_step[in_obs]
            v[in_obs] *= -1.0

        self.pos[active_mask] = p
        self.vel_turb[active_mask] = v

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, dt: float):
        """
        A complete timestep of the LASAT discrete Markov scheme (VDI 3945 Blatt 3).

        Sequence (LASAT 5.3.1) :
        ──────────────────────────
        0. Plume : Δz and σ_plume for particles with vs_exit > 0
        1. VDI 3783 Blatt 8 turbulence profiles (σ, K, ∂σ²/∂z)
        2. Interpolated DMK fields (V, σ̂, K̂) + effective wind Veff (Eq. 5.3.166)
        3. Total variances and diffusivities (background + wake + plume)
        4. Constraint τ ≤ 2·T_L (Eq. 5.3.204-205) → τ_eff per particle
        5. Ψ and Ω tensors (Eq. 5.3.188-5.3.200)
        6. Cholesky decomposition : Λ (Eq. 5.3.168)
        7. Drift W (Eq. 5.3.176)
        8. Velocity update : u_new = Ψ·u + W + Λ·R  (Eq. 5.3.164)
        9. Displacement : x_new = x + τ_eff·(Veff + u_new)     (Eq. 5.3.165)
        10. Boundary conditions (ground, CBL, buildings)
        11. Deactivation outside horizontal domain
        """
        active = self.is_alive
        if not active.any():
            return

        p_active = self.pos[active].clone()
        u_active = self.vel_turb[active].clone()  # vitesse turbulente (relative ? V)
        z_active = p_active[:, 2]
        N = p_active.shape[0]

        # ?? 0. PANACHE (AUSTAL ?3.4) ??????????????????????????????????????????
        # AUSTAL : ? la turbulence initiale induite par le panache est fix?e ? 10%
        #           de la vitesse de sortie r?sultante et effective. ?
        # vs_exit d?cro?t exponentiellement avec le temps caract?ristique ts_plume.
        vs_exit_active = self.vs_exit[active].clone()
        ts_pl_active = self.ts_plume[active]
        has_plume = ts_pl_active > 0.0

        dz_plume = torch.zeros(N, device=self.device)
        sig_plume = torch.zeros(N, device=self.device)  # ?_plume = 0.1?vs_exit

        if has_plume.any():
            decay = torch.exp(-dt / ts_pl_active[has_plume])
            vs_new = vs_exit_active[has_plume] * decay
            # Mont?e trap?zo?dale : ?z = 0.5?(vs + vs_new)?dt
            dz_plume[has_plume] = 0.5 * (vs_exit_active[has_plume] + vs_new) * dt
            # Turbulence initiale = 10% de la vitesse de sortie effective (AUSTAL ?3.4)
            sig_plume[has_plume] = 0.1 * vs_exit_active[has_plume]
            vs_exit_active[has_plume] = vs_new
            self.vs_exit[active] = vs_exit_active

        # ?? 1. PROFILS VDI 3783 Blatt 8 ???????????????????????????????????????
        sig_u, sig_v, sig_w, T_Lu, T_Lv, T_Lw, d_var_w_dz = self.profile.get_turbulence_params(
            z_active
        )
        sig_u_sq_bg = sig_u**2  # variances de fond ? r?serv?es ? W
        sig_v_sq_bg = sig_v**2
        sig_w_sq_bg = sig_w**2  # r?serv? ? la d?rive W (gradient analytique)

        # Diffusivit?s de fond
        K_u_bg = sig_u_sq_bg * T_Lu
        K_v_bg = sig_v_sq_bg * T_Lv
        K_w_bg = sig_w_sq_bg * T_Lw

        # Gradient ??_u?/?z ? d?riv? analytiquement du profil VDI 3783 :
        # ?_u = fu?u??exp(?z/hm) ? ??_u?/?z = ?2/hm ? ?_u?  (stable/neutre)
        # En instable, la d?pendance diff?re mais ?_u ? ?_w.
        # Approximation document?e : on utilise le m?me facteur que pour ?_w.
        # LASAT simplifie en utilisant seulement d_var_w_dz pour le couplage W_wu.
        d_sig_u_sq_dz = -(2.0 / self.profile.h_m) * sig_u_sq_bg
        d_sig_w_sq_dz = d_var_w_dz  # calcul? par turbulence.py

        # ?? 2. CHAMPS DMK + VENT EFFECTIF ?????????????????????????????????????
        # V(x?) au point initial
        V_init, extra_turb, extra_K = self._get_dmk_fields(p_active)

        # Veff = 0.5?[V(x?) + V(x? + dt?V(x?))]  (LASAT Eq. 5.3.166)
        p_pred = p_active + dt * V_init
        V_pred, _, _ = self._get_dmk_fields(p_pred)
        V_eff = 0.5 * (V_init + V_pred)

        # ?? 3. VARIANCES ET DIFFUSIVIT?S TOTALES ??????????????????????????????
        # Variances additives : ??_tot = ??_bg + ??? + ??_plume
        extra_sq = extra_turb**2
        plume_sq = sig_plume**2

        sig_u_sq = sig_u_sq_bg + extra_sq + plume_sq
        sig_v_sq = sig_v_sq_bg + extra_sq + plume_sq
        sig_w_sq = sig_w_sq_bg + extra_sq + plume_sq

        # Diffusivit?s totales : K_tot = K_bg + K?  (AUSTAL D.35)
        K_u = K_u_bg + extra_K
        K_v = K_v_bg + extra_K
        K_w = K_w_bg + extra_K

        # ?? 4. CONTRAINTE ? ? 2?T_L (LASAT Eq. 5.3.204-205) ??????????????????
        # Pour garantir la d?finition positive de ?, p_u = ?/2??_uu ? 1
        # ? ? ? 2?T_L = 2?K/??
        T_Lu_eff = (K_u / sig_u_sq.clamp(min=1e-8)).clamp(min=1e-4)
        T_Lv_eff = (K_v / sig_v_sq.clamp(min=1e-8)).clamp(min=1e-4)
        T_Lw_eff = (K_w / sig_w_sq.clamp(min=1e-8)).clamp(min=1e-4)

        tau = torch.full((N,), dt, device=self.device, dtype=torch.float32)
        tau = torch.minimum(tau, 2.0 * T_Lu_eff)
        tau = torch.minimum(tau, 2.0 * T_Lv_eff)
        tau = torch.minimum(tau, 2.0 * T_Lw_eff)
        # Garde basse : ? ne peut pas ?tre nul (?vite W = 0)
        tau = tau.clamp(min=1e-3)

        # u?? pour le couplage ?_uw (LASAT Eq. 5.3.185)
        u_star_sq = self.profile.u_star**2

        # ?? 5. TENSEURS ? ET ? ????????????????????????????????????????????????
        (
            Psi_uu,
            Psi_ww,
            Psi_uw,
            Psi_wu,
            Psi_vv,
            Omega_uu,
            Omega_ww,
            Omega_uw,
            Omega_vv,
        ) = self._compute_markov_tensors(
            sig_u_sq,
            sig_v_sq,
            sig_w_sq,
            K_u,
            K_v,
            K_w,
            u_star_sq,
            tau,
        )

        # ?? 6. D?COMPOSITION DE CHOLESKY ? (LASAT Eq. 5.3.168) ???????????????
        L_uu, L_wu, L_ww = self._cholesky_2x2(Omega_uu, Omega_ww, Omega_uw)
        L_vv = torch.sqrt(Omega_vv.clamp(min=1e-12))

        # ?? 7. D?RIVE W (LASAT Eq. 5.3.176) ???????????????????????????????????
        W_u, W_w = self._compute_drift_W(
            Psi_uu,
            Psi_ww,
            Psi_uw,
            Psi_wu,
            d_sig_u_sq_dz,
            d_sig_w_sq_dz,
            tau,
        )
        # W_v = 0 (pas de gradient horizontal dans le profil atmosph?rique 1D)
        W_v = torch.zeros_like(W_u)

        # ?? 8. MISE ? JOUR DE LA VITESSE (LASAT Eq. 5.3.164) ?????????????????
        # u_new = ??u_old + W + ??R
        # R ~ N(0, I) ? vecteur de bruit blanc gaussien (3 composantes ind?pendantes)
        R = torch.randn((N, 3), device=self.device, dtype=torch.float32)
        R_u, R_v, R_w = R[:, 0], R[:, 1], R[:, 2]

        # Le sch?ma de Markov est exprim? dans le rep?re local du vent (u = sous le vent).
        # Pour des vents ? direction variable, les composantes u_active[:,0]
        # et u_active[:,2] jouent le r?le de u (sous le vent) et w (verticale).
        # La composante u_active[:,1] est la composante transverse v.
        #
        # LASAT ?5.3 note 12 : ? le rep?re du vent est un rep?re cart?sien
        # dont l'axe z est dirig? vers le haut et l'axe x align? sous le vent. ?
        # Dans notre impl?mentation GPU, l'axe x du domaine est align? avec le
        # vent (pas de rotation explicite du rep?re ? approximation valable pour
        # des vents majoritairement orient?s selon x). Pour des simulations
        # avec une rotation significative du vent, une rotation explicite devrait
        # ?tre appliqu?e ici.

        u_old = u_active[:, 0]
        v_old = u_active[:, 1]
        w_old = u_active[:, 2]

        # Plan u/w (coupl? via ? et ? 2?2)
        u_new = Psi_uu * u_old + Psi_uw * w_old + W_u + L_uu * R_u
        w_new = Psi_wu * u_old + Psi_ww * w_old + W_w + L_wu * R_u + L_ww * R_w
        # Composante v (d?coupl?e)
        v_new = Psi_vv * v_old + W_v + L_vv * R_v

        u_active[:, 0] = u_new
        u_active[:, 1] = v_new
        u_active[:, 2] = w_new

        # ?? 9. D?PLACEMENT (LASAT Eq. 5.3.165) ????????????????????????????????
        # x_new = x + ?_eff?(V_eff + u_new)
        # + contribution verticale du panache ?z
        disp = tau.unsqueeze(1) * (V_eff + u_active)
        disp[:, 2] += dz_plume

        p_before = p_active.clone()  # position de repli en cas de collision
        p_active = p_active + disp

        # ?criture en m?moire GPU
        self.pos[active] = p_active
        self.vel_turb[active] = u_active

        # ?? 10. CONDITIONS AUX LIMITES (autorit? unique) ?????????????????????
        self._apply_boundaries(p_before, active)

        # ?? 11. D?SACTIVATION HORS DU DOMAINE HORIZONTAL ?????????????????????
        out = (
            (self.pos[:, 0] < self.bounds["x_min"])
            | (self.pos[:, 0] > self.bounds["x_max"])
            | (self.pos[:, 1] < self.bounds["y_min"])
            | (self.pos[:, 1] > self.bounds["y_max"])
        )
        self.is_alive[out] = False
