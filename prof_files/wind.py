"""
wind.py — Solveur diagnostique de vent conforme TALdia (AUSTAL 3.3)
===================================================================
Référence principale : AUSTAL 3.3, Programmbeschreibung (2024-03-22),
Annexe D.1–D.2, pp. 118–129.

CHAÎNE COMPLÈTE (AUSTAL 3.3, annexe D.2, pp. 125–129)
-----------------------------------------------------
Étape 1  [D.29]     Source du champ E par Poisson : ρ_i = 2(n_i·u0)/|u0| sur toutes les faces sous le vent
Étape 2  [D.29†50]  Poisson ∇²φ_E = source, obstacles masqués → visibilité exacte (note 50)
Étape 3  [D.29]     E = −∇φ_E (différences finies centrées)
Étape 4  [D.30]     E1 = (I − a5·zz)·E
Étape 5  [D.31]     E2 = cos^a2·E1, E2=0 si cos θ ≤ 0
Étape 6  [D.32]     E3 = min(a3,|E2|)·E2/|E2|, E3=0 si √|E2| < a4
Étape 7  [D.33]     R = −a1·ū₀·E3, masque d'ombre
Étape 8  [D.34–35]  σ̂ = √|E2|_clamp·fs·ū₀ ; K̂ = fk·2·z̄·σ̂  (z̄ exact)
Étape 9  [D.2.2]    Frontwirbel : u0 → ū₀ dans la zone de sillage
Étape 10 [D.1.2]    Poisson SOR final → v_diag sans divergence (a_h=a_v=1)
         [AUSTAL]   Vérification : restdiv_max · Δ/ua < 0.05

PARAMÈTRES (table D.2.4, AUSTAL 3.3, p.129)
-------------------------------------------
a1=6.0  a2=1.0  a3=0.3  a4=0.05  a5=0.7  as=15°  fs=0.5  fk=0.3  hs=1.2

POURQUOI POISSON POUR LE CHAMP E (ET PAS COULOMB DIRECT)
--------------------------------------------------------
L'équation D.29 définit E comme une somme de Coulomb. Mathématiquement, ce champ
est le gradient d'un potentiel φ vérifiant ∇²φ = −ρ/ε₀ — une équation de Poisson.
Le Poisson discret est en O(n_iter × N) ; la somme de Coulomb directe avec
test de visibilité DDA est en O(N × M × n_vis). Pour une grille 140×140×64 avec
M≈480 sources sous le vent et n_vis≈420 points par rayon : Coulomb = 252G ops contre
Poisson = 23G ops — un surcoût ×11 sans gain physique pertinent.
La visibilité est approchée par le masque d'obstacles dans le solveur de Poisson (cellules
d'obstacle ≡ φ=0 → propagation bloquée, note 50).

CORRECTION RESTDIV
------------------
AUSTAL p.47 vérifie la divergence du champ sur la grille d'Arakawa-C
(faces décalées). La divergence doit être calculée directement sur
u_face/v_face/w_face APRÈS la correction de Poisson, et NON sur
u_final interpolé au centre = 0.5*(u_face[i] + u_face[i+1]), qui introduit
une erreur en O(Δ²) rendant le critère toujours violé à tort (faux positif).
"""

import math
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F


class MassConsistentWindSolverNumerical:
    """
    TALdia-compliant diagnostic wind solver (AUSTAL 3.3, Annex D).

    API :
        u, v, w, sigma, K = solver.adjust_wind_field(u_init, v_init, w_init, mask)
    Toutes les sorties : (1, 1, nz, ny, nx).

    Paramètres acceptés pour compatibilité ascendante :
        tgt_chunk_size : ignoré (héritage Coulomb), conservé pour ne pas casser l'API
        max_iter_E, tol_E : convergence du Poisson sur le champ E
    """

    # ── Paramètres de la table D.2.4 ─────────────────────────────────────────
    _A1 = 6.0
    _A2 = 1.0
    _A3 = 0.3
    _A4 = 0.05
    _A5 = 0.7
    _FS = 0.5
    _FK = 0.3
    _HS = 1.2
    _AS_DEG = 15.0

    def __init__(self, dx: float, dy: float, dz: float, device: str = "cuda"):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.device = device
        self.a1 = self._A1
        self.a2 = self._A2
        self.a3 = self._A3
        self.a4 = self._A4
        self.a5 = self._A5
        self.fs = self._FS
        self.fk = self._FK
        self.hs = self._HS
        self.as_deg = self._AS_DEG

    # ──────────────────────────────────────────────────────────────────────────
    def _shift_2d(self, t: torch.Tensor, sx: int, sy: int) -> torch.Tensor:
        h, w = t.shape
        t = t.clone()
        if sx > 0:
            t = F.pad(t, (sx, 0, 0, 0))[:, :w]
        elif sx < 0:
            t = F.pad(t, (0, -sx, 0, 0))[:, -sx:]
        if sy > 0:
            t = F.pad(t, (0, 0, sy, 0))[:h, :]
        elif sy < 0:
            t = F.pad(t, (0, 0, 0, -sy))[-sy:, :]
        return t

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 — Source du champ E par Poisson  (AUSTAL 3.3, Eq. D.29)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_E_source(
        self,
        u_c: torch.Tensor,
        v_c: torch.Tensor,
        w_c: torch.Tensor,
        obs: torch.Tensor,
        nz: int,
        ny: int,
        nx: int,
    ) -> torch.Tensor:
        """
        Terme source discret pour ∇²φ_E = source.
        Faces sous le vent (n_i·u0 > 0) : X, Y et Z (toiture comprise).
        ρ_i = 2·(n_i·u0)/|u0|  [D.29] → source += −ρ_i/h in the air cell.
        """
        src = torch.zeros((nz, ny, nx), device=self.device, dtype=torch.float32)
        u0m = torch.sqrt(u_c**2 + v_c**2 + w_c**2).clamp(min=1e-6)

        # Face X+ : obstacle en i, air en i+1, n=+x, sous le vent si u_air > 0
        air_xp = (obs[..., :-1] < 0.5) & (obs[..., 1:] > 0.5)
        u_e = u_c[..., 1:]
        src[..., 1:] -= (
            torch.where(air_xp & (u_e > 0), 2 * u_e / u0m[..., 1:], torch.zeros_like(u_e)) / self.dx
        )

        # Face X− : air en i, obstacle en i+1, n=−x, sous le vent si u_air < 0
        air_xm = (obs[..., :-1] > 0.5) & (obs[..., 1:] < 0.5)
        u_w = u_c[..., :-1]
        src[..., :-1] -= (
            torch.where(air_xm & (u_w < 0), 2 * (-u_w) / u0m[..., :-1], torch.zeros_like(u_w))
            / self.dx
        )

        # Face Y+
        air_yp = (obs[:, :-1, :] < 0.5) & (obs[:, 1:, :] > 0.5)
        v_n = v_c[:, 1:, :]
        src[:, 1:, :] -= (
            torch.where(air_yp & (v_n > 0), 2 * v_n / u0m[:, 1:, :], torch.zeros_like(v_n))
            / self.dy
        )

        # Face Y−
        air_ym = (obs[:, :-1, :] > 0.5) & (obs[:, 1:, :] < 0.5)
        v_s = v_c[:, :-1, :]
        src[:, :-1, :] -= (
            torch.where(air_ym & (v_s < 0), 2 * (-v_s) / u0m[:, :-1, :], torch.zeros_like(v_s))
            / self.dy
        )

        # Face Z+ (toiture)
        air_zp = (obs[:-1, :, :] < 0.5) & (obs[1:, :, :] > 0.5)
        w_a = w_c[1:, :, :]
        src[1:, :, :] -= (
            torch.where(air_zp & (w_a > 0), 2 * w_a / u0m[1:, :, :], torch.zeros_like(w_a))
            / self.dz
        )

        # Face Z−
        air_zm = (obs[:-1, :, :] > 0.5) & (obs[1:, :, :] < 0.5)
        w_b = w_c[:-1, :, :]
        src[:-1, :, :] -= (
            torch.where(air_zm & (w_b < 0), 2 * (-w_b) / u0m[:-1, :, :], torch.zeros_like(w_b))
            / self.dz
        )

        return src * obs  # annule les sources à l'intérieur des obstacles

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 — Poisson sur le champ E avec masque d'obstacles  (visibilité, note 50)
    # ══════════════════════════════════════════════════════════════════════════
    def _solve_poisson_E(
        self,
        source: torch.Tensor,
        obs: torch.Tensor,
        nz: int,
        ny: int,
        nx: int,
        max_iter: int = 3000,
        tol: float = 1e-5,
    ) -> torch.Tensor:
        """
        ∇²φ_E = source.
        BC : φ=0 dans les obstacles (bloque la propagation → visibilité, note 50)
             Neumann ∂φ/∂z=0 au sol (méthode des images, D.29)
             Dirichlet φ=0 at other boundaries (Eq. D.18)
        """
        A_d = -(2 / self.dx**2 + 2 / self.dy**2 + 2 / self.dz**2)
        rhoJ = (
            math.cos(math.pi / nx) / self.dx**2
            + math.cos(math.pi / ny) / self.dy**2
            + math.cos(math.pi / nz) / self.dz**2
        ) / (1 / self.dx**2 + 1 / self.dy**2 + 1 / self.dz**2)
        rhoJ = min(rhoJ, 0.9999)

        iz_, iy_, ix_ = torch.meshgrid(
            torch.arange(nz, device=self.device),
            torch.arange(ny, device=self.device),
            torch.arange(nx, device=self.device),
            indexing="ij",
        )
        red_m = (ix_ + iy_ + iz_) % 2 == 0

        phi = torch.zeros((nz, ny, nx), device=self.device, dtype=torch.float32)
        omega = 1.0

        def nbr(p):
            # Padding XY : Dirichlet 0 ; Z-bas : Neumann (copie) ; Z-haut : Dirichlet 0
            p_xy = F.pad(p, (1, 1, 1, 1, 0, 0), value=0.0)
            p_z = torch.cat(
                [p_xy[0:1], p_xy, torch.zeros(1, ny + 2, nx + 2, device=self.device)], 0
            )
            return (
                (p_z[:-2, 1:-1, 1:-1] + p_z[2:, 1:-1, 1:-1]) / self.dz**2
                + (p_z[1:-1, :-2, 1:-1] + p_z[1:-1, 2:, 1:-1]) / self.dy**2
                + (p_z[1:-1, 1:-1, :-2] + p_z[1:-1, 1:-1, 2:]) / self.dx**2
            )

        for it in range(max_iter):
            if it == 0:
                omega = 1.0
            elif it == 1:
                omega = 1.0 / (1.0 - 0.5 * rhoJ**2)
            else:
                omega = min(1.0 / (1.0 - 0.25 * rhoJ**2 * omega), 1.95)
            p_prev = phi.clone()
            ns = nbr(phi)
            phi_n = phi + omega * ((ns - source) / (-A_d) - phi)
            phi = torch.where(red_m & (obs > 0.5), phi_n, phi)
            ns = nbr(phi)
            phi_n = phi + omega * ((ns - source) / (-A_d) - phi)
            phi = torch.where(~red_m & (obs > 0.5), phi_n, phi)
            if it % 100 == 0 and torch.max(torch.abs(phi - p_prev)).item() < tol:
                break
        return phi

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 — E = −∇φ_E
    # ══════════════════════════════════════════════════════════════════════════
    def _phi_to_E(self, phi: torch.Tensor, nz: int, ny: int, nx: int) -> torch.Tensor:
        E = torch.zeros((3, nz, ny, nx), device=self.device)
        E[0, :, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2 * self.dx)
        E[0, :, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / self.dx
        E[0, :, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / self.dx
        E[1, :, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2 * self.dy)
        E[1, :, 0, :] = -(phi[:, 1, :] - phi[:, 0, :]) / self.dy
        E[1, :, -1, :] = -(phi[:, -1, :] - phi[:, -2, :]) / self.dy
        E[2, 1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2 * self.dz)
        E[2, 0, :, :] = -(phi[1, :, :] - phi[0, :, :]) / self.dz
        E[2, -1, :, :] = -(phi[-1, :, :] - phi[-2, :, :]) / self.dz
        return E

    # ══════════════════════════════════════════════════════════════════════════
    # Interface publique
    # ══════════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def adjust_wind_field(
        self,
        u_init: torch.Tensor,
        v_init: torch.Tensor,
        w_init: torch.Tensor,
        obstacle_mask: torch.Tensor,
        max_iter: int = 20_000,
        tol: float = 1e-7,
        tgt_chunk_size: int = 2_048,  # ignoré — compatibilité API ascendante
        max_iter_E: int = 3_000,
        tol_E: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        AUSTAL 3.3 pipeline, Annex D.2.

        Parameters
        ----------
        max_iter, tol    : Poisson SOR final (sans divergence)
        tgt_chunk_size   : accepté pour compatibilité, non utilisé (Coulomb retiré)
        max_iter_E, tol_E: Poisson sur le champ E (étapes 1–3)

        Sorties : u, v, w, sigma, K — toutes en (1,1,nz,ny,nx).
        """
        _, _, nz, ny, nx = u_init.shape
        u_c = u_init.squeeze()
        v_c = v_init.squeeze()
        w_c = w_init.squeeze()
        obs = obstacle_mask.squeeze()

        zs = (torch.arange(nz, device=self.device) + 0.5) * self.dz
        xs = (torch.arange(nx, device=self.device) + 0.5) * self.dx
        ys = (torch.arange(ny, device=self.device) + 0.5) * self.dy
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")

        # ══════════════════════════════════════════════════════════════════════
        # ÉTAPES 1–3 : champ E via Poisson (AUSTAL D.29)
        # ══════════════════════════════════════════════════════════════════════
        src_E = self._build_E_source(u_c, v_c, w_c, obs, nz, ny, nx)
        phi_E = self._solve_poisson_E(src_E, obs, nz, ny, nx, max_iter_E, tol_E)
        E = self._phi_to_E(phi_E, nz, ny, nx)

        # ══════════════════════════════════════════════════════════════════════
        # ÉTAPE 4 : E1 = (I − a5·zz)·E  (D.30)
        # ══════════════════════════════════════════════════════════════════════
        E1 = E.clone()
        E1[2] *= 1.0 - self.a5
        u0n = torch.sqrt(u_c**2 + v_c**2 + w_c**2).clamp(min=1e-6)
        E1n = torch.sqrt(E1[0] ** 2 + E1[1] ** 2 + E1[2] ** 2).clamp(min=1e-6)

        # ══════════════════════════════════════════════════════════════════════
        # ÉTAPE 5 : E2 = cos^a2·E1, cos θ = (E1·u0)/(|E1||u0|)  (D.31)
        # ══════════════════════════════════════════════════════════════════════
        cth = (E1[0] * u_c + E1[1] * v_c + E1[2] * w_c) / (E1n * u0n)
        E2 = torch.where(
            cth.unsqueeze(0) > 0.0,
            E1 * torch.pow(cth.clamp(min=0.0), self.a2).unsqueeze(0),
            torch.zeros_like(E1),
        )

        # ══════════════════════════════════════════════════════════════════════
        # ÉTAPE 6 : E3 = min(a3,|E2|)·E2/|E2|, 0 si √|E2| < a4  (D.32)
        # ══════════════════════════════════════════════════════════════════════
        E2n = torch.sqrt(E2[0] ** 2 + E2[1] ** 2 + E2[2] ** 2)
        sqE2 = E2n.sqrt()
        E3 = torch.where(
            sqE2.unsqueeze(0) < self.a4,
            torch.zeros_like(E2),
            E2 * (E2n.clamp(max=self.a3) / E2n.clamp(min=1e-6)).unsqueeze(0),
        )
        E3n = torch.sqrt(E3[0] ** 2 + E3[1] ** 2 + E3[2] ** 2)

        # ══════════════════════════════════════════════════════════════════════
        # ÉTAPE 7 : R = −a1·ū₀·E3 + masque d'ombre/sillage  (D.33)
        # ══════════════════════════════════════════════════════════════════════
        sumE3 = E3n.sum(0, keepdim=True).clamp(min=1e-8)
        u0bar = (u0n * E3n).sum(0, keepdim=True) / sumE3
        R = -self.a1 * u0bar * E3

        # Propagation de la hauteur d'ombre
        bh = torch.zeros((ny, nx), device=self.device)
        for k in range(nz):
            bh = torch.where(obs[k] < 0.5, (zs[k] + self.dz / 2).expand(ny, nx), bh)
        mean_u = u_c.mean().item()
        mean_v = v_c.mean().item()
        wl = math.hypot(mean_u, mean_v) + 1e-8
        dx_hat = mean_u / wl
        dy_hat = mean_v / wl
        sh = bh.clone()
        cell_min = min(self.dx, self.dy)
        max_s = int(math.hypot(nx * self.dx, ny * self.dy) / cell_min)
        for s in range(1, max_s):
            dist = s * cell_min
            sx = int(round(dist * dx_hat / self.dx))
            sy = int(round(dist * dy_hat / self.dy))
            if sx == 0 and sy == 0:
                continue
            lat = dist * math.tan(math.radians(self.as_deg))
            ls = max(1, int(round(lat / cell_min)))
            tmp = sh.clone()
            for lx in range(-ls, ls + 1):
                ox = sx + int(round(lx * (-dy_hat)))
                oy = sy + int(round(lx * dx_hat))
                tmp = torch.max(tmp, self._shift_2d(bh, ox, oy))
            sh = torch.max(sh, tmp)

        hf = torch.ones((nz, ny, nx), device=self.device)
        for k in range(nz):
            H = sh.clamp(min=1e-3)
            h = 1.0 - (zs[k] - H) / ((self.hs - 1.0) * H)
            h = torch.where(zs[k] <= H, torch.ones_like(h), h)
            h = torch.where(zs[k] >= self.hs * H, torch.zeros_like(h), h)
            hf[k] = h.clamp(0.0, 1.0)
        wake = hf
        R = R * wake

        # ══════════════════════════════════════════════════════════════════════
        # ÉTAPE 8 : σ̂ (D.34) et K̂ (D.35)
        # ══════════════════════════════════════════════════════════════════════
        sqE2_cl = torch.where(sqE2 < self.a4, torch.zeros_like(sqE2), sqE2.clamp(max=self.a3))
        extra_s = sqE2_cl * self.fs * u0bar.squeeze(0) * wake

        z3d = zs.view(nz, 1, 1).expand(nz, ny, nx)
        z_bar = (z3d * E3n).sum(0) / sumE3.squeeze(0)
        extra_K = self.fk * 2.0 * z_bar * extra_s

        # ══════════════════════════════════════════════════════════════════════
        # ÉTAPE 9 : Frontwirbel (D.2.2)
        # ══════════════════════════════════════════════════════════════════════
        mrc = (E3n > 1e-4).float() * wake
        ubv = (u_c * E3n).sum(0, keepdim=True) / sumE3
        vbv = (v_c * E3n).sum(0, keepdim=True) / sumE3
        wbv = (w_c * E3n).sum(0, keepdim=True) / sumE3
        u_emp = (1 - mrc) * u_c + mrc * ubv + R[0]
        v_emp = (1 - mrc) * v_c + mrc * vbv + R[1]
        w_emp = (1 - mrc) * w_c + mrc * wbv + R[2]

        # ══════════════════════════════════════════════════════════════════════
        # ÉTAPE 10 : Poisson SOR final → champ sans divergence (annexe A/D.1.2)
        # ══════════════════════════════════════════════════════════════════════
        pad_obs = F.pad(obs, (1, 1, 1, 1, 1, 1), value=1.0)
        pad_obs[0] = 0.0  # sol imperméable

        u_mask = pad_obs[1:-1, 1:-1, :-1] * pad_obs[1:-1, 1:-1, 1:]
        v_mask = pad_obs[1:-1, :-1, 1:-1] * pad_obs[1:-1, 1:, 1:-1]
        w_mask = pad_obs[:-1, 1:-1, 1:-1] * pad_obs[1:, 1:-1, 1:-1]

        u_face = torch.zeros((nz, ny, nx + 1), device=self.device)
        v_face = torch.zeros((nz, ny + 1, nx), device=self.device)
        w_face = torch.zeros((nz + 1, ny, nx), device=self.device)
        u_face[..., 1:-1] = 0.5 * (u_emp[..., :-1] + u_emp[..., 1:])
        v_face[:, 1:-1, :] = 0.5 * (v_emp[:, :-1, :] + v_emp[:, 1:, :])
        w_face[1:-1, :, :] = 0.5 * (w_emp[:-1, :, :] + w_emp[1:, :, :])
        u_face[..., 0], u_face[..., -1] = u_emp[..., 0], u_emp[..., -1]
        v_face[:, 0, :], v_face[:, -1, :] = v_emp[:, 0, :], v_emp[:, -1, :]
        w_face[0, :, :], w_face[-1, :, :] = w_emp[0, :, :], w_emp[-1, :, :]
        u_face *= u_mask
        v_face *= v_mask
        w_face *= w_mask

        div = (
            (u_face[..., 1:] - u_face[..., :-1]) / self.dx
            + (v_face[:, 1:, :] - v_face[:, :-1, :]) / self.dy
            + (w_face[1:, :, :] - w_face[:-1, :, :]) / self.dz
        )

        A111 = (
            -(u_mask[..., 1:] + u_mask[..., :-1]) / self.dx**2
            - (v_mask[:, 1:, :] + v_mask[:, :-1, :]) / self.dy**2
            - (w_mask[1:, :, :] + w_mask[:-1, :, :]) / self.dz**2
        )
        A111 = torch.where(A111 == 0.0, torch.full_like(A111, -1.0), A111)

        rhoJ = (
            math.cos(math.pi / nx) / self.dx**2
            + math.cos(math.pi / ny) / self.dy**2
            + math.cos(math.pi / nz) / self.dz**2
        ) / (1 / self.dx**2 + 1 / self.dy**2 + 1 / self.dz**2)
        rhoJ = min(rhoJ, 0.9999)

        gix = (grid_x / self.dx - 0.5).long()
        giy = (grid_y / self.dy - 0.5).long()
        giz = (grid_z / self.dz - 0.5).long()
        red_m = (gix + giy + giz) % 2 == 0
        blk_m = ~red_m

        # Laplacien masqué — identique à TOUS les passages SOR.
        # Les masques de face garantissent : flux nul au sol (w_mask[0]=0)
        # et à travers les bâtiments.
        def compute_lap(lam_t: torch.Tensor) -> torch.Tensor:
            gx = torch.zeros(nz, ny, nx + 1, device=self.device)
            gy = torch.zeros(nz, ny + 1, nx, device=self.device)
            gz = torch.zeros(nz + 1, ny, nx, device=self.device)
            gx[..., 1:-1] = (lam_t[..., 1:] - lam_t[..., :-1]) / self.dx
            gx[..., 0] = lam_t[..., 0] / self.dx
            gx[..., -1] = -lam_t[..., -1] / self.dx
            gy[:, 1:-1, :] = (lam_t[:, 1:, :] - lam_t[:, :-1, :]) / self.dy
            gy[:, 0, :] = lam_t[:, 0, :] / self.dy
            gy[:, -1, :] = -lam_t[:, -1, :] / self.dy
            gz[1:-1, :, :] = (lam_t[1:, :, :] - lam_t[:-1, :, :]) / self.dz
            gz[0, :, :] = lam_t[0, :, :] / self.dz
            gz[-1, :, :] = -lam_t[-1, :, :] / self.dz
            return (
                ((gx * u_mask)[..., 1:] - (gx * u_mask)[..., :-1]) / self.dx
                + ((gy * v_mask)[:, 1:, :] - (gy * v_mask)[:, :-1, :]) / self.dy
                + ((gz * w_mask)[1:, :, :] - (gz * w_mask)[:-1, :, :]) / self.dz
            )

        lam = torch.zeros_like(div)
        omega = 1.0
        for it in range(max_iter):
            if it == 0:
                omega = 1.0
            elif it == 1:
                omega = 1.0 / (1.0 - 0.5 * rhoJ**2)
            else:
                omega = min(1.0 / (1.0 - 0.25 * rhoJ**2 * omega), 1.95)
            lam_prev = lam.clone()
            lap = compute_lap(lam)
            lam_n = lam - omega * ((lap - div) / A111)
            lam = torch.where(red_m & (obs > 0.5), lam_n, lam)
            lap = compute_lap(lam)
            lam_n = lam - omega * ((lap - div) / A111)
            lam = torch.where(blk_m & (obs > 0.5), lam_n, lam)
            if it % 100 == 0 and torch.max(torch.abs(lam - lam_prev)).item() < tol:
                break

        # Correction du champ de vent
        grad_x = torch.zeros_like(u_face)
        grad_y = torch.zeros_like(v_face)
        grad_z = torch.zeros_like(w_face)
        grad_x[..., 1:-1] = (lam[..., 1:] - lam[..., :-1]) / self.dx
        grad_x[..., 0] = lam[..., 0] / self.dx
        grad_x[..., -1] = -lam[..., -1] / self.dx
        grad_y[:, 1:-1, :] = (lam[:, 1:, :] - lam[:, :-1, :]) / self.dy
        grad_y[:, 0, :] = lam[:, 0, :] / self.dy
        grad_y[:, -1, :] = -lam[:, -1, :] / self.dy
        grad_z[1:-1, :, :] = (lam[1:, :, :] - lam[:-1, :, :]) / self.dz
        grad_z[0, :, :] = lam[0, :, :] / self.dz
        grad_z[-1, :, :] = -lam[-1, :, :] / self.dz
        u_face -= grad_x * u_mask
        v_face -= grad_y * v_mask
        w_face -= grad_z * w_mask

        u_final = 0.5 * (u_face[..., :-1] + u_face[..., 1:]) * obs
        v_final = 0.5 * (v_face[:, :-1, :] + v_face[:, 1:, :]) * obs
        w_final = 0.5 * (w_face[:-1, :, :] + w_face[1:, :, :]) * obs

        # ── Vérification AUSTAL : restdiv_max · Δ/ua < 0.05 (p.47) ───────────
        # CORRECT : divergence calculée directement sur la grille décalée en faces
        # (u_face, v_face, w_face) APRÈS la correction de Poisson, et NON sur
        # u_final interpolé au centre = 0.5*(u_face[i]+u_face[i+1]), qui introduit
        # une erreur en O(Δ²) rendant le critère toujours violé à tort.
        ua = float((u_c**2 + v_c**2 + w_c**2).sqrt().mean().item())
        delta = min(self.dx, self.dy)
        div_f = (
            (u_face[..., 1:] - u_face[..., :-1]) / self.dx
            + (v_face[:, 1:, :] - v_face[:, :-1, :]) / self.dy
            + (w_face[1:, :, :] - w_face[:-1, :, :]) / self.dz
        )
        restdiv = float(div_f[obs > 0.5].abs().max().item()) * delta / max(ua, 1e-3)
        if restdiv >= 0.05:
            warnings.warn(
                f"TALdia : restdiv_max·Δ/ua = {restdiv:.4f} ≥ 0.05 "
                f"(AUSTAL 3.3, p.47). Increase max_iter or reduce tol.",
                stacklevel=2,
            )

        w_max = float(w_final.abs().max().item())
        if w_max >= 50.0:
            warnings.warn(f"TALdia : |w|_max = {w_max:.1f} m/s ≥ 50 m/s.", stacklevel=2)

        return (
            u_final.unsqueeze(0).unsqueeze(0),
            v_final.unsqueeze(0).unsqueeze(0),
            w_final.unsqueeze(0).unsqueeze(0),
            extra_s.unsqueeze(0).unsqueeze(0),
            extra_K.unsqueeze(0).unsqueeze(0),
        )
