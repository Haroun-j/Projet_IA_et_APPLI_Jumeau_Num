"""
turbulence.py — Atmospheric turbulence profiles (VDI 3783 Blatt 8 / AUSTAL 3.3)
=================================================================================
Canonical references
────────────────────
  [AUSTAL]  AUSTAL 3.3, Programmbeschreibung (2024-03-22), Annex G (pp. 134–136)
  [LASAT]   LASAT 3.4 Reference Book (Janicke Consulting, 2018), Section 5.2.3
            "Boundary layer profiles and classification according to TA Luft"
  [TA]      TA Luft 2021, Annex 3


CONSISTENCY WITH wind.py
─────────────────────────
  KAPPA = 0.40  (von Kármán) — identical in wind.py and turbulence.py.
  wind.py uses u★ = KAPPA·ua/ln(ha/z₀).
  turbulence.py receives u★ and z₀ from the caller (solver.py).
"""

import math
import torch


class AustalAtmosphericProfile:
    """
    Wind and turbulence profiles according to VDI 3783 Blatt 8 (2017) / AUSTAL 3.3.
    Equivalent to the boundary layer version 2.6 of LASAT (LASAT 3.4, Sec. 5.2.3).
    Optimized for batch processing on GPU with PyTorch.

    Canonical constants
    ───────────────────
    κ    = 0.40              von Kármán  (TA Luft / VDI 3783 Blatt 8)
    FC   = 1.1×10⁻⁴ s⁻¹     Standard AUSTAL 3.3 Coriolis, Annex G note 54
                              (rounded, covers latitudes 49–52°N)
    OMEGA = 7.29×10⁻⁵ s⁻¹   Earth's angular velocity  (Annex G note 53)
    F25  = 6.1618×10⁻⁵ s⁻¹  fc at 25°N — lower bound for hm  (Eq G.3)

    VDI 3783 Blatt 8 constants  (LASAT 5.2.3, Eq 5.2.115–5.2.119)
    ──────────────────────────
    fw   = 1.3    σ_w coefficient
    fv   = 1.8    σ_v coefficient  (vs 1.6 in the old BUB convention)
    fu   = 2.4    σ_u coefficient  (vs 2.2 in the old BUB convention)
    F_K  = 2.0    K coefficient = F·σ⁴/(5.7·ε)
    α    = 0.3    exponential decay in ε  (LASAT Table p.232)
    β    = 1.0    T_L lower bound multiplier  (LASAT Eq 5.2.108)
    Tmax = 1200 s T_L upper bound  (LASAT Eq 5.2.108)
    """

    # ?? Constantes fondamentales ??????????????????????????????????????????????
    KAPPA = 0.40
    FC = 1.1e-4  # AUSTAL 3.3, annexe G note 54
    OMEGA = 7.29e-5  # rad/s ? AUSTAL annexe G note 53
    F25 = 6.1618e-5  # fc ? 25?N ? AUSTAL annexe G Eq G.3

    # ?? Constantes VDI 3783 Blatt 8 (LASAT 5.2.3) ???????????????????????????
    FW = 1.3  # Eq 5.2.115
    FV = 1.8  # Eq 5.2.116  (1.8 ? 1.6 ancien BUB)
    FU = 2.4  # Eq 5.2.117  (2.4 ? 2.2 ancien BUB)
    F_K = 2.0  # Eq 5.2.118 : F_{u,v,w} = 2
    ALPHA = 0.3  # d?croissance de ?  (table LASAT ?5.2.3)

    # ?? Bornes document?es de T_L (LASAT 5.2.2.4, Eq 5.2.108) ???????????????
    BETA = 1.0  # T_L_min = β·z₀/u★
    TMAX = 1200.0  # s ? borne sup?rieure globale

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        u_star: float,
        L_M: float,
        z0: float,
        device: str = "cuda",
        latitude_deg: float = 50.0,
    ):
        """
        Parameters
        ----------
        u_star      : friction velocity [m/s]
        L_M         : Monin-Obukhov length [m]  (negative = unstable)
        z0          : roughness length [m]
        device      : 'cuda' or 'cpu'
        latitude_deg: geographic latitude [°]  (default 50°N, central Germany)
                      Used for fc = 2·ω·sin(φ) [AUSTAL Annex G note 53]
        """
        self.u_star = float(u_star)
        self.L_M = float(L_M)
        self.z0 = float(z0)
        self.device = device

        # ?? Param?tre de Coriolis pour la latitude donn?e ?????????????????????
        # AUSTAL 3.3, Annex G note 53 : fc = 2·ω·sin(φ)   with  ω = 7.29×10⁻⁵ rad/s
        # note 54 : "Standardwert ist gemäß Richtlinie 1.1·10⁻⁴ 1/s"
        self.fc = self._compute_fc(latitude_deg)

        # ?? Seuil physique z? (LASAT 5.2.3, Eq 5.2.121) ??????????????????????
        # "For heights below 6z₀+d₀, the other profiles are set constant."
        # d? = 0 (pas de hauteur de d?placement dans ce mod?le)
        self.z_floor = 6.0 * self.z0  # [m]

        # ?? Hauteur de m?lange ????????????????????????????????????????????????
        self.h_m = self._compute_mixing_height()

        # ?? Borne inf?rieure document?e de T_L (LASAT Eq 5.2.108) ????????????
        # T_L_min = β · z₀ / u★   [s],  β = 1
        self.T_L_min = self.BETA * self.z0 / max(self.u_star, 1e-4)

        # ?? Vitesse convective w? (Deardorff 1970) ????????????????????????????
        if self.L_M < 0.0:
            self.w_star = (-(self.u_star**3) * self.h_m / (self.KAPPA * self.L_M)) ** (1.0 / 3.0)
        else:
            self.w_star = 0.0

        # ?? Param?tre de rotation du vent D_h ?????????????????????????????????
        self.D_h = self._compute_wind_turning_parameter()

    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def _compute_fc(cls, latitude_deg: float) -> float:
        """
        Coriolis parameter for an arbitrary latitude.

        AUSTAL 3.3, Annex G, note 53 :
            fc = 2·ω·sin(φ)   with  ω = 7.29×10⁻⁵ rad/s

        note 54 :
            "Standardwert ist gemäß Richtlinie 1.1·10⁻⁴ 1/s"
            (rounded value for 49–52°N)

        The value |fc| is used for hm. AUSTAL enforces
        |fc| ≥ f₂₅ = 6.1618×10⁻⁵ s⁻¹ (fc at ±25°) to avoid non-physical hm
        values at tropical latitudes. (Annex G, Eq G.3)
        """
        fc = 2.0 * cls.OMEGA * math.sin(math.radians(latitude_deg))
        return fc  # peut ?tre n?gatif dans l'h?misph?re sud (|fc| utilis? pour hm)

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_mixing_height(self) -> float:
        """
        Mixing height h_m.

        AUSTAL 3.3, Annex G, Eq G.1–G.4 + VDI 3783 Blatt 8, Eq 66.

        Unstable (L_M < 0) :
            h_m = h_m,i + Δ
            Δ = 300 m for classes IV and V (AUSTAL Annex G §5)
            Δ = 0 m for class III/2
            Source : "Für die Klassen IV und V sind die Standardwerte der
            Mischungsschichthöhe 1100 m" — AUSTAL Annex G §2.
            Implementation : hm=1100 m for L_M<0 (default value for central
            Germany, latitude 49-52°N). For non-standard latitudes → h_m,i + Δ.

        Stable (L_M > 0) :
            Zilitinkevich formula (VDI 3783 Blatt 8, Eq 66) :
              if L_M ≥ 2·u★/fc : h_m = 0.3·u★/fc  [asymptotic]
              if L_M <  2·u★/fc : h_m = 0.3·(u★/fc) / √(1 + 1.9·u★/(fc·L_M))
            Capped at h_m,i (Eq G.4) — replaces the fixed 800 m for non-standard
            latitudes.

        Neutral (L_M → ∞) :
            h_m = min(h_m,i, 0.3·u★/fc)  (Eq G.1)
            h_m,i = 800·(1.1×10⁻⁴)/max(f₂₅, |fc|)  (Eq G.4)
        """
        fc_abs = max(abs(self.fc), self.F25)  # |fc| ≥ f₂₅  (Eq G.3)

        # h_m,i : remplace les 800 m fixes pour les latitudes non standards  (Eq G.4)
        # AUSTAL : "800 m · (1.1×10⁻⁴) / max(f₂₅, |fc|)"
        hm_i = 800.0 * self.FC / fc_abs

        if self.L_M < 0.0:
            # Instable : 1100 m pour l'Allemagne centrale (AUSTAL annexe G ?2)
            # Pour les autres latitudes : hm_i + 300 m (? classes IV/V, annexe G ?5)
            if abs(self.fc - self.FC) / self.FC < 1e-5:
                return 1100.0  # latitude standard : valeur tabul?e
            return hm_i + 300.0  # latitude non standard

        if self.L_M > 0.0:
            # Stable : Zilitinkevich (VDI 3783 Blatt 8, Eq 66)
            u_over_fc = self.u_star / fc_abs
            if self.L_M >= 2.0 * u_over_fc:
                hm = 0.3 * u_over_fc
            else:
                hm = 0.3 * u_over_fc / math.sqrt(1.0 + 1.9 * self.u_star / (fc_abs * self.L_M))
            return min(hm_i, hm)

        # Neutre pur (L_M = 0) : Eq G.1
        return min(hm_i, 0.3 * self.u_star / fc_abs)

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_wind_turning_parameter(self) -> float:
        """
        Maximum wind rotation angle D_h [°].

        Source : TA Luft Annex 3, Table 5 / LASAT 5.2.3, Eq 5.2.114 :
            hm/L < −10          → D_h = 0°
            −10 ≤ hm/L < 0     → D_h = 45 + 4.5·hm/L  ∈ [0°, 45°]
            L > 0               → D_h = 45°
            Pure neutral        → D_h = 0°
        """
        if self.L_M > 0.0:
            return 45.0
        if self.L_M < 0.0:
            ratio = self.h_m / self.L_M  # < 0
            if ratio < -10.0:
                return 0.0
            return max(0.0, 45.0 + 4.5 * ratio)
        return 0.0  # neutre pur

    # ──────────────────────────────────────────────────────────────────────────
    def get_wind_direction_turning(self, z: torch.Tensor) -> torch.Tensor:
        """
        Angular deviation D(z) of wind direction [°].

        LASAT 5.2.3, Eq 5.2.114  /  TA Luft Annex 3, Eq 2 :
            D(z) = 1.23 · D_h · [1 − exp(−1.75 · z / h_m)]
        """
        z_safe = torch.clamp(z, max=self.h_m)
        return 1.23 * self.D_h * (1.0 - torch.exp(-1.75 * z_safe / self.h_m))

    # ──────────────────────────────────────────────────────────────────────────
    def get_wind_direction(self, z: torch.Tensor, r_a: float, h_a: float) -> torch.Tensor:
        """
        Absolute wind direction r(z) [°].

        LASAT 5.2.3, Eq 5.2.113  /  TA Luft Annex 3, Eq 1 :
            r(z) = r_a + D(z) − D(h_a)
        """
        D_z = self.get_wind_direction_turning(z)
        h_a_s = min(h_a, self.h_m)
        D_ha = 1.23 * self.D_h * (1.0 - math.exp(-1.75 * h_a_s / self.h_m))
        return r_a + D_z - D_ha

    # ──────────────────────────────────────────────────────────────────────────
    def _dissipation_rate(self, z_safe: torch.Tensor) -> torch.Tensor:
        """
        Turbulent kinetic energy dissipation rate ε.

        LASAT 5.2.3, Eq 5.2.119 :
            ε = u★³/(κz) · { (1 + 4z/L)·exp(−6α·z/hm)      for L > 0
                            { max[g(z), 1]                    for L < 0

            g(z) = (1 − z/hm)² + z/hm + (−z/L)·[1.5 − 1.3·(z/hm)^(1/3)]

        With κ = 0.40, α = 0.30.
        Regularization : z_safe ≥ z_floor → ε > 0 guaranteed.
        """
        z_h = z_safe / self.h_m
        factor = (self.u_star**3) / (self.KAPPA * z_safe)

        if self.L_M >= 0.0:
            # Stable / neutre : (1 + 4z/L) ? exp(?6??z/hm)
            # Pour un neutre pur (L_M = 0) : le terme 4z/L ? 0 ? eps = u??/(?z)?exp(?6??z/hm)
            if self.L_M > 0.0:
                stab_term = 1.0 + 4.0 * z_safe / self.L_M
            else:
                stab_term = torch.ones_like(z_safe)
            eps = factor * stab_term * torch.exp(-6.0 * self.ALPHA * z_h)
        else:
            # Instable : max[g(z), 1]
            g = (
                (1.0 - z_h) ** 2
                + z_h
                + (-z_safe / self.L_M) * (1.5 - 1.3 * z_h.clamp(min=1e-8) ** (1.0 / 3.0))
            )
            eps = factor * torch.clamp(g, min=1.0)

        return eps.clamp(min=1e-8)

    # ──────────────────────────────────────────────────────────────────────────
    def get_turbulence_params(self, z: torch.Tensor):
        """
        VDI 3783 Blatt 8 turbulence profiles for the Langevin equation.
        Equivalent to LASAT boundary layer version 2.6 (LASAT 5.2.3).

        Input parameter
        ───────────────
        z : (N,) heights above ground [m]

        Returns
        ───────
        sigma_u     (N,) [m/s]    longitudinal fluctuation
        sigma_v     (N,) [m/s]    transverse fluctuation
        sigma_w     (N,) [m/s]    vertical fluctuation
        T_Lu        (N,) [s]      longitudinal Lagrangian timescale
        T_Lv        (N,) [s]      transverse Lagrangian timescale
        T_Lw        (N,) [s]      vertical Lagrangian timescale
        d_var_w_dz  (N,) [m/s²/m] gradient ∂σ_w²/∂z (Thomson WMC drift)

        VDI 3783 Blatt 8 profiles (LASAT 5.2.3, Eq 5.2.115–5.2.122)
        ─────────────────────────────────────────────────────────────
        Stable / neutral (L ≥ 0) :
            σ_w = fw·u★·exp(−z/hm)
            σ_v = fv·u★·exp(−z/hm)
            σ_u = fu·u★·exp(−z/hm)

        Unstable (L < 0) :
            σ_w = fw·u★·[exp(−3z/hm) + 2.5·(−z/L)·(1−0.8z/hm)³]^(1/3)
            σ_v = fv·u★·[1 + 0.0880·(−hm/L)]^(1/3)·exp(−z/hm)
            σ_u = fu·u★·[1 + 0.0371·(−hm/L)]^(1/3)·exp(−z/hm)

        K diffusion (LASAT 5.2.3, Eq 5.2.118) :
            K = F·σ⁴/(5.7·ε)   with F = 2.0

        T_L = K/σ² = F·σ²/(5.7·ε)  (LASAT 5.2.2.4, Eq 5.2.107)

        Floor z_floor = 6·z₀ (LASAT 5.2.3, Eq 5.2.121) :
            Constant profiles for z ≤ 6·z₀.
            d_var_w_dz = 0 in this region (zero gradient for constant profile).

        T_L bounds (LASAT 5.2.2.4, Eq 5.2.108) :
            β·z₀/u★ ≤ T_L ≤ 1200 s   with β = 1.

        Note on d_var_w_dz
        ──────────────────
        Used EXCLUSIVELY for the Thomson WMC drift correction :
            a_w = −w'/T_L + 0.5 · ∂σ_w²/∂z · (1 + w'²/σ_w²)
        Analytical derivative of σ_w²(z) with respect to z.
        For z ≤ 6·z₀ : d_var_w_dz = 0  (constant profile, zero gradient).

        Guard above h_m
        ───────────────
        VDI 3783 Blatt 8 defines σ and T_L in [z_floor, h_m] only.
        For z > h_m : σ → 1×10⁻⁴ m/s (non-zero floor),  d_var_w_dz → 0.
        """
        hm = self.h_m
        zf = self.z_floor  # 6·z₀
        ustr = self.u_star

        # ?? z_eff : hauteur effective d'?valuation du profil ??????????????????
        # Profil ?valu? en max(z, z_floor), mais le seuil sera appliqu? a posteriori.
        # On garantit z_eff ? [z_floor, 0.9999?hm] pour ?viter les singularit?s internes.
        z_eff = z.clamp(min=zf, max=hm * 0.9999)
        z_h = z_eff / hm  # z/hm ∈ [0, 1)

        # ─────────────────────────────────────────────────────────────────────
        # R?GIME STABLE / NEUTRE  (L_M ? 0)
        # LASAT 5.2.3, Eq 5.2.115–5.2.117  (L > 0 or L = 0)
        # ─────────────────────────────────────────────────────────────────────
        if self.L_M >= 0.0:
            # Att?nuation verticale : exp(?z/hm)
            att = torch.exp(-z_h)  # att ∈ (exp(−1), 1]

            sigma_w = (self.FW * ustr * att).clamp(min=1e-4)
            sigma_v = (self.FV * ustr * att).clamp(min=1e-4)
            sigma_u = (self.FU * ustr * att).clamp(min=1e-4)

            # ??_w?/?z ? d?riv?e analytique de ?_w? = (fw?u?)??exp(?2z/hm)
            # d/dz = −2/hm · (fw·u★)² · exp(−2z/hm) = −2/hm · σ_w²
            # Pas de singularit? : r?gulier en z=0 (?_w(0) = fw?u? fini)
            d_var_w_dz = -(2.0 / hm) * sigma_w**2

        # ─────────────────────────────────────────────────────────────────────
        # R?GIME INSTABLE  (L_M < 0)
        # LASAT 5.2.3, Eq 5.2.115–5.2.117
        # ─────────────────────────────────────────────────────────────────────
        else:
            L_abs = abs(self.L_M)

            # ?_w : terme mixte m?canique-convectif
            # B(z) = exp(−3z/hm) + 2.5·(z/|L|)·(1−0.8z/hm)³  ≥ 0
            # σ_w = fw·u★·B^(1/3)
            B = (
                torch.exp(-3.0 * z_h)
                + 2.5 * (z_eff / L_abs) * (1.0 - 0.8 * z_h).clamp(min=0.0) ** 3
            ).clamp(min=1e-8)

            sigma_w = (self.FW * ustr * B ** (1.0 / 3.0)).clamp(min=1e-4)

            # ?_v, ?_u : att?nuation exp(?z/hm) avec renforcement convectif
            # σ_v = fv·u★·[1 + 0.0880·(hm/|L|)]^(1/3)·exp(−z/hm)
            # σ_u = fu·u★·[1 + 0.0371·(hm/|L|)]^(1/3)·exp(−z/hm)
            conv_boost_v = (1.0 + 0.0880 * hm / L_abs) ** (1.0 / 3.0)
            conv_boost_u = (1.0 + 0.0371 * hm / L_abs) ** (1.0 / 3.0)
            att = torch.exp(-z_h)

            sigma_v = (self.FV * ustr * conv_boost_v * att).clamp(min=1e-4)
            sigma_u = (self.FU * ustr * conv_boost_u * att).clamp(min=1e-4)

            # ??_w?/?z ? d?riv?e analytique de ?_w? = (fw?u?)??B^(2/3)
            # d/dz = (fw·u★)²·(2/3)·B^(−1/3)·∂B/∂z
            #
            # ∂B/∂z = −3/hm·exp(−3z/hm)
            #        + 2.5/|L|·(1−0.8z/hm)³
            #        + 2.5·(z/|L|)·3·(1−0.8z/hm)²·(−0.8/hm)
            #
            # Pas de singularit? : B(0) = 1, B^(?1/3)(0) = 1.
            conv_arg = (1.0 - 0.8 * z_h).clamp(min=0.0)
            d_B_dz = (
                -3.0 / hm * torch.exp(-3.0 * z_h)
                + 2.5 / L_abs * conv_arg**3
                - 2.5 * (z_eff / L_abs) * 3.0 * conv_arg**2 * (0.8 / hm)
            )
            d_var_w_dz = (self.FW * ustr) ** 2 * (2.0 / 3.0) * B ** (-1.0 / 3.0) * d_B_dz

        # ─────────────────────────────────────────────────────────────────────
        # DIFFUSION K ET ?CHELLE DE TEMPS LAGRANGIENNE T_L
        # LASAT 5.2.3, Eq 5.2.118–5.2.119  +  5.2.2.4, Eq 5.2.107–5.2.108
        # K = F·σ⁴/(5.7·ε)   with F=2
        # T_L = K/σ² = F·σ²/(5.7·ε)
        # β·z₀/u★ ≤ T_L ≤ Tmax = 1200 s
        # ─────────────────────────────────────────────────────────────────────
        eps = self._dissipation_rate(z_eff)  # (N,)

        # Coefficient K/?? = F???/(5.7??) ? calcul? s?par?ment pour chaque composante.
        def _T_L(sigma: torch.Tensor) -> torch.Tensor:
            """T_L = F·σ²/(5.7·ε)  clamped in [T_L_min, Tmax]."""
            T = self.F_K * sigma**2 / (5.7 * eps)
            return T.clamp(min=self.T_L_min, max=self.TMAX)

        T_Lw = _T_L(sigma_w)
        T_Lv = _T_L(sigma_v)
        T_Lu = _T_L(sigma_u)

        # ─────────────────────────────────────────────────────────────────────
        # PLANCHER z_floor = 6?z?  (LASAT 5.2.3, Eq 5.2.121)
        # "For heights below 6z₀, the other profiles are set constant."
        # ? remplac? par la valeur ?valu?e en z_floor.
        # ─────────────────────────────────────────────────────────────────────
        below_floor = z < zf  # (N,) bool?en sur le z original (non tronqu?)

        if below_floor.any():
            # ?valuation de tous les profils en z_floor (scalaire ? diffusion)
            z_f_t = torch.full_like(z, zf)
            sig_w_f, sig_v_f, sig_u_f, T_Lw_f, T_Lv_f, T_Lu_f, _ = self._eval_at_z(z_f_t)

            sigma_w = torch.where(below_floor, sig_w_f, sigma_w)
            sigma_v = torch.where(below_floor, sig_v_f, sigma_v)
            sigma_u = torch.where(below_floor, sig_u_f, sigma_u)
            T_Lw = torch.where(below_floor, T_Lw_f, T_Lw)
            T_Lv = torch.where(below_floor, T_Lv_f, T_Lv)
            T_Lu = torch.where(below_floor, T_Lu_f, T_Lu)
            # Profil constant ? gradient nul ? d?rive WMC = 0
            d_var_w_dz = torch.where(below_floor, torch.zeros_like(d_var_w_dz), d_var_w_dz)

        # ─────────────────────────────────────────────────────────────────────
        # GARDE AU-DESSUS DE h_m
        # VDI 3783 Blatt 8 d?finit les profils uniquement dans [z_floor, h_m].
        # Pour z > h_m : turbulence nulle, d?rive WMC nulle.
        # ? ? 1?10?? m/s (plancher non nul pour que T_L_eff = K_tot/??_tot reste
        # d?fini m?me si extra_turb = 0 dans solver.py).
        # ─────────────────────────────────────────────────────────────────────
        above_hm = z > hm
        floor_sig = torch.full_like(sigma_w, 1e-4)
        zero_dvdz = torch.zeros_like(d_var_w_dz)

        sigma_u = torch.where(above_hm, floor_sig, sigma_u)
        sigma_v = torch.where(above_hm, floor_sig, sigma_v)
        sigma_w = torch.where(above_hm, floor_sig, sigma_w)
        d_var_w_dz = torch.where(above_hm, zero_dvdz, d_var_w_dz)
        # T_L inchang? au-dessus de h_m : avec ??1e-4, b_w?O(10??)  m/s?s??/?

        return sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw, d_var_w_dz

    # ──────────────────────────────────────────────────────────────────────────
    def _eval_at_z(self, z: torch.Tensor):
        """
        Internal profile evaluation at an arbitrary height z, WITHOUT
        applying the z_floor guard or the above_hm guard.
        Used by get_turbulence_params to compute the floor value.
        """
        hm = self.h_m
        ustr = self.u_star

        z_eff = z.clamp(min=self.z_floor, max=hm * 0.9999)
        z_h = z_eff / hm

        if self.L_M >= 0.0:
            att = torch.exp(-z_h)
            sigma_w = (self.FW * ustr * att).clamp(min=1e-4)
            sigma_v = (self.FV * ustr * att).clamp(min=1e-4)
            sigma_u = (self.FU * ustr * att).clamp(min=1e-4)
            d_var = -(2.0 / hm) * sigma_w**2
        else:
            L_abs = abs(self.L_M)
            B = (
                torch.exp(-3.0 * z_h)
                + 2.5 * (z_eff / L_abs) * (1.0 - 0.8 * z_h).clamp(min=0.0) ** 3
            ).clamp(min=1e-8)
            sigma_w = (self.FW * ustr * B ** (1.0 / 3.0)).clamp(min=1e-4)
            conv_boost_v = (1.0 + 0.0880 * hm / L_abs) ** (1.0 / 3.0)
            conv_boost_u = (1.0 + 0.0371 * hm / L_abs) ** (1.0 / 3.0)
            att = torch.exp(-z_h)
            sigma_v = (self.FV * ustr * conv_boost_v * att).clamp(min=1e-4)
            sigma_u = (self.FU * ustr * conv_boost_u * att).clamp(min=1e-4)
            conv_arg = (1.0 - 0.8 * z_h).clamp(min=0.0)
            d_B_dz = (
                -3.0 / hm * torch.exp(-3.0 * z_h)
                + 2.5 / L_abs * conv_arg**3
                - 2.5 * (z_eff / L_abs) * 3.0 * conv_arg**2 * (0.8 / hm)
            )
            d_var = (self.FW * ustr) ** 2 * (2.0 / 3.0) * B ** (-1.0 / 3.0) * d_B_dz

        eps = self._dissipation_rate(z_eff)

        def _T_L(sig):
            return (self.F_K * sig**2 / (5.7 * eps)).clamp(min=self.T_L_min, max=self.TMAX)

        return (
            sigma_u,
            sigma_v,
            sigma_w,
            _T_L(sigma_u),
            _T_L(sigma_v),
            _T_L(sigma_w),
            d_var,
        )
