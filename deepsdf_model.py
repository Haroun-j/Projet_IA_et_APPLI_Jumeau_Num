"""
Architecture DeepSDF + Fourier Feature Mapping
==============================================================================
Architecture:
  (x, y, z) → FourierFeatureMapping → 7-layer MLP (512 neurons, skip@4) → SDF
"""

import math

import torch
import torch.nn as nn
import yaml


# ─────────────────────────────────────────────────────────────────────────────

class FourierFeatureMapping(nn.Module):
    """
    Encodage positionnel (Tancik et al., 2020 – "Fourier Features let Networks
    learn High Frequency Functions in Low Dimensional Domains").

    Pour chaque coordonnée p :
        γ(p) = [p,
                sin(2^0 π p), cos(2^0 π p),
                sin(2^1 π p), cos(2^1 π p),
                …
                sin(2^(L-1) π p), cos(2^(L-1) π p)]

    Dimension d'entrée : input_dim          (3 pour des coordonnées 3D)
    Dimension de sortie : input_dim + 2 * n_frequencies * input_dim
    """

    def __init__(self, n_frequencies: int = 8, input_dim: int = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.input_dim     = input_dim
        self.output_dim    = input_dim + 2 * n_frequencies * input_dim

        # Bandes de fréquence : [2^0 π, 2^1 π, …, 2^(L-1) π]
        freqs = torch.tensor(
            [2 ** i * math.pi for i in range(n_frequencies)], dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)  # (L,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, input_dim)
        # x_freq : (B, input_dim, L)
        x_freq = x.unsqueeze(-1) * self.freqs        # diffusion automatique
        sin_f  = torch.sin(x_freq).reshape(x.shape[0], -1)   # (B, input_dim*L)
        cos_f  = torch.cos(x_freq).reshape(x.shape[0], -1)   # (B, input_dim*L)
        return torch.cat([x, sin_f, cos_f], dim=-1)           # (B, output_dim)


# ─────────────────────────────────────────────────────────────────────────────

class Sine(nn.Module):
    """Activation sinus pour les réseaux de type SIREN."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


# ─────────────────────────────────────────────────────────────────────────────

class DeepSDFNetwork(nn.Module):
    """
    MLP DeepSDF avec Fourier Features et connexion de saut.

    Paramètres
    ----------
    n_frequencies     : nombre de bandes de fréquence dans l'encodage positionnel
    hidden_dim        : largeur de chaque couche cachée
    n_hidden_layers   : profondeur (8 dans l'article original)
    skip_connection_at: indice de couche (base 0) où l'entrée encodée est
                        concaténée à nouveau → force le réseau à mémoriser
                        la coordonnée globale
    activation        : "relu" ou "sine"
    """

    def __init__(
        self,
        n_frequencies:      int  = 8,
        hidden_dim:         int  = 512,
        n_hidden_layers:    int  = 8,
        skip_connection_at: int  = 4,
        activation:         str  = "relu",
    ):
        super().__init__()
        self.skip_at = skip_connection_at

        # Encodage positionnel
        self.fourier    = FourierFeatureMapping(n_frequencies, input_dim=3)
        enc_dim         = self.fourier.output_dim

        # Construit les couches cachées sous forme de deux ModuleLists pour que
        # la connexion de saut soit gérée explicitement dans forward()
        self.layers      = nn.ModuleList()
        self.activations = nn.ModuleList()

        in_dim = enc_dim
        for i in range(n_hidden_layers):
            if i == skip_connection_at:
                in_dim = hidden_dim + enc_dim     # concatène l'entrée encodée
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "sine":
                self.activations.append(Sine())
            else:
                self.activations.append(nn.ReLU(inplace=False))
            in_dim = hidden_dim

        self.output_layer = nn.Linear(hidden_dim, 1)

        self._init_weights(activation)

    # ── Initialisation des poids ──────────────────────────────────────────────
    def _init_weights(self, activation: str):
        for layer in self.layers:
            if activation == "sine":
                # Initialisation SIREN (Sitzmann et al., 2020)
                nn.init.uniform_(layer.weight, -1.0 / layer.in_features,
                                  1.0 / layer.in_features)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    # ── Propagation avant ─────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paramètres
        ----------
        x : (B, 3)   – coordonnées normalisées dans [-1, 1]

        Retourne
        -------
        sdf : (B, 1)  – distance signée prédite
        """
        enc = self.fourier(x)   # (B, enc_dim)
        h   = enc
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            if i == self.skip_at:
                h = torch.cat([h, enc], dim=-1)
            h = act(layer(h))
        return self.output_layer(h)   # (B, 1)

    @torch.no_grad()
    def predict_sdf(self, x: torch.Tensor) -> torch.Tensor:
        """Raccourci pratique — renvoie un tenseur (B,) sans gradient."""
        return self.forward(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> DeepSDFNetwork:
    """Instancie un DeepSDFNetwork à partir d'un dictionnaire de configuration."""
    m = cfg["model"]
    return DeepSDFNetwork(
        n_frequencies      = m["n_frequencies"],
        hidden_dim         = m["hidden_dim"],
        n_hidden_layers    = m["n_hidden_layers"],
        skip_connection_at = m["skip_connection_at"],
        activation         = m["activation"],
    )


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = DeepSDFNetwork()
    x     = torch.randn(1024, 3)
    out   = model(x)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print(f"Params : {n_p:,}")
    enc_dim = model.fourier.output_dim
    print(f"Fourier output dim : {enc_dim}")
