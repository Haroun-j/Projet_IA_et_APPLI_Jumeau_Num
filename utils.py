"""
Shared utility functions for the DeepSDF airport project.
"""

import json
import os

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_normalization_params(dataset_dir: str):
    """Load normalization center and scale saved by sdf_sampling.py."""
    path = os.path.join(dataset_dir, "normalization.json")
    with open(path) as f:
        params = json.load(f)
    center = np.array(params["center"], dtype=np.float32)
    scale = float(params["scale"])
    return center, scale


def normalize_coords(pts: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Map world coordinates → [-1, 1]."""
    return (pts - center) / scale


def denormalize_coords(pts_norm: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Map normalized coordinates → world coordinates."""
    return pts_norm * scale + center


def get_device(preferred: str = "cuda") -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
