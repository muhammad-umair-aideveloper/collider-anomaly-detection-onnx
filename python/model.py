"""Autoencoder model with internal standardization and optional in-model feature selection.

This module implements a symmetric PyTorch autoencoder with in-model feature selection
and persistent standardization buffers. When feature selection is configured, raw
detector events of shape [N, 28] are accepted; the selected feature subset is extracted
and standardized internally.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config

# Pre-declared model variants specification
PRE_DECLARED_VARIANTS: Dict[str, Dict[str, Union[str, Tuple[int, ...], Optional[Tuple[int, ...]]]]] = {
    "V0": {
        "description": "Baseline autoencoder using all 28 features (latent bottleneck: 8)",
        "selected_features": None,
        "bottleneck_dim": 8,
        "layers": (28, 16, 8, 16, 28),
    },
    "V1": {
        "description": "Autoencoder excluding 6 azimuthal phi angles (22 features, bottleneck: 8)",
        "selected_features": config.NON_PHI_FEATURE_INDICES,
        "bottleneck_dim": 8,
        "layers": (22, 16, 8, 16, 22),
    },
    "V2": {
        "description": "Autoencoder using 7 high-level invariant mass features (bottleneck: 4)",
        "selected_features": config.HIGH_LEVEL_FEATURE_INDICES,
        "bottleneck_dim": 4,
        "layers": (7, 16, 4, 16, 7),
    },
    "V3": {
        "description": "Autoencoder excluding phi with compressed bottleneck (22 features, bottleneck: 4)",
        "selected_features": config.NON_PHI_FEATURE_INDICES,
        "bottleneck_dim": 4,
        "layers": (22, 16, 4, 16, 22),
    },
}


class Autoencoder(nn.Module):
    """Symmetric Autoencoder with internal feature selection and input standardization.

    The model consumes raw 28-dimensional collision events [N, 28]. If a feature subset
    is configured, the selection is performed inside the forward pass via registered buffers,
    preserving full input interface compatibility for ONNX export and downstream C++ inference.
    """

    def __init__(
        self,
        layers: Optional[Tuple[int, ...]] = None,
        selected_features: Optional[Union[List[int], Tuple[int, ...]]] = None,
        bottleneck_dim: Optional[int] = None,
        epsilon: float = config.STANDARDIZATION_EPSILON,
    ) -> None:
        """Initialize Autoencoder architecture and standardization buffers.

        Args:
            layers: Optional explicit 5-element tuple (in, h1, latent, h2, out).
            selected_features: Optional sequence of feature indices in [0, 27].
            bottleneck_dim: Optional latent dimension override (default: 4 if <12 features, else 8).
            epsilon: Numerical stability constant for standardization denominator.
        """
        super().__init__()
        self.raw_input_dim: int = config.NUM_FEATURES
        self.epsilon: float = float(epsilon)

        # Configure feature selection buffer
        if selected_features is not None:
            indices_list = [int(i) for i in selected_features]
            for idx in indices_list:
                if idx < 0 or idx >= self.raw_input_dim:
                    raise ValueError(
                        f"Feature index {idx} out of valid bounds [0, {self.raw_input_dim - 1}]."
                    )
            self.has_feature_selection = True
            indices_tensor = torch.tensor(indices_list, dtype=torch.int64)
            num_selected = len(indices_list)
        else:
            self.has_feature_selection = False
            indices_tensor = torch.arange(self.raw_input_dim, dtype=torch.int64)
            num_selected = self.raw_input_dim

        self.register_buffer("selected_indices", indices_tensor)
        self.num_selected: int = num_selected

        # Determine architecture dimensions
        if layers is not None:
            if len(layers) != 5:
                raise ValueError(
                    f"Expected 5 layer dimensions (in, h1, latent, h2, out), got {len(layers)}."
                )
            if layers[0] != num_selected or layers[4] != num_selected:
                raise ValueError(
                    f"Layer endpoints ({layers[0]}, {layers[4]}) do not match selected features count {num_selected}."
                )
            self.layers = layers
        else:
            if bottleneck_dim is not None:
                b_dim = int(bottleneck_dim)
            else:
                b_dim = 4 if num_selected < 12 else 8
            self.layers = (num_selected, 16, b_dim, 16, num_selected)

        in_dim, h1_dim, latent_dim, h2_dim, out_dim = self.layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.bottleneck_dim = latent_dim

        # Register persistent standardization buffers for selected features
        self.register_buffer("mean", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(in_dim, dtype=torch.float32))

        # Encoder network: in_dim -> 16 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder network: latent_dim -> 16 -> out_dim (linear output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, out_dim),
        )

    def select_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract configured feature subset from raw input tensor.

        Args:
            x: Input tensor of shape [..., 28] or [..., num_selected].

        Returns:
            torch.Tensor: Feature subset tensor of shape [..., num_selected].
        """
        if x.shape[-1] == self.raw_input_dim:
            return torch.index_select(x, dim=-1, index=self.selected_indices)
        elif x.shape[-1] == self.in_dim:
            return x
        else:
            raise ValueError(
                f"Expected last dimension {self.raw_input_dim} or {self.in_dim}, got {x.shape[-1]}."
            )

    def standardize(self, x_selected: torch.Tensor) -> torch.Tensor:
        """Standardize selected features using internal mean and std buffers.

        Args:
            x_selected: Tensor of shape [..., in_dim].

        Returns:
            torch.Tensor: Standardized tensor of shape [..., in_dim].
        """
        return (x_selected - self.mean) / (self.std + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process raw input features and produce reconstruction in standardized space.

        Args:
            x: Raw input tensor of shape [N, 28].

        Returns:
            torch.Tensor: Reconstructed features of shape [N, in_dim].
        """
        x_selected = self.select_features(x)
        x_norm = self.standardize(x_selected)
        latent = self.encoder(x_norm)
        reconstruction = self.decoder(latent)
        return reconstruction

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate per-event mean squared reconstruction error over selected features.

        Args:
            x: Raw input tensor of shape [N, 28].

        Returns:
            torch.Tensor: Per-event MSE error of shape [N].
        """
        x_selected = self.select_features(x)
        x_norm = self.standardize(x_selected)
        latent = self.encoder(x_norm)
        reconstruction = self.decoder(latent)
        # Mean squared error over the selected feature dimensions
        squared_diff = (x_norm - reconstruction) ** 2
        return torch.mean(squared_diff, dim=-1)

    def set_standardization(
        self, train_features: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Compute and set mean and std buffers from training set features only.

        Args:
            train_features: Feature array or tensor of shape [N, 28] or [N, in_dim].

        Raises:
            ValueError: If feature dimension is incompatible.
        """
        if isinstance(train_features, np.ndarray):
            tensor_feat = torch.from_numpy(train_features).float()
        elif isinstance(train_features, torch.Tensor):
            tensor_feat = train_features.float()
        else:
            raise TypeError(
                f"Expected numpy.ndarray or torch.Tensor, got {type(train_features)}"
            )

        if tensor_feat.ndim != 2:
            raise ValueError(f"Expected 2D array [N, features], got {list(tensor_feat.shape)}")

        # Extract selected features if full 28-feature array is provided
        if tensor_feat.shape[1] == self.raw_input_dim:
            selected_data = tensor_feat[:, self.selected_indices]
        elif tensor_feat.shape[1] == self.in_dim:
            selected_data = tensor_feat
        else:
            raise ValueError(
                f"Expected feature count {self.raw_input_dim} or {self.in_dim}, got {tensor_feat.shape[1]}."
            )

        mean_val = torch.mean(selected_data, dim=0)
        std_val = torch.std(selected_data, dim=0, unbiased=False)

        # Clamp standard deviation to avoid division by zero
        std_val = torch.clamp(std_val, min=self.epsilon)

        self.mean.copy_(mean_val)
        self.std.copy_(std_val)

    @classmethod
    def from_variant(cls, variant_key: str) -> "Autoencoder":
        """Instantiate pre-declared autoencoder variant (V0, V1, V2, V3).

        Args:
            variant_key: Key in PRE_DECLARED_VARIANTS ('V0', 'V1', 'V2', 'V3').

        Returns:
            Autoencoder: Configured model instance.
        """
        if variant_key not in PRE_DECLARED_VARIANTS:
            raise KeyError(
                f"Unknown variant '{variant_key}'. Available: {list(PRE_DECLARED_VARIANTS.keys())}."
            )
        spec = PRE_DECLARED_VARIANTS[variant_key]
        return cls(
            layers=spec["layers"],
            selected_features=spec["selected_features"],
            bottleneck_dim=spec["bottleneck_dim"],
        )
