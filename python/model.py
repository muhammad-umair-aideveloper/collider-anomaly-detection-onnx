"""Autoencoder model with internal standardization for collider anomaly detection.

This module implements a symmetric PyTorch autoencoder (28 -> 16 -> 8 -> 16 -> 28)
with standardization parameters registered as model buffers. Input features are
standardized directly inside forward passes, allowing raw detector features to be
consumed directly without external preprocessing pipelines.
"""

import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config


class Autoencoder(nn.Module):
    """Symmetric Autoencoder with internal input standardization.

    Architecture:
        Input: 28 features (raw)
        Encoder: Linear(28, 16) -> ReLU -> Linear(16, 8) -> ReLU (latent dimension: 8)
        Decoder: Linear(8, 16) -> ReLU -> Linear(16, 28) (linear output)

    Standardization:
        x_norm = (x - mean) / (std + eps)
        Mean and standard deviation are registered as persistent buffers.
    """

    def __init__(
        self,
        layers: Tuple[int, ...] = config.AUTOENCODER_LAYERS,
        epsilon: float = config.STANDARDIZATION_EPSILON,
    ) -> None:
        """Initialize Autoencoder architecture and standardization buffers.

        Args:
            layers: Layer dimension tuple (e.g. (28, 16, 8, 16, 28)).
            epsilon: Small constant added to denominator to avoid division by zero.
        """
        super().__init__()
        if len(layers) != 5:
            raise ValueError(
                f"Expected 5 layer dimensions (input, h1, latent, h2, output), got {len(layers)}."
            )

        in_dim, h1_dim, latent_dim, h2_dim, out_dim = layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = float(epsilon)

        # Register persistent standardization buffers
        self.register_buffer("mean", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(in_dim, dtype=torch.float32))

        # Encoder network: 28 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder network: 8 -> 16 -> 28 (linear output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, out_dim),
        )

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize raw features using internal mean and standard deviation buffers.

        Args:
            x: Raw input tensor of shape [..., 28].

        Returns:
            torch.Tensor: Standardized tensor of identical shape.
        """
        return (x - self.mean) / (self.std + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize raw input and reconstruct in standardized feature space.

        Args:
            x: Raw input tensor of shape [N, 28].

        Returns:
            torch.Tensor: Reconstructed features of shape [N, 28] in standardized space.
        """
        x_norm = self.standardize(x)
        latent = self.encoder(x_norm)
        reconstruction = self.decoder(latent)
        return reconstruction

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate per-event mean squared reconstruction error in standardized space.

        Args:
            x: Raw input tensor of shape [N, 28].

        Returns:
            torch.Tensor: Per-event MSE error of shape [N].
        """
        x_norm = self.standardize(x)
        latent = self.encoder(x_norm)
        reconstruction = self.decoder(latent)
        # Mean squared error over the 28 feature dimensions
        squared_diff = (x_norm - reconstruction) ** 2
        return torch.mean(squared_diff, dim=-1)

    def set_standardization(
        self, train_features: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Compute and set mean and std buffers from training set features only.

        Args:
            train_features: Feature array or tensor of shape [N, 28] from training set.

        Raises:
            ValueError: If feature dimension does not match in_dim.
        """
        if isinstance(train_features, np.ndarray):
            tensor_feat = torch.from_numpy(train_features).float()
        elif isinstance(train_features, torch.Tensor):
            tensor_feat = train_features.float()
        else:
            raise TypeError(
                f"Expected numpy.ndarray or torch.Tensor, got {type(train_features)}"
            )

        if tensor_feat.ndim != 2 or tensor_feat.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected feature shape [N, {self.in_dim}], got {list(tensor_feat.shape)}"
            )

        mean_val = torch.mean(tensor_feat, dim=0)
        std_val = torch.std(tensor_feat, dim=0, unbiased=False)

        # Ensure no negative or zero std values are stored
        std_val = torch.clamp(std_val, min=self.epsilon)

        self.mean.copy_(mean_val)
        self.std.copy_(std_val)
