"""Unit tests for the Autoencoder model, internal standardization, and feature selection variants."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from model import PRE_DECLARED_VARIANTS, Autoencoder


@pytest.fixture
def sample_raw_batch():
    """Generate a synthetic batch of raw unscaled features."""
    torch.manual_seed(config.SEED)
    # Raw features with diverse scales and offsets simulating detector features
    scales = torch.tensor([10.0 ** (i % 3) for i in range(config.NUM_FEATURES)])
    offsets = torch.tensor([float(i) for i in range(config.NUM_FEATURES)])
    raw_data = torch.randn(32, config.NUM_FEATURES) * scales + offsets
    return raw_data


def test_model_output_shapes(sample_raw_batch):
    """Verify that forward and reconstruction_error return the expected tensor shapes."""
    model = Autoencoder()
    batch_size = sample_raw_batch.shape[0]

    reconstruction = model(sample_raw_batch)
    assert reconstruction.shape == (
        batch_size,
        config.NUM_FEATURES,
    ), f"Expected shape ({batch_size}, {config.NUM_FEATURES}), got {reconstruction.shape}"

    errors = model.reconstruction_error(sample_raw_batch)
    assert errors.shape == (
        batch_size,
    ), f"Expected shape ({batch_size},), got {errors.shape}"


def test_reconstruction_error_non_negative(sample_raw_batch):
    """Verify that per-event mean squared reconstruction error is strictly non-negative."""
    model = Autoencoder()
    errors = model.reconstruction_error(sample_raw_batch)

    assert torch.all(errors >= 0.0), "Encountered negative reconstruction error values."
    assert torch.all(torch.isfinite(errors)), "Encountered NaN or Inf in reconstruction errors."


def test_forward_and_backward_finite(sample_raw_batch):
    """Verify forward and backward gradients are finite without numerical instability."""
    model = Autoencoder()
    model.train()

    errors = model.reconstruction_error(sample_raw_batch)
    loss = errors.mean()

    assert torch.isfinite(loss), "Training loss is non-finite."

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for parameter {name}."
        assert torch.all(
            torch.isfinite(param.grad)
        ), f"Non-finite gradient in parameter {name}."


def test_standardization_affects_output(sample_raw_batch):
    """Verify that changing mean/std buffers modifies model output on identical raw inputs."""
    model = Autoencoder()
    model.eval()

    with torch.no_grad():
        initial_recon = model(sample_raw_batch).clone()
        initial_error = model.reconstruction_error(sample_raw_batch).clone()

    # Create synthetic training set with significant shift
    shifted_data = sample_raw_batch.numpy() * 5.0 + 100.0
    model.set_standardization(shifted_data)

    with torch.no_grad():
        updated_recon = model(sample_raw_batch)
        updated_error = model.reconstruction_error(sample_raw_batch)

    # Output and error must differ after setting realistic non-default standardization
    assert not torch.allclose(
        initial_recon, updated_recon, atol=1e-4
    ), "Reconstruction output did not change after updating standardization buffers."
    assert not torch.allclose(
        initial_error, updated_error, atol=1e-4
    ), "Reconstruction error did not change after updating standardization buffers."


def test_raw_features_with_zero_variance_protection():
    """Verify that near-zero variance features are guarded by epsilon."""
    model = Autoencoder()
    constant_features = np.ones((50, config.NUM_FEATURES), dtype=np.float32)

    # Constant features have std=0, buffer should clamp to epsilon
    model.set_standardization(constant_features)

    assert torch.all(
        model.std >= config.STANDARDIZATION_EPSILON
    ), "Standard deviation buffer fell below epsilon floor."

    test_input = torch.ones(5, config.NUM_FEATURES, dtype=torch.float32)
    selected_input = model.select_features(test_input)
    standardized = model.standardize(selected_input)
    assert torch.all(
        torch.isfinite(standardized)
    ), "Standardization produced non-finite values for constant features."


@pytest.mark.parametrize("variant_key", ["V0", "V1", "V2", "V3"])
def test_pre_declared_variants_shapes_and_execution(sample_raw_batch, variant_key):
    """Verify that all pre-declared variants accept raw [N, 28] features and execute cleanly."""
    model = Autoencoder.from_variant(variant_key)
    model.eval()

    spec = PRE_DECLARED_VARIANTS[variant_key]
    expected_dim = spec["layers"][0]
    expected_latent = spec["layers"][2]

    assert model.in_dim == expected_dim
    assert model.latent_dim == expected_latent
    assert model.mean.shape == (expected_dim,)
    assert model.std.shape == (expected_dim,)

    with torch.no_grad():
        reconstruction = model(sample_raw_batch)
        assert reconstruction.shape == (sample_raw_batch.shape[0], expected_dim)

        errors = model.reconstruction_error(sample_raw_batch)
        assert errors.shape == (sample_raw_batch.shape[0],)
        assert torch.all(errors >= 0.0)
        assert torch.all(torch.isfinite(errors))


def test_variant_feature_selection_and_training_pass(sample_raw_batch):
    """Verify end-to-end forward/backward passes and standardization on subset variants."""
    # Test high-level features only (V2: 7 features)
    model_v2 = Autoencoder.from_variant("V2")
    model_v2.train()

    # Ensure setting standardization directly from 28-column training data works
    synthetic_train_data = sample_raw_batch.numpy()
    model_v2.set_standardization(synthetic_train_data)
    assert model_v2.mean.shape == (7,)
    assert model_v2.std.shape == (7,)

    errors = model_v2.reconstruction_error(sample_raw_batch)
    loss = errors.mean()
    assert torch.isfinite(loss)

    loss.backward()
    for name, param in model_v2.named_parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad))
