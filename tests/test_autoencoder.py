"""Unit tests for TransactionAutoencoder architecture, training, and scoring."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from src.autoencoder import TransactionAutoencoder, train_autoencoder, score_transactions_ae


@pytest.fixture
def dummy_ae_bundle():
    np.random.seed(42)
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    training_data = np.random.randn(100, 29).astype(np.float32)

    scaler = StandardScaler().fit(training_data)
    model = TransactionAutoencoder(
        input_dim=29,
        hidden_dims=(16, 8),
        latent_dim=4,
    )
    return {
        "model_version": "1.0",
        "feature_names": feature_names,
        "scaler": scaler,
        "model": model,
        "threshold": 1.5,
    }


def test_autoencoder_forward_pass_dimensions():
    model = TransactionAutoencoder(input_dim=29, hidden_dims=(20, 12), latent_dim=6)
    x = torch.randn(16, 29)
    out = model(x)
    assert out.shape == (16, 29)

    mse = model.compute_reconstruction_error(x)
    assert mse.shape == (16,)
    assert (mse >= 0).all()


def test_score_transactions_ae_validates_columns_and_computes_anomalies(dummy_ae_bundle):
    feature_names = dummy_ae_bundle["feature_names"]
    normal_tx = pd.DataFrame([np.zeros(29)], columns=feature_names)
    extreme_tx = pd.DataFrame([np.ones(29) * 10.0], columns=feature_names)

    batch = pd.concat([normal_tx, extreme_tx], ignore_index=True)
    results = score_transactions_ae(batch, dummy_ae_bundle)

    assert "reconstruction_error" in results.columns
    assert "anomaly_score" in results.columns
    assert "is_anomaly" in results.columns
    assert len(results) == 2
    # Extreme transaction should have higher reconstruction error than zeros
    assert results.loc[1, "reconstruction_error"] > results.loc[0, "reconstruction_error"]


def test_score_transactions_ae_rejects_missing_columns(dummy_ae_bundle):
    bad_df = pd.DataFrame([[0.0] * 5], columns=["V1", "V2", "V3", "V4", "V5"])
    with pytest.raises(ValueError, match="Features do not match"):
        score_transactions_ae(bad_df, dummy_ae_bundle)

