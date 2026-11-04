"""Autoencoder anomaly-detection model, training, and inference pipeline.

Frames fraud detection as nonlinear manifold reconstruction:
Trained strictly on legitimate transactions, using reconstruction error
(MSE) as the anomaly score.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TransactionAutoencoder(nn.Module):
    """Undercomplete / Deep Autoencoder for tabular transaction feature reconstruction."""

    def __init__(
        self,
        input_dim: int = 29,
        hidden_dims: Tuple[int, ...] = (20, 12),
        latent_dim: int = 6,
        dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.1))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.1))
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample mean squared reconstruction error (MSE)."""
        reconstructed = self.forward(x)
        # Per-sample MSE across all feature dimensions
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


def train_autoencoder(
    train_features: np.ndarray,
    val_features: np.ndarray,
    input_dim: int = 29,
    hidden_dims: Tuple[int, ...] = (20, 12),
    latent_dim: int = 6,
    batch_size: int = 256,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    noise_factor: float = 0.0,
    seed: int = 42,
    device: Optional[str] = None,
) -> Tuple[TransactionAutoencoder, Dict[str, List[float]]]:
    """Train autoencoder strictly on legitimate features with early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransactionAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
    ).to(device)

    train_tensor = torch.tensor(train_features, dtype=torch.float32)
    val_tensor = torch.tensor(val_features, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    patience = 7
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            if noise_factor > 0.0:
                noisy_x = batch_x + noise_factor * torch.randn_like(batch_x)
                output = model(noisy_x)
            else:
                output = model(batch_x)

            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_tensor)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_x = val_tensor.to(device)
            val_recon = model(val_x)
            epoch_val_loss = criterion(val_recon, val_x).item()

        scheduler.step(epoch_val_loss)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.to("cpu")
    model.eval()

    return model, history


def score_transactions_ae(
    transactions: pd.DataFrame,
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    """Score transactions using fitted Autoencoder bundle.
    
    Returns DataFrame with columns:
      - reconstruction_error
      - anomaly_score (same as reconstruction_error)
      - is_anomaly (boolean based on calibrated threshold)
    """
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame")
    if transactions.empty:
        raise ValueError("transactions must contain at least one row")

    feature_names = bundle["feature_names"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    threshold = float(bundle["threshold"])

    supplied_columns = [col for col in transactions.columns if col != "Class"]
    if supplied_columns != feature_names:
        raise ValueError("Features do not match expected feature_names order")

    features = transactions.loc[:, feature_names].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values")

    scaled = scaler.transform(features)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        errors = model.compute_reconstruction_error(tensor).numpy().astype(float)

    return pd.DataFrame(
        {
            "reconstruction_error": errors,
            "anomaly_score": errors,
            "is_anomaly": errors >= threshold,
        },
        index=transactions.index,
    )

