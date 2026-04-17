"""
PatchTST-based time series forecasting for stock prices.

Implements the PatchTST architecture from "A Time Series is Worth 64 Words:
Long-term Forecasting with Transformers" (Nie et al., 2023).

Key components:
- Patching module that extracts patches from time series
- Channel-independent tokenization
- Transformer encoder for temporal pattern learning
- Linear head for multi-horizon forecasting
- Monte Carlo dropout for uncertainty estimation
"""

from __future__ import annotations

import copy
import logging
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Engineering Utilities
# ---------------------------------------------------------------------------

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute common technical indicators from OHLCV data.

    Expects columns: open, high, low, close, volume.
    Returns a DataFrame with additional indicator columns.
    """
    result = df.copy()

    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]

    # Moving averages
    result["sma_5"] = close.rolling(window=5, min_periods=1).mean()
    result["sma_10"] = close.rolling(window=10, min_periods=1).mean()
    result["sma_20"] = close.rolling(window=20, min_periods=1).mean()
    result["ema_12"] = close.ewm(span=12, adjust=False).mean()
    result["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    result["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma_20 = close.rolling(window=20, min_periods=1).mean()
    std_20 = close.rolling(window=20, min_periods=1).std().fillna(0)
    result["bb_upper"] = sma_20 + 2 * std_20
    result["bb_lower"] = sma_20 - 2 * std_20
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / (sma_20 + 1e-10)

    # ATR (Average True Range)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    result["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

    # Volume indicators
    result["vol_sma_10"] = volume.rolling(window=10, min_periods=1).mean()
    result["vol_ratio"] = volume / (result["vol_sma_10"] + 1e-10)

    # Returns
    result["log_return_1d"] = np.log(close / close.shift(1) + 1e-10)
    result["log_return_5d"] = np.log(close / close.shift(5) + 1e-10)

    # Stochastic oscillator
    low_14 = low.rolling(window=14, min_periods=1).min()
    high_14 = high.rolling(window=14, min_periods=1).max()
    result["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)

    # Fill NaN with 0 for initial periods
    result = result.fillna(0).replace([np.inf, -np.inf], 0)

    return result


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for time series forecasting.

    Parameters
    ----------
    series : np.ndarray
        2-D array of shape ``(n_timesteps, n_features)``.
    exog : np.ndarray or None
        2-D array of shape ``(n_timesteps, n_exog_features)``.
    context_length : int
        Number of historical steps used as input.
    forecast_horizon : int
        Number of future steps to predict.
    stride : int
        Step between consecutive windows.
    """

    def __init__(
        self,
        series: np.ndarray,
        exog: Optional[np.ndarray] = None,
        context_length: int = 96,
        forecast_horizon: int = 24,
        stride: int = 1,
    ) -> None:
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        total_len = context_length + forecast_horizon

        assert series.ndim == 2, f"series must be 2-D, got {series.ndim}-D"
        self.n_features = series.shape[1]

        if exog is not None:
            assert exog.ndim == 2
            assert exog.shape[0] == series.shape[0]
            self.n_exog = exog.shape[1]
        else:
            self.n_exog = 0

        # Build valid start indices
        self.start_indices: List[int] = list(
            range(0, series.shape[0] - total_len + 1, stride)
        )

        self.series = series.astype(np.float32)
        self.exog = exog.astype(np.float32) if exog is not None else None

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start = self.start_indices[idx]
        end = start + self.context_length

        x = self.series[start:end]  # (context_len, n_features)
        y = self.series[end : end + self.forecast_horizon, 0:1]  # predict close

        items: Tuple[torch.Tensor, ...] = (
            torch.from_numpy(x),
            torch.from_numpy(y),
        )

        if self.exog is not None:
            exog_x = self.exog[start:end]
            exog_y = self.exog[end : end + self.forecast_horizon]
            items = (*items, torch.from_numpy(exog_x), torch.from_numpy(exog_y))

        return items


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Learnable positional encoding added to patch tokens."""

    def __init__(self, d_model: int, n_patches: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_patches, d_model)
        x = x + self.pos_embed
        return self.dropout(x)


class Patching(nn.Module):
    """Converts a time series of shape (B, C, L) into (B, C, N_patches, patch_len).

    Parameters
    ----------
    patch_len : int
        Length of each patch.
    stride : int
        Stride between consecutive patches.
    """

    def __init__(self, patch_len: int, stride: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, C, L)`` where L >= patch_len.

        Returns
        -------
        torch.Tensor
            Shape ``(B, C, N_patches, patch_len)``.
        """
        B, C, L = x.shape
        n_patches = (L - self.patch_len) // self.stride + 1

        # Use unfold to extract patches along the length dimension
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: (B, C, N_patches, patch_len)
        return patches


class PatchTSTEncoder(nn.Module):
    """Single Transformer encoder block with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, N_patches, d_model)
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout1(x_attn)

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class PatchTST(nn.Module):
    """Full PatchTST model for time series forecasting.

    Parameters
    ----------
    n_features : int
        Number of input features (channels).
    context_length : int
        Total length of the historical context window.
    forecast_horizon : int
        Number of future steps to predict.
    patch_len : int
        Length of each patch.
    stride : int
        Stride between consecutive patches.
    d_model : int
        Dimension of the Transformer model.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_features: int,
        context_length: int,
        forecast_horizon: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.mc_dropout = False  # Toggle for MC dropout

        # Patching
        self.patching = Patching(patch_len=patch_len, stride=stride)
        n_patches = (context_length - patch_len) // stride + 1
        assert n_patches > 0, (
            f"context_length={context_length} too short for "
            f"patch_len={patch_len} and stride={stride}"
        )
        self.n_patches = n_patches

        # Per-channel: Linear projection from patch_len to d_model
        self.patch_projection = nn.Linear(patch_len, d_model)

        # Positional encoding (shared across channels)
        self.pos_encoding = PositionalEncoding(d_model, n_patches, dropout)

        # Shared Transformer encoder across channels
        encoder_layers = [
            PatchTSTEncoder(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        self.encoder = nn.ModuleList(encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Flatten per-channel representations -> forecast
        self.head = nn.Sequential(
            nn.Linear(n_patches * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, forecast_horizon),
        )

        # Separate head for exogenous features
        self.exog_head: Optional[nn.Module] = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_mc_dropout(self, enabled: bool = True) -> None:
        """Enable or disable Monte Carlo dropout."""
        self.mc_dropout = enabled
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train() if enabled else m.eval()

    def forward(
        self,
        x: torch.Tensor,
        exog_x: Optional[torch.Tensor] = None,
        exog_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, context_length, n_features)``.
        exog_x : torch.Tensor, optional
            Exogenous input ``(B, context_length, n_exog)``.
        exog_y : torch.Tensor, optional
            Exogenous targets ``(B, forecast_horizon, n_exog)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(B, forecast_horizon)``.
        """
        # Transpose to (B, C, L) for patching
        x = x.transpose(1, 2)  # (B, n_features, context_length)

        # Patch: (B, C, N_patches, patch_len)
        patches = self.patching(x)

        B, C, N_p, P_len = patches.shape

        # Reshape for per-channel processing
        # -> (B * C, N_patches, patch_len)
        patches_flat = patches.reshape(B * C, N_p, P_len)

        # Project each patch to d_model
        # -> (B * C, N_patches, d_model)
        tokens = self.patch_projection(patches_flat)

        # Add positional encoding
        tokens = self.pos_encoding(tokens)  # (B * C, N_patches, d_model)

        # Transformer encoder (shared weights across channels)
        for block in self.encoder:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)  # (B * C, N_patches, d_model)

        # Flatten and pass through head
        # -> (B * C, N_patches * d_model)
        tokens_flat = tokens.reshape(B * C, N_p * self.d_model)

        # -> (B * C, forecast_horizon)
        out = self.head(tokens_flat)

        # Reshape back: (B, C, forecast_horizon) -> aggregate over channels
        out = out.reshape(B, C, self.forecast_horizon)

        # Average across channels for the final prediction
        prediction = out.mean(dim=1)  # (B, forecast_horizon)

        # Add exogenous contribution if available
        if exog_y is not None and self.exog_head is not None:
            exog_contrib = self.exog_head(exog_y)  # (B, forecast_horizon)
            prediction = prediction + exog_contrib

        return prediction


# ---------------------------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------------------------

class SeriesPreprocessor:
    """Normalizes time series data for PatchTST training.

    Stores per-column mean and std for inverse transformation.
    """

    def __init__(self) -> None:
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "SeriesPreprocessor":
        self.feature_names = list(df.columns)
        self.means = df.values.mean(axis=0).astype(np.float32)
        self.stds = df.values.std(axis=0).astype(np.float32)
        self.stds[self.stds < 1e-8] = 1.0  # Avoid division by zero
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return ((df.values - self.means) / self.stds).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise RuntimeError("Preprocessor not fitted.")
        return arr * self.stds + self.means

    def inverse_transform_predictions(
        self, predictions: np.ndarray, target_col_idx: int = 0
    ) -> np.ndarray:
        """Denormalize predictions that target a specific column."""
        if self.means is None or self.stds is None:
            raise RuntimeError("Preprocessor not fitted.")
        return predictions * self.stds[target_col_idx] + self.means[target_col_idx]


# ---------------------------------------------------------------------------
# Main Forecaster Class
# ---------------------------------------------------------------------------

class PatchTSTForecaster:
    """High-level PatchTST-based time series forecaster for stock prices.

    Provides end-to-end functionality:
    - Data preprocessing with technical indicator feature engineering
    - Training with MLflow experiment tracking
    - Single and batch prediction
    - Confidence intervals via Monte Carlo dropout
    - Model persistence (save / load)

    Example
    -------
    >>> forecaster = PatchTSTForecaster(context_length=128, forecast_horizon=24)
    >>> forecaster.train(price_df, n_epochs=50)
    >>> preds = forecaster.predict(price_series, horizon=24)
    """

    def __init__(
        self,
        context_length: int = 128,
        forecast_horizon: int = 24,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        mlflow_experiment: Optional[str] = None,
    ) -> None:
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mlflow_experiment = mlflow_experiment

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[PatchTST] = None
        self.preprocessor = SeriesPreprocessor()
        self._fitted = False

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self, df: pd.DataFrame, val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation DataLoaders from price DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
        val_split : float
            Fraction of data used for validation.

        Returns
        -------
        tuple[DataLoader, DataLoader]
            Training and validation DataLoaders.
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        df = df.copy().sort_index()

        # Feature engineering
        df = compute_technical_indicators(df)

        # Select core features for the model
        feature_cols = [
            "close", "open", "high", "low", "volume",
            "sma_5", "sma_20", "ema_12", "ema_26",
            "macd", "macd_signal", "rsi_14",
            "bb_width", "atr_14", "vol_ratio",
            "log_return_1d", "stoch_k",
        ]
        # Ensure all columns exist (some may not be computed if data is short)
        feature_cols = [c for c in feature_cols if c in df.columns]
        df_features = df[feature_cols].copy()

        # Normalize
        data = self.preprocessor.fit_transform(df_features)

        # Train / val split
        split_idx = int(len(data) * (1 - val_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        train_ds = TimeSeriesDataset(
            train_data,
            context_length=self.context_length,
            forecast_horizon=self.forecast_horizon,
            stride=4,
        )
        val_ds = TimeSeriesDataset(
            val_data,
            context_length=self.context_length,
            forecast_horizon=self.forecast_horizon,
            stride=4,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device == "cuda",
        )

        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        n_epochs: int = 50,
        val_split: float = 0.2,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the PatchTST model on price data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV price data with DatetimeIndex.
        n_epochs : int
            Maximum number of training epochs.
        val_split : float
            Validation set fraction.
        patience : int
            Early stopping patience.
        verbose : bool
            Print training progress.

        Returns
        -------
        dict
            Training history with loss curves.
        """
        train_loader, val_loader = self._prepare_data(df, val_split)

        # Infer n_features from the first batch
        x_sample, y_sample = train_loader.dataset[0]
        n_features = x_sample.shape[1]

        # Initialize model
        self.model = PatchTST(
            n_features=n_features,
            context_length=self.context_length,
            forecast_horizon=self.forecast_horizon,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
        criterion = nn.MSELoss()

        # MLflow setup
        mlflow_active = False
        if self.mlflow_experiment:
            try:
                mlflow.set_experiment(self.mlflow_experiment)
                mlflow.start_run()
                mlflow_active = True
                mlflow.log_params({
                    "context_length": self.context_length,
                    "forecast_horizon": self.forecast_horizon,
                    "patch_len": self.patch_len,
                    "stride": self.stride,
                    "d_model": self.d_model,
                    "n_heads": self.n_heads,
                    "n_layers": self.n_layers,
                    "d_ff": self.d_ff,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "n_epochs": n_epochs,
                    "n_features": n_features,
                })
            except Exception as exc:
                logger.warning("MLflow initialization failed: %s", exc)

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        epochs_no_improve = 0
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, n_epochs + 1):
            # --- Train ---
            self.model.train()
            train_losses: List[float] = []
            for batch in train_loader:
                x, y = batch[0].to(self.device), batch[1].squeeze(-1).to(self.device)
                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train_loss = float(np.mean(train_losses))

            # --- Validate ---
            self.model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(self.device), batch[1].squeeze(-1).to(self.device)
                    pred = self.model(x)
                    loss = criterion(pred, y)
                    val_losses.append(loss.item())

            avg_val_loss = float(np.mean(val_losses))
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            if verbose:
                print(
                    f"Epoch {epoch:>3}/{n_epochs} | "
                    f"train_loss: {avg_train_loss:.6f} | "
                    f"val_loss: {avg_val_loss:.6f}"
                )

            if mlflow_active:
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=epoch)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}.")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._fitted = True

        if mlflow_active:
            mlflow.log_metrics({"best_val_loss": best_val_loss})
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()

        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        series: pd.Series,
        horizon: Optional[int] = None,
        confidence_level: float = 0.95,
        n_mc_samples: int = 100,
    ) -> pd.DataFrame:
        """Forecast future values for a single price series.

        Parameters
        ----------
        series : pd.Series
            Historical close prices with DatetimeIndex.
        horizon : int or None
            Number of steps ahead to forecast. Defaults to ``self.forecast_horizon``.
        confidence_level : float
            Confidence level for prediction intervals (0, 1).
        n_mc_samples : int
            Number of Monte Carlo dropout samples for uncertainty.

        Returns
        -------
        pd.DataFrame
            Columns: ``prediction``, ``lower``, ``upper``, indexed by future dates.
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        horizon = horizon or self.forecast_horizon
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()

        values = series.values.astype(np.float32)

        # Normalize using stored stats (close is assumed column 0)
        mean_close = self.preprocessor.means[0]
        std_close = self.preprocessor.stds[0]
        normalized = (values - mean_close) / std_close

        # Pad or truncate to context_length
        if len(normalized) < self.context_length:
            pad_len = self.context_length - len(normalized)
            normalized = np.pad(normalized, (pad_len, 0), mode="edge")
        context = normalized[-self.context_length:]

        # Expand to (1, context_length, n_features) by replicating across features
        n_features = self.model.n_features
        x = np.tile(context[:, np.newaxis], (1, n_features))
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, L, C)

        # --- Point prediction ---
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).squeeze(0).cpu().numpy()

        # Trim to requested horizon
        pred = pred[:horizon]
        pred_denorm = pred * std_close + mean_close

        # --- Confidence intervals via MC Dropout ---
        self.model.set_mc_dropout(True)
        mc_preds: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_mc_samples):
                p = self.model(x_tensor).squeeze(0).cpu().numpy()[:horizon]
                mc_preds.append(p * std_close + mean_close)
        self.model.set_mc_dropout(False)

        mc_preds = np.stack(mc_preds, axis=0)  # (n_samples, horizon)
        lower = np.percentile(mc_preds, (1 - confidence_level) / 2 * 100, axis=0)
        upper = np.percentile(mc_preds, (1 + confidence_level) / 2 * 100, axis=0)

        # Build result index
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index)
        if freq is None:
            freq = "B"  # Business day default
        future_index = pd.date_range(
            start=last_date, periods=horizon + 1, freq=freq
        )[1:]

        result = pd.DataFrame(
            {
                "prediction": pred_denorm,
                "lower": lower,
                "upper": upper,
            },
            index=future_index,
        )
        return result

    def predict_batch(
        self,
        price_data: Dict[str, pd.DataFrame],
        horizon: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Predict for multiple tickers.

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
            Mapping ticker -> DataFrame with OHLCV data.
        horizon : int or None
            Forecast horizon.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping ticker -> prediction DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}
        for ticker, df in price_data.items():
            try:
                if "close" in df.columns:
                    preds = self.predict(df["close"], horizon=horizon)
                else:
                    preds = self.predict(df.iloc[:, 0], horizon=horizon)
                results[ticker] = preds
            except Exception as exc:
                logger.error("Prediction failed for %s: %s", ticker, exc)
        return results

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model, preprocessor, and config to disk.

        Parameters
        ----------
        path : str
            Directory to save into.
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Cannot save an untrained model.")

        os.makedirs(path, exist_ok=True)

        # Save PyTorch model weights
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

        # Save preprocessor state
        np.savez(
            os.path.join(path, "preprocessor.npz"),
            means=self.preprocessor.means,
            stds=self.preprocessor.stds,
            feature_names=np.array(self.preprocessor.feature_names),
        )

        # Save config
        config = {
            "context_length": self.context_length,
            "forecast_horizon": self.forecast_horizon,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "n_features": self.model.n_features,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            import json
            json.dump(config, f, indent=2)

        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PatchTSTForecaster":
        """Load a saved model from disk.

        Parameters
        ----------
        path : str
            Directory containing saved model.
        device : str or None
            Device to load onto.

        Returns
        -------
        PatchTSTForecaster
            Loaded forecaster ready for inference.
        """
        import json

        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)

        n_features = config.pop("n_features")
        forecaster = cls(**config, device=device)

        forecaster.model = PatchTST(
            n_features=n_features,
            context_length=config["context_length"],
            forecast_horizon=config["forecast_horizon"],
            patch_len=config["patch_len"],
            stride=config["stride"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
        ).to(forecaster.device)

        forecaster.model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location=forecaster.device, weights_only=True)
        )

        # Restore preprocessor
        prep_data = np.load(os.path.join(path, "preprocessor.npz"), allow_pickle=True)
        forecaster.preprocessor.means = prep_data["means"].astype(np.float32)
        forecaster.preprocessor.stds = prep_data["stds"].astype(np.float32)
        forecaster.preprocessor.feature_names = list(prep_data["feature_names"])
        forecaster._fitted = True

        logger.info("Model loaded from %s", path)
        return forecaster
