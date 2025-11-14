"""
Temporal Fusion Transformer for multi-horizon volatility forecasting.

Simplified implementation using LSTM encoder and attention mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for volatility forecasting.
    
    Architecture:
    1. Feature embedding
    2. LSTM encoder for temporal dependencies
    3. Attention mechanism for variable selection
    4. Multi-horizon output heads
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        num_horizons: int = 3
    ):
        """
        Initialize TFT model.
        
        Args:
            num_features: Number of input features
            hidden_size: Hidden dimension size
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            num_horizons: Number of forecast horizons
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_horizons = num_horizons
        
        # Feature embedding
        self.feature_embedding = nn.Linear(num_features, hidden_size)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism for temporal aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection layers (one per horizon)
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
            for _ in range(num_horizons)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, sequence_length, num_features]
            
        Returns:
            Predictions [batch, num_horizons]
        """
        # Feature embedding
        embedded = self.feature_embedding(x)  # [batch, seq, hidden]
        embedded = F.relu(embedded)
        embedded = self.dropout(embedded)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)  # [batch, seq, hidden]
        
        # Self-attention for temporal aggregation
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [batch, seq, hidden]
        
        # Take last timestep
        last_hidden = attn_out[:, -1, :]  # [batch, hidden]
        
        # Multi-horizon predictions
        predictions = []
        for output_layer in self.output_layers:
            pred = output_layer(last_hidden)  # [batch, 1]
            predictions.append(pred)
        
        # Stack predictions
        output = torch.cat(predictions, dim=1)  # [batch, num_horizons]
        
        return output


class SimpleLSTMForecaster(nn.Module):
    """
    Simpler LSTM-based forecaster (alternative to TFT).
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_horizons: int = 3
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            num_features: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_horizons: Number of forecast horizons
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_horizons)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, sequence_length, num_features]
            
        Returns:
            Predictions [batch, num_horizons]
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take last hidden state
        last_hidden = hidden[-1]  # [batch, hidden]
        
        # Fully connected output
        output = self.fc(last_hidden)  # [batch, num_horizons]
        
        return output


def create_model(
    num_features: int,
    model_type: str = 'tft',
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    num_horizons: int = 3
) -> nn.Module:
    """
    Create a forecasting model.
    
    Args:
        num_features: Number of input features
        model_type: 'tft' or 'lstm'
        hidden_size: Hidden dimension size
        num_layers: Number of layers
        dropout: Dropout rate
        num_horizons: Number of forecast horizons
        
    Returns:
        Model instance
    """
    if model_type == 'tft':
        model = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=hidden_size,
            num_lstm_layers=num_layers,
            dropout=dropout,
            num_horizons=num_horizons
        )
    elif model_type == 'lstm':
        model = SimpleLSTMForecaster(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_horizons=num_horizons
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == '__main__':
    # Test model
    batch_size = 32
    seq_length = 90
    num_features = 50
    num_horizons = 3
    
    # Create dummy input
    x = torch.randn(batch_size, seq_length, num_features)
    
    # Test TFT
    print("Testing TFT...")
    tft = create_model(
        num_features=num_features,
        model_type='tft',
        hidden_size=128,
        num_horizons=num_horizons
    )
    
    output = tft(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in tft.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Test LSTM
    print("\nTesting LSTM...")
    lstm = create_model(
        num_features=num_features,
        model_type='lstm',
        hidden_size=128,
        num_horizons=num_horizons
    )
    
    output = lstm(x)
    print(f"Output shape: {output.shape}")
    
    num_params = sum(p.numel() for p in lstm.parameters())
    print(f"Number of parameters: {num_params:,}")

