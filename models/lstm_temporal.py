"""
PyTorch Bidirectional LSTM Model for Temporal Sequence Analysis across Video Frames.
"""

from typing import Tuple

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class LSTMTemporalModel(nn.Module):
        """
        Bidirectional LSTM PyTorch model to process sequences of ViT 768-dim spatial feature vectors.
        Evaluates temporal inconsistencies across video frames.
        """

        def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_classes: int = 2,
            dropout: float = 0.2
        ):
            super(LSTMTemporalModel, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.num_classes = num_classes

            # 2-layer Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

            # Sequence-level classifier head (hidden_dim * 2 = 512 due to bidirectionality)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass over sequence of frame feature vectors.

            Args:
                x: PyTorch Float Tensor of shape (batch_size, seq_len, input_dim).

            Returns:
                Tuple of (logits: Tensor(batch_size, 2), frame_seq_features: Tensor(batch_size, seq_len, 512))
            """
            # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
            lstm_out, _ = self.lstm(x)

            # Temporal average pooling across sequence length
            pooled_context = torch.mean(lstm_out, dim=1)

            logits = self.classifier(pooled_context)
            return logits, lstm_out

else:
    class LSTMTemporalModel:  # type: ignore
        """Stub class used when torch is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. LSTMTemporalModel requires torch.")
