"""
Vision Transformer (ViT) PyTorch Model Wrapper for Spatial Deepfake Detection.
"""

from typing import Tuple

try:
    import torch
    import torch.nn as nn
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class ViTClassifier(nn.Module):
        """
        PyTorch Vision Transformer (ViT-B/16) model for spatial synthetic face classification.
        Produces 768-dimensional latent feature embeddings and binary logits (0 = REAL, 1 = SYNTHETIC).
        """

        def __init__(self, num_classes: int = 2, pretrained: bool = True, embedding_dim: int = 768):
            super(ViTClassifier, self).__init__()
            self.embedding_dim = embedding_dim
            self.num_classes = num_classes

            # Load Vision Transformer backbone
            if pretrained:
                weights = ViT_B_16_Weights.DEFAULT
                self.vit = vit_b_16(weights=weights)
            else:
                self.vit = vit_b_16(weights=None)

            # Replace classification head
            in_features = self.vit.heads.head.in_features
            self.vit.heads = nn.Identity()  # Remove default head to access 768-dim CLS embedding directly

            self.classifier_head = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(in_features, num_classes)
            )

        def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
            """
            Extract 768-dimensional CLS token embeddings from input image batch.

            Args:
                pixel_values: Float tensor of shape (batch_size, 3, 224, 224) normalized.

            Returns:
                Tensor of shape (batch_size, 768).
            """
            features = self.vit(pixel_values)
            return features

        def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass through ViT backbone and classification head.

            Args:
                pixel_values: Float tensor of shape (batch_size, 3, 224, 224).

            Returns:
                Tuple of (logits: Tensor(batch_size, 2), features: Tensor(batch_size, 768))
            """
            features = self.extract_features(pixel_values)
            logits = self.classifier_head(features)
            return logits, features

else:
    class ViTClassifier:  # type: ignore
        """Stub class used when torch is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. ViTClassifier requires torch and torchvision.")
