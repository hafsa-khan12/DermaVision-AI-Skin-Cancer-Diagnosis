import torch
import torch.nn as nn
from torchvision import models


class TransferResNet(nn.Module):
    """Transfer learning model using pretrained ResNet-18."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        if freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)


def count_parameters(model: nn.Module) -> dict:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }