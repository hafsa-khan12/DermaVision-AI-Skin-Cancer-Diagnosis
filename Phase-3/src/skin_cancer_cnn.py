import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two convolution layers."""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.relu(out)
        return out


class SkinCancerCNN(nn.Module):
    """CNN model for benign vs malignant skin cancer classification."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.conv1 = self._conv_block(3, 32, use_batchnorm)
        self.conv2 = self._conv_block(32, 64, use_batchnorm)
        self.conv3 = self._conv_block(64, 128, use_batchnorm)

        self.residual = ResidualBlock(128) if use_residual else nn.Identity()

        self.conv4 = self._conv_block(128, 256, use_batchnorm)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool,
    ) -> nn.Sequential:
        """Create convolution block."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ])

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input: [batch, 3, 224, 224]
        x = self.conv1(x)
        # Shape: [batch, 32, 112, 112]

        x = self.conv2(x)
        # Shape: [batch, 64, 56, 56]

        x = self.conv3(x)
        # Shape: [batch, 128, 28, 28]

        x = self.residual(x)
        # Shape: [batch, 128, 28, 28]

        x = self.conv4(x)
        # Shape: [batch, 256, 14, 14]

        x = self.global_pool(x)
        # Shape: [batch, 256, 1, 1]

        x = self.classifier(x)
        # Shape: [batch, num_classes]

        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)