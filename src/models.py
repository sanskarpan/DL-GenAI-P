"""
CNN models for audio genre classification.

Strategy: treat mel spectrogram as image → pretrained EfficientNet backbone.
This leverages ImageNet pretraining and is a proven approach for audio classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class AudioEfficientNet(nn.Module):
    """
    EfficientNet backbone for audio classification via mel spectrograms.
    Input: (B, 1, N_MELS, T) log-mel spectrogram
    Output: (B, num_classes) logits

    Uses 1-channel input since mel spec is grayscale.
    img_size=None: pass raw (128, 431) directly — no resize, much faster on MPS.
    img_size=int:  resize to a square before passing to backbone.
    """

    def __init__(self,
                 model_name: str = config.MODEL_NAME,
                 num_classes: int = config.NUM_CLASSES,
                 pretrained: bool = config.PRETRAINED,
                 dropout: float = config.DROPOUT,
                 img_size: int = None):
        super().__init__()
        import timm
        self.img_size = img_size  # None = no resize

        # Load pretrained backbone
        # in_chans=1 for grayscale mel spectrogram
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,           # single channel mel spectrogram
            num_classes=0,        # remove original classifier
            global_pool='avg',    # global avg pool handles any spatial size
        )

        # Get feature dimension using actual spec dimensions (no resize needed)
        with torch.no_grad():
            if img_size is not None:
                dummy = torch.zeros(1, 1, img_size, img_size)
            else:
                dummy = torch.zeros(1, 1, config.N_MELS, config.N_FRAMES)
            n_features = self.backbone(dummy).shape[-1]

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, N_MELS, T)
        if self.img_size is not None:
            # Only resize if explicitly requested
            if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
                x = F.interpolate(x, size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False)

        features = self.backbone(x)  # (B, n_features)
        logits = self.classifier(features)  # (B, num_classes)
        return logits


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for quick experimentation.
    Input: (B, 1, N_MELS, T)
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 n_mels: int = config.N_MELS):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # (B, 32, 64, T/2)
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # (B, 64, 32, T/4)
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # (B, 128, 16, T/8)
            nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 256, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Basic residual block for audio CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class AudioResNet(nn.Module):
    """
    Custom ResNet-style CNN for audio.
    Deeper but still lightweight compared to EfficientNet.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )

        self.layer2 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        self.layer3 = nn.Sequential(ResidualBlock(256), ResidualBlock(256))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x)


def get_model(model_type: str = "efficientnet",
              **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: "efficientnet", "simple_cnn", or "resnet"
    """
    if model_type == "efficientnet":
        return AudioEfficientNet(**kwargs)
    elif model_type == "simple_cnn":
        return SimpleCNN(**kwargs)
    elif model_type == "resnet":
        return AudioResNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
