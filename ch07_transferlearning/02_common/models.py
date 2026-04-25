"""Simple 1D CNN models for Chapter 7 teaching experiments.

Input shape:
    [B, 1, 1024]

Feature shape:
    [B, feature_dim]
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from grl import GradientReverseLayer


class FeatureExtractor1D(nn.Module):
    """A lightweight 1D CNN feature extractor."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4),
            nn.GroupNorm(1, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(1, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).squeeze(-1)
        return self.proj(features)


class LabelClassifier(nn.Module):
    """Map extracted features [B, F] to fault logits [B, num_classes]."""

    def __init__(self, feature_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class DomainDiscriminator(nn.Module):
    """Map features [B, F] to domain logits [B, num_domains]."""

    def __init__(self, feature_dim: int = 128, num_domains: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class BaseClassifier(nn.Module):
    """Feature extractor + label classifier baseline."""

    def __init__(self, feature_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor1D(feature_dim=feature_dim)
        self.label_classifier = LabelClassifier(feature_dim=feature_dim, num_classes=num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.label_classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.extract_features(x))


class DANNModel(nn.Module):
    """Feature extractor + classifier + domain discriminator with GRL."""

    def __init__(self, feature_dim: int = 128, num_classes: int = 4, num_domains: int = 2) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor1D(feature_dim=feature_dim)
        self.label_classifier = LabelClassifier(feature_dim=feature_dim, num_classes=num_classes)
        self.domain_discriminator = DomainDiscriminator(feature_dim=feature_dim, num_domains=num_domains)
        self.grl = GradientReverseLayer()

    def forward(self, x: torch.Tensor, lambda_grl: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        class_logits = self.label_classifier(features)
        domain_logits = self.domain_discriminator(self.grl(features, lambda_grl))
        return class_logits, domain_logits, features

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.label_classifier(features)


class ADDAModel(nn.Module):
    """Container used by ADDA two-stage training."""

    def __init__(self, feature_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.source_encoder = FeatureExtractor1D(feature_dim=feature_dim)
        self.target_encoder = copy.deepcopy(self.source_encoder)
        self.classifier = LabelClassifier(feature_dim=feature_dim, num_classes=num_classes)
        self.discriminator = DomainDiscriminator(feature_dim=feature_dim, num_domains=2)

    def predict_with_source(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.source_encoder(x))

    def predict_with_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.target_encoder(x))
