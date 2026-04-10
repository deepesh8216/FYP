import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ImageOnlyFakeNewsDetector(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, image):
        features = self.backbone(image).flatten(1)
        return self.classifier(features)
