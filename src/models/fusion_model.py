import torch
import torch.nn as nn

from src.models.image_model import ImageOnlyFakeNewsDetector
from src.models.text_model import TextOnlyFakeNewsDetector


class LateFusionFakeNewsDetector(nn.Module):
    """
    Late fusion baseline:
    fused_logits = alpha * text_logits + (1 - alpha) * image_logits
    """

    def __init__(self, hidden_dim=256, dropout=0.3, alpha=0.5):
        super().__init__()
        self.text_model = TextOnlyFakeNewsDetector(hidden_dim=hidden_dim, dropout=dropout)
        self.image_model = ImageOnlyFakeNewsDetector(hidden_dim=hidden_dim, dropout=dropout)
        self.alpha = float(alpha)

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def forward(self, input_ids, attention_mask, image):
        text_logits = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        image_logits = self.image_model(image=image)
        return self.alpha * text_logits + (1.0 - self.alpha) * image_logits


@torch.no_grad()
def fuse_logits(text_logits, image_logits, alpha):
    alpha = float(alpha)
    return alpha * text_logits + (1.0 - alpha) * image_logits
