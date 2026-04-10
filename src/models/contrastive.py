import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model import ImageBranch, TextBranch


class ContrastiveTextImageModel(nn.Module):
    """
    CLIP-style contrastive pretraining for paired (text, image).
    Produces normalized embeddings and a learnable temperature (logit scale).
    """

    def __init__(self, hidden_dim=256, proj_dim=256, dropout=0.3, init_logit_scale=1 / 0.07):
        super().__init__()
        self.text_encoder = TextBranch(hidden_dim=hidden_dim, dropout=dropout)
        self.image_encoder = ImageBranch(hidden_dim=hidden_dim, dropout=dropout)

        self.text_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.image_proj = nn.Linear(hidden_dim, proj_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)).log())

    def encode_text(self, input_ids, attention_mask):
        x = self.text_encoder(input_ids, attention_mask)
        x = self.text_proj(x)
        return F.normalize(x, dim=-1)

    def encode_image(self, image):
        x = self.image_encoder(image)
        x = self.image_proj(x)
        return F.normalize(x, dim=-1)

    def forward(self, input_ids, attention_mask, image):
        text_emb = self.encode_text(input_ids, attention_mask)
        image_emb = self.encode_image(image)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        return text_emb, image_emb, logit_scale


def clip_contrastive_loss(text_emb, image_emb, logit_scale):
    """
    Symmetric InfoNCE over the batch.
    text_emb: [B, D] normalized
    image_emb: [B, D] normalized
    """
    logits = logit_scale * (text_emb @ image_emb.t())  # [B, B]
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_t2i = F.cross_entropy(logits, targets)
    loss_i2t = F.cross_entropy(logits.t(), targets)
    return (loss_t2i + loss_i2t) / 2.0
