import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models import resnet50, ResNet50_Weights


# ──────────────────────────────────────────────────────────
# 1. TEXT BRANCH  (DistilBERT)
# ──────────────────────────────────────────────────────────
class TextBranch(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Project 768 → hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]   # [B, 768]
        return self.projector(cls_token)              # [B, hidden_dim]


# ──────────────────────────────────────────────────────────
# 2. IMAGE BRANCH  (ResNet50)
# ──────────────────────────────────────────────────────────
class ImageBranch(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove final FC layer, keep feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Project 2048 → hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, image):
        features = self.backbone(image)          # [B, 2048, 1, 1]
        features = features.flatten(1)           # [B, 2048]
        return self.projector(features)          # [B, hidden_dim]


# ──────────────────────────────────────────────────────────
# 3. ATTENTION FUSION
# ──────────────────────────────────────────────────────────
class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()

        # Cross attention: text attends to image
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross attention: image attends to text
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat, image_feat):
        # Reshape to [B, 1, hidden_dim] for attention
        t = text_feat.unsqueeze(1)
        i = image_feat.unsqueeze(1)

        # Text queries image
        t_attended, _ = self.text_to_image(
            query=t, key=i, value=i
        )
        t_out = self.norm1(text_feat + t_attended.squeeze(1))

        # Image queries text
        i_attended, _ = self.image_to_text(
            query=i, key=t, value=t
        )
        i_out = self.norm2(image_feat + i_attended.squeeze(1))

        # Concatenate both attended features
        fused = torch.cat([t_out, i_out], dim=1)  # [B, hidden_dim*2]
        return fused


# ──────────────────────────────────────────────────────────
# 4. FULL MODEL
# ──────────────────────────────────────────────────────────
class MultimodalFakeNewsDetector(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()

        self.text_branch  = TextBranch(hidden_dim, dropout)
        self.image_branch = ImageBranch(hidden_dim, dropout)
        self.fusion       = AttentionFusion(hidden_dim, num_heads, dropout)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)   # 2 classes: fake / real
        )

    def forward(self, input_ids, attention_mask, image):
        text_feat  = self.text_branch(input_ids, attention_mask)
        image_feat = self.image_branch(image)
        fused      = self.fusion(text_feat, image_feat)
        logits     = self.classifier(fused)
        return logits


# ──────────────────────────────────────────────────────────
# 5. QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = MultimodalFakeNewsDetector().to(device)

    # Dummy inputs
    input_ids      = torch.randint(0, 1000, (4, 512)).to(device)
    attention_mask = torch.ones(4, 512).long().to(device)
    image          = torch.randn(4, 3, 224, 224).to(device)

    # Forward pass
    logits = model(input_ids, attention_mask, image)
    print(f"Input  - text : {input_ids.shape}")
    print(f"Input  - image: {image.shape}")
    print(f"Output - logits: {logits.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params    : {total_params:,}")
    print(f"Trainable params: {trainable:,}")
    print("\n✅ Model forward pass working!")