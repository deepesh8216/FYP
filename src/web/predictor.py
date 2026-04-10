import io
import os
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import DistilBertTokenizer

from src.models.model import MultimodalFakeNewsDetector


def _load_checkpoint_config(ckpt: dict) -> Tuple[int, int, float]:
    cfg = ckpt.get("config") or {}
    if isinstance(cfg, dict):
        return (
            int(cfg.get("hidden_dim", 256)),
            int(cfg.get("num_heads", 4)),
            float(cfg.get("dropout", 0.5)),
        )
    return 256, 4, 0.5


def build_model_from_checkpoint(path: str, device: str) -> MultimodalFakeNewsDetector:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    hidden_dim, num_heads, dropout = _load_checkpoint_config(ckpt)
    model = MultimodalFakeNewsDetector(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    state = ckpt.get("model_state") or ckpt.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint missing model_state")
    model.load_state_dict(state)
    model.eval()
    return model


class MultimodalPredictor:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        max_text_len: int = 512,
        img_size: int = 224,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = checkpoint_path or os.environ.get(
            "FAKE_NEWS_CHECKPOINT",
            "outputs/week5/attention/seed_1337/best_model.pt",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Set FAKE_NEWS_CHECKPOINT or pass checkpoint_path."
            )
        self.model = build_model_from_checkpoint(path, self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_text_len = max_text_len
        self.img_size = img_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _tokenize(self, text: str):
        enc = self.tokenizer(
            text or "",
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def _image_tensor(self, image_bytes: Optional[bytes]) -> torch.Tensor:
        if not image_bytes:
            return torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            t = self.image_transform(img).unsqueeze(0).to(self.device)
            return t
        except Exception:
            return torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)

    @torch.no_grad()
    def predict(self, text: str, image_bytes: Optional[bytes] = None):
        input_ids, attention_mask = self._tokenize(text)
        image = self._image_tensor(image_bytes)
        logits = self.model(input_ids, attention_mask, image)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred = int(torch.argmax(logits, dim=1).item())
        label_names = ["Fake", "Real"]
        return {
            "label": label_names[pred],
            "label_id": pred,
            "prob_fake": round(probs[0], 4),
            "prob_real": round(probs[1], 4),
        }
