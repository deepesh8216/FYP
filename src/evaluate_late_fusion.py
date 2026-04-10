import sys
sys.path.append(".")

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.dataset.dataset import FakeNewsDataset
from src.models.fusion_model import LateFusionFakeNewsDetector
from src.utils.metrics import compute_metrics, pretty_print_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate late-fusion baseline")
    parser.add_argument("--test_csv", default="data/processed/test.csv")
    parser.add_argument("--checkpoint", default="outputs/week4/fusion/fusion_model.pt")
    parser.add_argument("--output_json", default="outputs/week4/fusion/test_metrics.json")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FakeNewsDataset(args.test_csv, augment=False, use_text=True, use_image=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LateFusionFakeNewsDetector(alpha=ckpt.get("alpha", 0.5)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                image=batch["image"].to(device),
            )
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch["label"].tolist())

    metrics = compute_metrics(y_true, y_pred)
    pretty_print_metrics(metrics)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
