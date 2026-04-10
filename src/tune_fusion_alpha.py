import sys
sys.path.append(".")

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.dataset.dataset import FakeNewsDataset
from src.models.fusion_model import LateFusionFakeNewsDetector
from src.utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Tune late-fusion alpha on validation set")
    parser.add_argument("--val_csv", default="data/processed/val.csv")
    parser.add_argument("--checkpoint", default="outputs/week4/fusion/fusion_model.pt")
    parser.add_argument("--output_json", default="outputs/week4/fusion/fusion_alpha_search.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FakeNewsDataset(args.val_csv, augment=False, use_text=True, use_image=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LateFusionFakeNewsDetector(alpha=0.5).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    search_results = []
    best = {"alpha": None, "f1_macro": -1.0}
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    for alpha in alphas:
        model.set_alpha(alpha)
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
        result = {
            "alpha": alpha,
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
        }
        search_results.append(result)
        if result["f1_macro"] > best["f1_macro"]:
            best = result

    payload = {"best": best, "results": search_results}
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Best alpha: {best['alpha']} (F1={best['f1_macro']:.4f})")
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
