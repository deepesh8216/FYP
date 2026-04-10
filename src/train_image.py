import sys
sys.path.append(".")

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import FakeNewsDataset
from src.models.image_model import ImageOnlyFakeNewsDetector
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train image-only baseline")
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--val_csv", default="data/processed/val.csv")
    parser.add_argument("--output_dir", default="outputs/week4/image")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = FakeNewsDataset(args.train_csv, augment=True, use_text=False, use_image=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = ImageOnlyFakeNewsDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [image]")
        for batch in loop:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(image=images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch} avg loss: {running_loss / max(1, len(train_loader)):.4f}")

    torch.save({"model_state": model.state_dict(), "seed": args.seed}, f"{args.output_dir}/image_model.pt")
    print(f"Saved: {args.output_dir}/image_model.pt")


if __name__ == "__main__":
    main()
