import sys
sys.path.append(".")

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import FakeNewsDataset
from src.models.text_model import TextOnlyFakeNewsDetector
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train text-only baseline")
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--val_csv", default="data/processed/val.csv")
    parser.add_argument("--output_dir", default="outputs/week4/text")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = FakeNewsDataset(args.train_csv, augment=False, use_text=True, use_image=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = TextOnlyFakeNewsDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [text]")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch} avg loss: {running_loss / max(1, len(train_loader)):.4f}")

    torch.save({"model_state": model.state_dict(), "seed": args.seed}, f"{args.output_dir}/text_model.pt")
    print(f"Saved: {args.output_dir}/text_model.pt")


if __name__ == "__main__":
    main()
