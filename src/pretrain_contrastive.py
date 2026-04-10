import sys
sys.path.append(".")

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import FakeNewsDataset
from src.models.contrastive import ContrastiveTextImageModel, clip_contrastive_loss
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Week 6: contrastive pretraining (CLIP-style)")
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--output_dir", default="outputs/week6/contrastive/seed_42")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ds = FakeNewsDataset(args.train_csv, augment=True, use_text=True, use_image=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = ContrastiveTextImageModel(
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs} [contrastive]")
        for batch in loop:
            optimizer.zero_grad()
            text_emb, image_emb, logit_scale = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                image=batch["image"].to(device),
            )
            loss = clip_contrastive_loss(text_emb, image_emb, logit_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", scale=f"{float(logit_scale):.2f}")

        avg = running / max(1, len(loader))
        history.append({"epoch": epoch, "loss": avg})
        print(f"Epoch {epoch} avg loss: {avg:.4f}")

    ckpt_path = f"{args.output_dir}/contrastive_pretrained.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
        },
        ckpt_path,
    )
    with open(f"{args.output_dir}/pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved: {ckpt_path}")
    print(f"Saved: {args.output_dir}/pretrain_history.json")


if __name__ == "__main__":
    main()
