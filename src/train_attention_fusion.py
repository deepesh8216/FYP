import sys
sys.path.append(".")

import argparse
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.dataset.dataset import FakeNewsDataset
from src.models.model import MultimodalFakeNewsDetector
from src.models.contrastive import ContrastiveTextImageModel
from src.utils.seed import set_seed


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return {"val_loss": avg_loss, "val_f1_macro": f1_macro}


def main():
    parser = argparse.ArgumentParser(description="Week 5 attention-fusion training")
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--val_csv", default="data/processed/val.csv")
    parser.add_argument("--output_dir", default="outputs/week5/attention/seed_42")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr_bert", type=float, default=2e-6)
    parser.add_argument("--lr_rest", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--freeze_bert_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init_contrastive",
        default=None,
        help="Path to outputs/week6/.../contrastive_pretrained.pt to init encoders",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = FakeNewsDataset(args.train_csv, augment=True, use_text=True, use_image=True)
    val_ds = FakeNewsDataset(args.val_csv, augment=False, use_text=True, use_image=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MultimodalFakeNewsDetector(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    if args.init_contrastive:
        ckpt = torch.load(args.init_contrastive, map_location=device)
        pre = ContrastiveTextImageModel(
            hidden_dim=args.hidden_dim,
            proj_dim=ckpt.get("config", {}).get("proj_dim", args.hidden_dim),
            dropout=args.dropout,
        ).to(device)
        pre.load_state_dict(ckpt["model_state"])
        model.text_branch.load_state_dict(pre.text_encoder.state_dict())
        model.image_branch.load_state_dict(pre.image_encoder.state_dict())
        print(f"Initialized encoders from: {args.init_contrastive}")

    bert_params = list(model.text_branch.bert.parameters())
    other_params = [p for p in model.parameters() if not any(p is bp for bp in bert_params)]
    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": args.lr_bert},
            {"params": other_params, "lr": args.lr_rest},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val_f1 = -1.0
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        freeze = epoch <= args.freeze_bert_epochs
        for p in model.text_branch.bert.parameters():
            p.requires_grad = not freeze

        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [attention]")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} val_f1={val_metrics['val_f1_macro']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_f1_macro": val_metrics["val_f1_macro"],
                "lr_bert": optimizer.param_groups[0]["lr"],
                "lr_rest": optimizer.param_groups[1]["lr"],
            }
        )

        if val_metrics["val_f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["val_f1_macro"]
            patience_count = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": args.seed,
                    "best_epoch": epoch,
                    "best_val_f1_macro": best_val_f1,
                    "config": vars(args),
                },
                f"{args.output_dir}/best_model.pt",
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {args.output_dir}/best_model.pt")
    print(f"Saved: {args.output_dir}/training_history.json")


if __name__ == "__main__":
    main()
