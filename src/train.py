import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup   # changed: cosine > linear
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.dataset.dataset import FakeNewsDataset
from src.models.model import MultimodalFakeNewsDetector

# ── Config ───────────────────────────────────────────────
CONFIG = {
    'train_csv'   : 'data/processed/train.csv',
    'val_csv'     : 'data/processed/val.csv',
    'output_dir'  : 'outputs/',
    'epochs'      : 20,           # more epochs — early stopping will cut it short anyway

    'batch_size'  : 8,
    'hidden_dim'  : 256,
    'num_heads'   : 4,

    # FIX 1 — higher dropout
    'dropout'     : 0.5,

    # FIX 2 — much lower LR for BERT; modest LR for rest
    'lr_bert'     : 2e-6,         # was 1e-5 — BERT fine-tuning needs to be very gentle
    'lr_rest'     : 3e-5,         # was 5e-5

    # FIX 3 — stronger weight decay
    'weight_decay': 0.1,          # was 5e-2

    # FIX 4 — longer warmup so LR ramps up slowly when BERT unfreezes
    'warmup_steps': 200,          # was 100

    # FIX 5 — label smoothing (prevents overconfident logits)
    'label_smoothing': 0.1,       # new

    # FIX 6 — freeze BERT for fewer epochs so it gets more gradual fine-tuning
    'freeze_bert_epochs': 2,      # was 3

    # FIX 7 — tighter early stopping
    'patience'    : 4,            # was 5; stop sooner once val plateaus
}
# ─────────────────────────────────────────────────────────

os.makedirs(CONFIG['output_dir'], exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ── Data ─────────────────────────────────────────────────
print("Loading data...")
train_dataset = FakeNewsDataset(CONFIG['train_csv'], augment=True)
val_dataset   = FakeNewsDataset(CONFIG['val_csv'],   augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=CONFIG['batch_size'],
    shuffle=True,  num_workers=2, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=CONFIG['batch_size'],
    shuffle=False, num_workers=2, pin_memory=False
)

# ── Model ─────────────────────────────────────────────────
print("Building model...")
model = MultimodalFakeNewsDetector(
    hidden_dim=CONFIG['hidden_dim'],
    num_heads=CONFIG['num_heads'],
    dropout=CONFIG['dropout']
).to(device)

# ── Optimizer ─────────────────────────────────────────────
bert_params  = list(model.text_branch.bert.parameters())
other_params = [p for p in model.parameters()
                if not any(p is bp for bp in bert_params)]

optimizer = torch.optim.AdamW([
    {'params': bert_params,  'lr': CONFIG['lr_bert']},
    {'params': other_params, 'lr': CONFIG['lr_rest']},
], weight_decay=CONFIG['weight_decay'])

total_steps = len(train_loader) * CONFIG['epochs']

# FIX 8 — cosine schedule decays LR smoothly instead of linear drop
# This naturally reduces the LR in later epochs, fighting memorisation
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CONFIG['warmup_steps'],
    num_training_steps=total_steps
)

# FIX 1 — label smoothing on the loss
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])

# ── Helper: evaluate ──────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images         = batch['image'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask, images)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss  = total_loss / len(loader)
    accuracy  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss'     : round(avg_loss, 4),
        'accuracy' : round(accuracy, 4),
        'f1'       : round(f1, 4),
        'precision': round(precision, 4),
        'recall'   : round(recall, 4),
    }

# ── Training Loop ─────────────────────────────────────────
print(f"\nStarting training for up to {CONFIG['epochs']} epochs...\n")

best_val_f1    = 0.0
patience_count = 0
history        = []

for epoch in range(1, CONFIG['epochs'] + 1):

    # Freeze / unfreeze BERT
    freeze = epoch <= CONFIG['freeze_bert_epochs']
    for p in model.text_branch.bert.parameters():
        p.requires_grad = not freeze
    if freeze:
        print(f"Epoch {epoch}: BERT frozen")
    else:
        print(f"Epoch {epoch}: BERT unfrozen")

    # ── Train ──────────────────────────────────────────
    model.train()
    train_loss = 0
    all_preds, all_labels = [], []

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [Train]")
    for batch in loop:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images         = batch['image'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)
        loss   = criterion(logits, labels)
        loss.backward()

        # FIX 9 — tighter gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)   # was 1.0

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        loop.set_postfix(loss=f"{loss.item():.4f}")

    train_acc = accuracy_score(all_labels, all_preds)
    train_f1  = f1_score(all_labels, all_preds, average='macro')
    avg_train_loss = train_loss / len(train_loader)

    # ── Validate ───────────────────────────────────────
    val_metrics = evaluate(model, val_loader, criterion, device)

    # ── Log ────────────────────────────────────────────
    current_lr_bert  = optimizer.param_groups[0]['lr']
    current_lr_rest  = optimizer.param_groups[1]['lr']
    print(f"\n  Train → loss: {avg_train_loss:.4f} | acc: {train_acc:.4f} | f1: {train_f1:.4f}")
    print(f"  Val   → loss: {val_metrics['loss']} | acc: {val_metrics['accuracy']} | f1: {val_metrics['f1']}")
    print(f"          precision: {val_metrics['precision']} | recall: {val_metrics['recall']}")
    print(f"  LR    → bert: {current_lr_bert:.2e} | rest: {current_lr_rest:.2e}\n")

    epoch_log = {
        'epoch'        : epoch,
        'train_loss'   : round(avg_train_loss, 4),
        'train_acc'    : round(float(train_acc), 4),
        'train_f1'     : round(float(train_f1), 4),
        'lr_bert'      : round(current_lr_bert, 8),
        'lr_rest'      : round(current_lr_rest, 8),
        **{f'val_{k}': v for k, v in val_metrics.items()}
    }
    history.append(epoch_log)

    # ── Save best model ────────────────────────────────
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience_count = 0
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config'     : CONFIG,
        }, f"{CONFIG['output_dir']}/best_model.pt")
        print(f"  ✅ Best model saved! (val_f1={best_val_f1:.4f})\n")
    else:
        patience_count += 1
        print(f"  No improvement. Patience: {patience_count}/{CONFIG['patience']}\n")
        if patience_count >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch}.")
            break

# ── Save history ──────────────────────────────────────────
with open(f"{CONFIG['output_dir']}/training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

print("\n=== TRAINING COMPLETE ===")
print(f"Best Val F1 : {best_val_f1:.4f}")
print(f"Model saved : {CONFIG['output_dir']}/best_model.pt")
print(f"History     : {CONFIG['output_dir']}/training_history.json")
