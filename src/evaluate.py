import sys
sys.path.append('.')

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

from src.dataset.dataset import FakeNewsDataset
from src.models.model import MultimodalFakeNewsDetector

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load best model ───────────────────────────────────────
print("Loading best model...")
checkpoint = torch.load('outputs/best_model.pt', map_location=device)

model = MultimodalFakeNewsDetector(
    hidden_dim=checkpoint['config']['hidden_dim'],
    num_heads =checkpoint['config']['num_heads'],
    dropout   =checkpoint['config']['dropout']
).to(device)

model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Val F1 at save time: {checkpoint['val_metrics']['f1']}")

# ── Load test set ─────────────────────────────────────────
test_dataset = FakeNewsDataset('data/processed/test.csv', augment=False)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)
print(f"\nTest samples: {len(test_dataset)}")

# ── Run evaluation ────────────────────────────────────────
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images         = batch['image'].to(device)
        labels         = batch['label']

        logits = model(input_ids, attention_mask, images)
        preds  = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ── Results ───────────────────────────────────────────────
acc  = accuracy_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds, average='macro')
prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
cm   = confusion_matrix(all_labels, all_preds)

print("\n=== TEST SET RESULTS ===")
print(f"Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
print(f"F1 Score  : {f1:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"\nConfusion Matrix:")
print(f"              Predicted")
print(f"              Fake  Real")
print(f"Actual Fake  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
print(f"Actual Real  [{cm[1][0]:4d}  {cm[1][1]:4d}]")
print(f"\nDetailed Report:")
print(classification_report(all_labels, all_preds,
      target_names=['Fake', 'Real']))