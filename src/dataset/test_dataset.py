import sys
sys.path.append('.')
from src.dataset.dataset import FakeNewsDataset
from torch.utils.data import DataLoader

print("Loading dataset...")
train_dataset = FakeNewsDataset(
    'data/processed/train.csv',
    augment=True
)
val_dataset = FakeNewsDataset(
    'data/processed/val.csv',
    augment=False
)

print(f"Train size : {len(train_dataset)}")
print(f"Val size   : {len(val_dataset)}")

# Test one sample
sample = train_dataset[0]
print(f"\nSample keys    : {list(sample.keys())}")
print(f"input_ids shape: {sample['input_ids'].shape}")
print(f"attn_mask shape: {sample['attention_mask'].shape}")
print(f"image shape    : {sample['image'].shape}")
print(f"label          : {sample['label']}")

# Test DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
batch = next(iter(train_loader))
print(f"\nBatch input_ids : {batch['input_ids'].shape}")
print(f"Batch images    : {batch['image'].shape}")
print(f"Batch labels    : {batch['label']}")
print("\n✅ Dataset loader working correctly!")