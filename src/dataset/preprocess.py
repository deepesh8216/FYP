import pandas as pd
import numpy as np
import os
import re
import json
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────────
INPUT_CSV   = "data/processed/fakenewsnet_scraped.csv"
OUTPUT_DIR  = "data/processed"
IMAGE_DIR   = "data/images/fakenewsnet_processed"
IMG_SIZE    = 224
MIN_TEXT_LEN = 50   # filter articles shorter than this
# ─────────────────────────────────────────────────────────

os.makedirs(IMAGE_DIR, exist_ok=True)

# ── 1. Load & Filter ─────────────────────────────────────
print("Step 1: Loading and filtering...")
df = pd.read_csv(INPUT_CSV)

# Keep only rows with both text and image
df = df[df['text'].notna() & df['image_path'].notna()].copy()

# Filter very short texts
df = df[df['text'].str.len() >= MIN_TEXT_LEN].copy()
df = df.reset_index(drop=True)
print(f"  After filtering: {len(df)} articles")

# ── 2. Clean Text ─────────────────────────────────────────
print("Step 2: Cleaning text...")

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

df['text_clean'] = df['text'].apply(clean_text)
df['title_clean'] = df['title'].apply(clean_text)

# Combine title + text for richer input
df['full_text'] = df['title_clean'] + ' [SEP] ' + df['text_clean']

# Truncate to 512 words max (BERT limit)
df['full_text'] = df['full_text'].apply(lambda x: ' '.join(x.split()[:512]))

print(f"  Text cleaning done.")

# ── 3. Validate & Process Images ─────────────────────────
print("Step 3: Validating and resizing images...")

def process_image(img_path, save_dir, article_id):
    """Validate, resize and save image. Return new path or None."""
    try:
        img = Image.open(img_path).convert('RGB')

        # Skip tiny images (likely icons/placeholders)
        if img.width < 50 or img.height < 50:
            return None

        # Resize to 224x224
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        # Save processed image
        new_path = os.path.join(save_dir, f"{article_id}.jpg")
        img.save(new_path, 'JPEG', quality=90)
        return new_path

    except Exception:
        return None

valid_images = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    new_path = process_image(row['image_path'], IMAGE_DIR, row['article_id'])
    valid_images.append(new_path)

df['image_processed'] = valid_images

# Drop rows where image processing failed
df = df[df['image_processed'].notna()].copy()
df = df.reset_index(drop=True)
print(f"  After image validation: {len(df)} articles")

# ── 4. Final Dataset ──────────────────────────────────────
print("Step 4: Building final dataset...")

final_df = df[[
    'article_id',
    'title_clean',
    'full_text',
    'image_processed',
    'label',
    'source'
]].copy()

final_df.columns = [
    'id',
    'title',
    'text',
    'image_path',
    'label',
    'source'
]

print(f"  Final samples: {len(final_df)}")
print(f"  Label distribution:\n{final_df['label'].value_counts()}")

# ── 5. Train / Val / Test Split ───────────────────────────
print("Step 5: Splitting dataset...")

# First split: 85% train+val, 15% test
train_val, test = train_test_split(
    final_df, test_size=0.15, random_state=42, stratify=final_df['label']
)

# Second split: 70% train, 15% val (from original)
train, val = train_test_split(
    train_val, test_size=0.176, random_state=42, stratify=train_val['label']
)

print(f"  Train : {len(train)} samples")
print(f"  Val   : {len(val)} samples")
print(f"  Test  : {len(test)} samples")

# ── 6. Save ───────────────────────────────────────────────
print("Step 6: Saving splits...")

train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val.to_csv(f"{OUTPUT_DIR}/val.csv",   index=False)
test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

# Save metadata
meta = {
    'total_samples' : len(final_df),
    'train_samples' : len(train),
    'val_samples'   : len(val),
    'test_samples'  : len(test),
    'img_size'      : IMG_SIZE,
    'label_map'     : {'0': 'fake', '1': 'real'},
    'train_label_dist': train['label'].value_counts().to_dict(),
    'val_label_dist'  : val['label'].value_counts().to_dict(),
    'test_label_dist' : test['label'].value_counts().to_dict(),
}
with open(f"{OUTPUT_DIR}/dataset_meta.json", 'w') as f:
    json.dump(meta, f, indent=2)

print("\n=== PREPROCESSING COMPLETE ===")
print(f"Train : {len(train)} | Val : {len(val)} | Test : {len(test)}")
print(f"Files saved to {OUTPUT_DIR}/")
print("  train.csv, val.csv, test.csv, dataset_meta.json")