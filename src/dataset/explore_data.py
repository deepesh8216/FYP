import pandas as pd
import os

base = "data/raw/FakeNewsNet"

# Load all 4 files
gc_fake = pd.read_csv(f"{base}/gossipcop_fake.csv")
gc_real = pd.read_csv(f"{base}/gossipcop_real.csv")
pf_fake = pd.read_csv(f"{base}/politifact_fake.csv")
pf_real = pd.read_csv(f"{base}/politifact_real.csv")

# Add labels
gc_fake['label'] = 0  # fake
gc_real['label'] = 1  # real
pf_fake['label'] = 0
pf_real['label'] = 1

# Add source
gc_fake['source'] = 'gossipcop'
gc_real['source'] = 'gossipcop'
pf_fake['source'] = 'politifact'
pf_real['source'] = 'politifact'

# Combine
df = pd.concat([gc_fake, gc_real, pf_fake, pf_real], ignore_index=True)

print("=== COMBINED DATASET ===")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst row:\n", df.iloc[0])
print("\nLabel distribution:\n", df['label'].value_counts())
print("\nMissing values:\n", df.isnull().sum())
print("\nSample titles:")
print(df['title'].head(5))