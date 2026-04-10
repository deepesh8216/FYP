import pandas as pd
import requests
import os
import time
import hashlib
from newspaper import Article
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
BASE       = "data/raw/FakeNewsNet"
IMG_DIR    = "data/images/fakenewsnet"
OUT_CSV    = "data/processed/fakenewsnet_scraped.csv"
MAX_ARTICLES = 2000   # start small to test, increase later
DELAY      = 0.5      # seconds between requests (be polite)
# ────────────────────────────────────────────────────────

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load all 4 CSVs
gc_fake = pd.read_csv(f"{BASE}/gossipcop_fake.csv"); gc_fake['label']=0; gc_fake['source']='gossipcop'
gc_real = pd.read_csv(f"{BASE}/gossipcop_real.csv"); gc_real['label']=1; gc_real['source']='gossipcop'
pf_fake = pd.read_csv(f"{BASE}/politifact_fake.csv"); pf_fake['label']=0; pf_fake['source']='politifact'
pf_real = pd.read_csv(f"{BASE}/politifact_real.csv"); pf_real['label']=1; pf_real['source']='politifact'

df = pd.concat([gc_fake, gc_real, pf_fake, pf_real], ignore_index=True)
df = df.dropna(subset=['news_url', 'title'])

# Balance: sample equal fake/real to avoid bias
fake_df = df[df['label'] == 0].sample(n=min(MAX_ARTICLES//2, len(df[df['label']==0])), random_state=42)
real_df = df[df['label'] == 1].sample(n=min(MAX_ARTICLES//2, len(df[df['label']==1])), random_state=42)
df_sample = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Scraping {len(df_sample)} articles...")

def download_image(img_url, save_dir, article_id):
    """Download image and return local path."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(img_url, timeout=10, headers=headers)
        if r.status_code == 200 and 'image' in r.headers.get('Content-Type', ''):
            ext = img_url.split('.')[-1].split('?')[0][:4]
            if ext not in ['jpg','jpeg','png','webp']:
                ext = 'jpg'
            filename = f"{article_id}.{ext}"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return filepath
    except Exception:
        pass
    return None

results = []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    url = row['news_url']
    if not url.startswith('http'):
        url = 'https://' + url

    article_id = hashlib.md5(url.encode()).hexdigest()[:12]
    record = {
        'id':         row['id'],
        'article_id': article_id,
        'title':      row['title'],
        'label':      row['label'],
        'source':     row['source'],
        'url':        url,
        'text':       None,
        'image_path': None,
    }

    try:
        a = Article(url, request_timeout=10)
        a.download()
        a.parse()

        record['text'] = a.text if a.text else None

        # Download top image
        if a.top_image:
            img_path = download_image(a.top_image, IMG_DIR, article_id)
            record['image_path'] = img_path

    except Exception as e:
        pass  # skip failed articles silently

    results.append(record)
    time.sleep(DELAY)

# Save results
result_df = pd.DataFrame(results)
result_df.to_csv(OUT_CSV, index=False)

# Summary
print("\n=== SCRAPING SUMMARY ===")
print(f"Total processed : {len(result_df)}")
print(f"With text       : {result_df['text'].notna().sum()}")
print(f"With image      : {result_df['image_path'].notna().sum()}")
print(f"Both text+image : {(result_df['text'].notna() & result_df['image_path'].notna()).sum()}")
print(f"Saved to        : {OUT_CSV}")