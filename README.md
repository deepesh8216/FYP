# FYP — Multimodal Fake News Detection

FakeNewsNet (GossipCop + PolitiFact) → multimodal models (DistilBERT + ResNet-50), attention fusion, optional contrastive pretraining, Week 7 evaluation figures, and a **Week 8 web API** (`src/serve_web.py`).

## What is on GitHub

- **Source code** under `src/` (training, evaluation, web app, Docker).
- **`docs/`** — project report.
- **`outputs/`** — metrics JSON, comparison CSVs, confusion-matrix PNGs (no `.pt` weights; too large for GitHub).

## What you keep locally (not in this repo)

- **`data/`** — raw/processed datasets and images (~GB). Obtain via [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) and your `src/dataset/` scripts.
- **Checkpoints** (`*.pt`) — upload separately (e.g. [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), Google Drive, Hugging Face Hub) and set `FAKE_NEWS_CHECKPOINT` when running the API.

### Where each `best_model.pt` comes from

| Week | Trains? | Default weight file (after you run training locally) |
|------|---------|------------------------------------------------------|
| **5** | Yes — attention fusion from scratch | `outputs/week5/attention/seed_<seed>/best_model.pt` |
| **6** | Yes — contrastive pretrain, then **same** attention fusion fine-tuned | `outputs/week6/finetune_attention/seed_<seed>/best_model.pt` |
| **7** | **No** — only merges metrics / figures from weeks 4–6 | *(no new checkpoint)* |

The web app defaults to **Week 5** `seed_1337` because that run had the best mean test F1 in your ablation. To serve the Week 6 model:

```bash
export FAKE_NEWS_CHECKPOINT=outputs/week6/finetune_attention/seed_1337/best_model.pt
python3 src/serve_web.py
```

## Quick start (code only)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# After you have data/processed/train.csv and a checkpoint:
export FAKE_NEWS_CHECKPOINT=/path/to/best_model.pt
python3 src/serve_web.py
# Open http://127.0.0.1:8000
```

## Docker

See comments in `Dockerfile` — mount a checkpoint into the container and set `FAKE_NEWS_CHECKPOINT`.

## Reproduce experiments

- Week 4: `python3 src/run_week4_experiments.py`
- Week 5: `python3 src/run_week5_experiments.py`
- Week 6: `python3 src/run_week6_experiments.py` (or run `pretrain_contrastive.py` then `train_attention_fusion.py --init_contrastive ...` manually)
- Week 7: `python3 src/run_week7_complete.py`
