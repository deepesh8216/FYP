# FYP — Multimodal Fake News Detection

FakeNewsNet (GossipCop + PolitiFact) → multimodal models (DistilBERT + ResNet-50), attention fusion, optional contrastive pretraining, Week 7 evaluation figures, and a **Week 8 web API** (`src/serve_web.py`).

## What is on GitHub

- **Source code** under `src/` (training, evaluation, web app, Docker).
- **`docs/`** — project report.
- **`outputs/`** — metrics JSON, comparison CSVs, confusion-matrix PNGs (no `.pt` weights; too large for GitHub).

## What you keep locally (not in this repo)

- **`data/`** — raw/processed datasets and images (~GB). Obtain via [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) and your `src/dataset/` scripts.
- **Checkpoints** (`*.pt`) — upload separately (e.g. [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), Google Drive, Hugging Face Hub) and set `FAKE_NEWS_CHECKPOINT` when running the API.

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
- Week 6: contrastive + finetune (see `src/pretrain_contrastive.py` and your finetune commands)
- Week 7: `python3 src/run_week7_complete.py`
