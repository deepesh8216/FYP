# Final Year Project — Full Technical Report  
## Multimodal Fake News Detection (FakeNewsNet)

**Academic Year:** 2025–2026  

This document combines **proposal-style framing** (objectives, scope, methodology) with a **week-by-week account** of work as implemented in this repository, including dataset construction, model variants, evaluation, ablation, and deployment.

---

## 1. Executive Summary

This project builds an end-to-end **multimodal** system that classifies news articles as **Fake** or **Real** using **article text** and an **associated image**. The corpus is derived from **FakeNewsNet** (GossipCop and PolitiFact splits), scraped and preprocessed into aligned text–image samples. Models progress from **unimodal baselines** and **late fusion** (Week 4) to **cross-attention fusion** (Week 5), with an optional **contrastive pretraining** stage (Week 6). **Week 7** consolidates metrics across seeds and variants into master tables, an ablation summary, and confusion-matrix figures. **Week 8** exposes the best checkpoint through a **FastAPI** service, static web UI, and **Docker** packaging for deployment.

**Important limitation (must appear in any demo or viva):** the system predicts **membership in the dataset’s Fake vs Real distribution**, not verified factual truth. User-typed headlines outside the training domain may be scored unreliably.

---

## 2. Problem Statement & Objectives

### 2.1 Problem  
Misinformation often spreads with **text and images**. Unimodal detectors may miss cues that only appear when both modalities are considered. This project studies whether **multimodal fusion** improves binary fake/real classification on a standard benchmark-style corpus.

### 2.2 Objectives  
1. **Data:** Collect and preprocess **FakeNewsNet**-based articles with **aligned images** and stratified **train / validation / test** splits.  
2. **Baselines (Week 4):** Train and evaluate **text-only**, **image-only**, and **late-fusion** multimodal models.  
3. **Advanced fusion (Week 5):** Implement **bidirectional cross-attention** between text and image embeddings before classification.  
4. **Representation learning (Week 6):** Explore **CLIP-style contrastive pretraining** on text–image pairs, then **fine-tune** the attention-fusion classifier.  
5. **Evaluation (Week 7):** Aggregate results across **multiple random seeds**, produce **comparison tables**, **ablation ranking**, and **confusion matrices**.  
6. **Deployment (Week 8):** Ship a **REST API** and browser UI; containerize with **Docker** for portable deployment.

---

## 3. Dataset & Early-Stage Pipeline (Weeks 1–3 — Data Track)

*The repository does not assign numeric “Week 1–3” flags; the following maps the **dataset scripts** to a typical early project phase.*

### 3.1 Source: FakeNewsNet  
- Raw metadata CSVs: `gossipcop_fake.csv`, `gossipcop_real.csv`, `politifact_fake.csv`, `politifact_real.csv`.  
- Columns include **id**, **news_url**, **title**, and tweet ids (see `data/raw/FakeNewsNet/README.md`).

### 3.2 Exploration  
- Script: `src/dataset/explore_data.py` — loads the four CSVs, assigns **label** (0 = fake, 1 = real) and **source**, and reports shape, missing values, and sample titles.

### 3.3 Scraping & Alignment  
- Script: `src/dataset/scrape_articles.py` — downloads article content via URLs, saves images under `data/images/…`, and writes `data/processed/fakenewsnet_scraped.csv` (balanced sampling configurable via `MAX_ARTICLES`).

### 3.4 Preprocessing  
- Script: `src/dataset/preprocess.py` — filters rows with **both** text and image, enforces **minimum text length** (`MIN_TEXT_LEN`), cleans text (URLs, emails, non-ASCII stripping), resizes images (e.g. 224×224), and emits **train / val / test** CSVs under `data/processed/`.

### 3.5 Loader for Training  
- Script: `src/dataset/dataset.py` — `FakeNewsDataset`: **DistilBERT** tokenization for text, **ImageNet-style** normalization for images, optional **modality ablation** flags (`use_text`, `use_image`), and **augmentation** when enabled.

**Scale note (typical in this project):** after strict multimodal filtering, **training** often contains on the order of **hundreds** of samples (e.g. ~738 train rows in a representative run). The scraped pool (`fakenewsnet_scraped.csv`) can be much larger; the bottleneck is **quality filters** and **image availability**, not only raw URL count.

---

## 4. Methodology: Model Family

### 4.1 Text Branch  
- **DistilBERT** (`distilbert-base-uncased`) → CLS pooled representation → linear projection to **hidden_dim** (default 256) with LayerNorm, ReLU, Dropout (`src/models/model.py`, `TextBranch`).

### 4.2 Image Branch  
- **ResNet-50** backbone (ImageNet weights) → global pooled features → projection to **hidden_dim** (`ImageBranch`).

### 4.3 Late Fusion (Week 4)  
- Implemented in `src/train_late_fusion.py` / `src/models/fusion_model.py` (separate from attention stack): combines modalities at a **logit or feature level** without cross-attention; **fusion weight α** can be tuned via `src/tune_fusion_alpha.py`.

### 4.4 Attention Fusion (Week 5)  
- **Bidirectional cross-attention:** text queries image, image queries text; residual + LayerNorm; **concatenated** attended features → MLP classifier (two classes) (`AttentionFusion`, `MultimodalFakeNewsDetector` in `src/models/model.py`).

### 4.5 Contrastive Pretraining (Week 6)  
- **CLIP-style** alignment: `src/pretrain_contrastive.py` trains `ContrastiveTextImageModel` (`src/models/contrastive.py`) with a **symmetric contrastive loss** over text and image projections; weights then feed **fine-tuning** of the attention-fusion detector (`outputs/week6/finetune_attention/…`).

### 4.6 Metrics  
- Accuracy, **macro** precision / recall / F1 (see `src/utils/metrics.py` and evaluation scripts). Multiple **seeds** (e.g. 42, 1337, 2026) for stability reporting.

---

## 5. Week-by-Week Deliverables (As in Codebase)

### Week 4 — Unimodal Baselines & Late Fusion  
**Orchestration:** `src/run_week4_experiments.py`  

| Step | Script | Output (per seed) |
|------|--------|-------------------|
| Text-only train | `train_text.py` | `outputs/week4/text/seed_*/text_model.pt` |
| Text-only eval | `evaluate_text.py` | `test_metrics.json` |
| Image-only train | `train_image.py` | `outputs/week4/image/seed_*/image_model.pt` |
| Image-only eval | `evaluate_image.py` | `test_metrics.json` |
| Late fusion train | `train_late_fusion.py` | `outputs/week4/fusion/seed_*/fusion_model.pt` |
| α search | `tune_fusion_alpha.py` | `fusion_alpha_search.json` |
| Late fusion eval | `evaluate_late_fusion.py` | `test_metrics.json` |

**Summary table:** `python3 src/summarize_week4.py` → `outputs/week4/comparison_table.csv`.

**Key idea:** Establish whether **multimodal late fusion** beats unimodal image; compare against strong **text-only** baseline.

---

### Week 5 — Attention-Based Multimodal Fusion  
**Orchestration:** `src/run_week5_experiments.py`  

- Train: `train_attention_fusion.py` → `outputs/week5/attention/seed_*/best_model.pt`  
- Evaluate: `evaluate_attention_fusion.py` → `test_metrics.json`  
- Summarize: `summarize_week5.py` → `outputs/week5/comparison_table.csv`  

**Key idea:** Replace naive fusion with **learned cross-modal attention** so each modality can **condition** on the other before classification.

---

### Week 6 — Contrastive Pretrain + Fine-Tune  
**Orchestration:** `src/run_week6_experiments.py` (per seed: pretrain → `train_attention_fusion.py --init_contrastive ...` → evaluate).

**Pipeline:**  
1. `pretrain_contrastive.py` → `outputs/week6/contrastive/seed_*/contrastive_pretrained.pt` (+ history JSON).  
2. `train_attention_fusion.py` with `--init_contrastive` → **`outputs/week6/finetune_attention/seed_*/best_model.pt`** (same filename as Week 5; different folder).  
3. `summarize_week6.py` → `outputs/week6/comparison_table.csv`.  

**Key idea:** Improve **shared text–image representation** before supervised classification; **compare** against Week 5 trained **end-to-end** without contrastive stage.

---

### Week 7 — Cross-Experiment Analysis & Figures  
**No training:** Week 7 does **not** produce `best_model.pt`; it only aggregates Week 4–6 metrics and exports figures.

**Orchestration:** `src/run_week7_complete.py`  

1. `summarize_week7.py` reads Week 4–6 comparison CSVs and writes:  
   - `outputs/week7/master_comparison.csv`  
   - `outputs/week7/aggregate_by_variant.csv`  
   - `outputs/week7/ablation_summary.json` (mean metrics per variant, **ranking** by mean macro-F1, short **ablation_notes**).  
2. `week7_export_figures.py` generates **confusion matrix** PNGs under `outputs/week7/figures/` from stored `test_metrics.json` files (requires matplotlib/seaborn).

**Representative aggregated results** (mean over 3 seeds, from `ablation_summary.json` in this repo):

| Rank | Week | Variant | Mean F1 (macro) | Mean Accuracy |
|------|------|---------|-----------------|---------------|
| 1 | 5 | attention_fusion | **0.708** | **0.711** |
| 2 | 4 | unimodal_text | 0.707 | 0.711 |
| 3 | 6 | contrastive_then_attention | 0.682 | 0.686 |
| 4 | 4 | late_fusion | 0.682 | 0.686 |
| 5 | 4 | unimodal_image | 0.629 | 0.633 |

*Interpretation for the report:* **Text is very strong** on this corpus; **attention fusion** matches or slightly edges text-only mean F1 here; **contrastive pretrain** did not improve mean test F1 in this run (worth discussing: data size, hyperparameters, catastrophic forgetting, or need for longer fine-tuning).

---

### Week 8 — Web API, UI & Docker  
**Local serve:** `python3 src/serve_web.py` (Uvicorn, `src.web.app:app`).  

**API:**  
- `GET /health` — liveness.  
- `GET /` — static UI (`src/web/static/index.html`).  
- `POST /predict` — `multipart/form-data`: `text`, optional `image`.  
- `POST /predict_json` — JSON with `text`, optional `image_base64`.  

**Inference:** `src/web/predictor.py` loads `MultimodalFakeNewsDetector` from checkpoint; missing image → **zero tensor** placeholder (same idea as training fallbacks).  

**Environment:** `FAKE_NEWS_CHECKPOINT` points to `best_model.pt` (default in code may reference Week 5 seed; override for Week 6 or other runs).  

**Docker:** `Dockerfile` — Python 3.11 slim, installs deps, copies `src/`, runs Uvicorn; mount checkpoint at `/model.pt` as documented in file comments.

---

## 6. Experimental Setup (for Reproducibility)

- **Seeds:** e.g. `42`, `1337`, `2026` (see `src/utils/seed.py`).  
- **Week 4** default epochs: 3 (`run_week4_experiments.py`); **Week 5** default: 8 epochs.  
- **Hardware:** CUDA when available; CPU fallback supported.  
- **Splits:** Fixed train/val/test CSVs under `data/processed/`.

---

## 7. Ethical & Legal Considerations

1. **Not a fact checker:** Outputs are **probabilistic** and **dataset-biased**; harmful or sensitive user text must not be treated as ground truth.  
2. **Dataset licensing & scraping:** Respect **FakeNewsNet** terms, site **robots.txt**, and **copyright** when scraping; document what was actually collected.  
3. **Deployment:** Add **disclaimers** in the UI; consider **rate limiting** and **logging** policy if public-facing.  
4. **Responsible demos:** Avoid optimizing for sensational or defamatory examples; focus on **methodology** and **known labeled samples** from `train.csv` / raw CSVs.

---

## 8. Future Work

1. **More multimodal supervision:** Increase **train** size by relaxing filters only where images/text remain valid, or add **augmentation**.  
2. **Domain robustness:** Evaluate on **out-of-domain** headlines or another dataset.  
3. **URL ingestion:** Fetch article + lead image from a user-supplied URL (with legal review).  
4. **Hybrid verification:** Optional retrieval over **trusted sources** or fact-check APIs (separate from the neural classifier).  
5. **Explainability:** Attention weights or gradient-based saliency across text tokens and image regions.

---

## 9. Repository Map (Quick Reference)

| Area | Path |
|------|------|
| Raw FakeNewsNet metadata | `data/raw/FakeNewsNet/*.csv` |
| Scraped / processed CSVs | `data/processed/` |
| Dataset & scrape code | `src/dataset/` |
| Core multimodal model | `src/models/model.py` |
| Week 4 runner | `src/run_week4_experiments.py` |
| Week 5 runner | `src/run_week5_experiments.py` |
| Week 6 runner | `src/run_week6_experiments.py`, `src/pretrain_contrastive.py`, `src/models/contrastive.py` |
| Week 7 aggregation | `src/summarize_week7.py`, `src/week7_export_figures.py` |
| Web app | `src/web/app.py`, `src/web/predictor.py`, `src/web/static/index.html` |
| Entrypoint | `src/serve_web.py` |
| Container | `Dockerfile` |

---

## 10. Conclusion

The project delivers a **complete pipeline**: from **FakeNewsNet-aligned multimodal data** through **progressive model complexity** (unimodal → late fusion → attention fusion → optional contrastive pretrain), **rigorous multi-seed evaluation** and **ablation reporting** (Week 7), to a **deployable inference service** (Week 8). The quantitative story in this codebase highlights the **strength of text** on this corpus and shows **attention fusion** as the **strongest multimodal** configuration under the reported mean metrics, with **contrastive pretraining** as an avenue that **warrants further tuning** rather than a guaranteed gain at small data scale.

---

*End of report. Replace bracketed fields ([Your Name], institution) and attach your supervisor-approved proposal PDF reference if required.*
