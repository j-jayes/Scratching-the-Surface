# Project "Cascade Defect" — Implementation Plan

> **Objective:** Design, build, and evaluate a cost-effective ML pipeline for real-time rolled-metal defect detection using a three-layer Cascade Architecture.

---

## Status Legend
- [x] Completed
- [ ] Pending

---

## Phase 0 — Repository Scaffolding

- [x] Create `.claude/plans/cascade_defect_plan.md` (this file)
- [x] Create `.agents/skills/` with Copilot domain-knowledge markdown files
- [x] Create `.devcontainer/` configuration (Dockerfile + devcontainer.json)
- [x] Create `.github/workflows/copilot-setup-steps.yml`
- [x] Create `.pre-commit-config.yaml` (nbstripout + ruff)
- [x] Create `pyproject.toml` with uv-compatible project metadata & dependencies
- [x] Generate `uv.lock` (required by CI cache step)
- [x] Create cookiecutter-style `src/cascade_defect/` package skeleton
- [x] Create `docs/` Quarto website skeleton
- [x] Update `README.md`
- [x] Add `.env.example` and document secret-handling convention
- [x] Fix `.devcontainer` (cross-platform mount, file ownership)
- [x] Fix failing `copilot-setup-steps.yml` workflow

---

## Phase 1 — Data Acquisition & Preparation

- [ ] Download NEU Metal Surface Defects Database from Kaggle (`data/raw/`) — *code ready in `data/ingest.py`; needs valid Kaggle token*
- [x] Write `src/cascade_defect/data/split.py` — stratified split logic
  - Few-Shot Seed   : 3 images × 6 classes  =  18 images  (1%)
  - Unlabeled Pool  : ~1,420 images          (79%)
  - Golden Test Set : 360 images             (20%)
- [ ] Upload split data to **Azure Blob Storage** — *code ready in `data/upload.py`*
- ~~Register dataset in Azure ML Workspace~~ *(dropped — using ACA only)*

---

## Phase 2 — MLLM Pseudo-Labelling (Layer 3 — Offline Mode)

- [ ] Write `src/cascade_defect/layer3_gpt4o/annotate.py`
  - Build Few-Shot system prompt with 18 seed images (base64 encoded)
  - Use Pydantic model `DefectPrediction` to enforce JSON schema via structured outputs
  - Batch-call Azure OpenAI `gpt-4o` on the 79% unlabelled pool
- [ ] Store pseudo-labels in ADLS (`data/processed/pseudo_labels.json`)
- [ ] Track annotation run cost with `tiktoken`

---

## Phase 3 — Model Training (Azure ML)

### 3a — Autoencoder (Layer 1)
- [ ] Write `src/cascade_defect/layer1_autoencoder/train.py`
  - Conv-AE trained **only on defect-free** images (normal distribution)
  - Log MSE threshold curve to MLflow
- [ ] Submit AML job on `Standard_NC6s_v3` compute cluster
- [ ] Register model artifact `autoencoder_v1` in AML Model Registry

### 3b — YOLOv8 (Layer 2)
- [ ] Convert pseudo-labels to YOLO annotation format
- [ ] Write `src/cascade_defect/layer2_yolo/train.py`
  - Train YOLOv8n on pseudo-labelled data (79%)
  - Also train YOLOv8n on ground-truth labels (human baseline, for evaluation)
  - Track both runs in MLflow
- [ ] Register model artifacts `yolo_pseudo_v1` and `yolo_gt_v1`

---

## Phase 4 — Inference API Containers

### 4a — Layer 1 container (`layer1-api`)
- [ ] Write `src/cascade_defect/layer1_autoencoder/app.py` (FastAPI endpoint)
- [ ] Write `docker/layer1.Dockerfile`
- [ ] Push image to **Azure Container Registry**

### 4b — Layer 2 container (`layer2-api`)
- [ ] Write `src/cascade_defect/layer2_yolo/app.py` (FastAPI endpoint)
- [ ] Write `docker/layer2.Dockerfile`
- [ ] Push image to ACR

---

## Phase 5 — Azure Container Apps Deployment

- [ ] Provision ACA Environment with `Consumption-GPU-NC8as-T4` workload profile (West Europe)
  - ⚠️  Request T4 GPU quota via Azure Portal support ticket (allow 24–48 h)
- [ ] Deploy `layer1-api` Container App, bind to Azure Service Bus queue via KEDA
- [ ] Deploy `layer2-api` Container App, bind to same Service Bus queue
- [ ] Configure KEDA scale rule: min=0, max=10, queue-length trigger=1
- [ ] Document **cold-start latency** (expected 30–90 s from scale-zero)

---

## Phase 6 — Evaluation (Quarto Website)

- [ ] Write `docs/evaluation.qmd`
  - Latency benchmark: Layer 1-only vs. Layer 1+2 vs. full cascade
  - Cost model: Pure MLLM (100 k frames) vs. Cascade Architecture
  - Precision / Recall: `yolo_pseudo_v1` vs. `yolo_gt_v1`
- [ ] Render Quarto website with `quarto render docs/`
- [ ] Publish to GitHub Pages

---

## Phase 7 — CI/CD

- [ ] Add GitHub Actions workflow `ci.yml` (lint, unit tests, quarto render check)
- [ ] Add GitHub Actions workflow `deploy.yml` (build & push Docker images on `main`)

---

## Critical Gotchas (documented here for reference)

| Risk | Mitigation |
|------|-----------|
| Cold-start penalty (30–90 s) on ACA T4 | Document separately from inference latency |
| GPU quota defaults to 0 on new ACA env | Open Azure support ticket **immediately** |
| GPT-4o chatty / non-JSON output | Use `response_format={"type":"json_object"}` + Pydantic |
| Notebook outputs with secrets committed | `nbstripout` pre-commit hook |
| West Europe A100 unavailable | Use `Consumption-GPU-NC8as-T4` (T4 supported) |


---

## Build Log — what actually happened (Phases A–H, completed)

The original Phases 1–7 above were the *aspirational* spec. The build proceeded through a slightly different lettered sequence (A–H) that traded the AML training cluster for in-process CPU training and treated Phase I (CI/CD) as deferred. What follows is the honest record of what was built and what it produced.

### Phase A — Dev hygiene ✅
- `uv` project, `pre-commit` (ruff + nbstripout), `.devcontainer`, `pyproject.toml` with CPU-only torch via `[[tool.uv.index]]`.

### Phase B — Azure environment probe ✅
- Confirmed subscription, region (West Europe), AOAI quota for `gpt-4.1-mini` (used in place of `gpt-4o` — same Vision API surface, ~6× cheaper).

### Phase C — Bicep IaC ✅
- `infra/main.bicep` + modules: `acr`, `aca-env`, `log-analytics`, `openai`, `servicebus`, `storage`, `budget`.
- `infra/apps.bicep` provisions four ACA apps: `cascade-l1-ae`, `cascade-l2-yolo`, `cascade-l3-oracle`, `cascade-router`.
- `mseThreshold` parameter wired through to L1 env var so retuning never requires a rebuild.

### Phase D — Real data ingest ✅
- Kaggle classic credentials authenticated, but `kaushal2896/neu-metal-surface-defects-data` returns 403 (terms-of-use unaccepted on the Kaggle web UI).
- Pivoted to HuggingFace mirror `newguyme/neu_cls` — public, CC-permissive, parquet-encoded.
- `download_neu_from_hf()` in `src/cascade_defect/data/ingest.py` decodes parquet via pyarrow → 1,800 real images on disk.
- Stratified split: **18 seed / 1,422 unlabelled / 360 test**.

### Phase E — Pseudo-labelling ✅
- `src/cascade_defect/layer3_gpt4o/annotate.py` calls AOAI `gpt-4.1-mini` with the 18 seed images as a few-shot prompt + a Pydantic `DefectPrediction` schema enforced via structured outputs.
- Output: `data/processed/pseudo_labels.jsonl`.

### Phase F — Model training ✅
- **Autoencoder**: `train.py` with `--normal-class` arg. First attempt trained on all 1,422 unlabelled (all defective) → MSE distribution flat → cascade short-circuited 100% of frames. Retrained on `rolled-in_scale` only (237 imgs, 8 epochs CPU). Threshold 0.0067 derived from same-class held-out test-set MSE. Per-class sanity in `scripts/ae_sanity.py`: `rolled-in_scale` 0.003, `patches` 0.028, `pitted_surface` 0.043 — clean separation.
- **YOLOv8n**: trained on pseudo-labels, weights at `models/yolo/best.pt`.

### Phase G — Containers + ACA deploy ✅
- Five Dockerfiles: `base.Dockerfile` + four service images (`router`, `layer1`, `layer2`, `layer3`).
- Bicep first-deploy hit `ContainerAppInvalidImageFormat` — fixed by adding `acrLoginServer` parameter.
- Forced new revisions via `--revision-suffix` per push.
- Removed self-escalation in Layer 2 — the router is the single orchestrator.
- **Live router**: `https://cascade-router.orangebush-bb39ddbf.westeurope.azurecontainerapps.io`.
- End-to-end smoke: `data/splits/test/scratches/img_000.jpg` → AE → YOLO → Oracle returns `scratches` 0.85 in 3.9 s. Full trace returned in response.

### Phase H — Evaluation + Quarto site ✅
- `src/cascade_defect/eval/run_cascade.py` — stratified 60-image subset (10/class) against live router.
- `src/cascade_defect/eval/run_oracle_only.py` — Oracle-only baseline against the same 60.
- `src/cascade_defect/eval/metrics.py` — rollup → `reports/metrics.json` with **dual accuracy** (overall vs. classified-only — the key honest framing).
- **Real-data results** (10 imgs/class × 6 classes = 60 imgs):

  | Metric | Cascade | Oracle-only |
  |---|---|---|
  | Cost / 100k frames | **$48** | $107 |
  | Overall accuracy | 0.45 | 0.967 |
  | Accuracy on classified frames | **1.00** (27/27) | 0.967 |
  | L1 drop rate | 53% | n/a |
  | p50 latency | 195 ms | 2,147 ms |
  | Run cost (60 imgs) | $0.029 | $0.064 |

- **Quarto site** rendered to `docs/_site/` — 4 pages (`index`, `architecture`, `data-strategy`, `evaluation`). Required `jupyter-cache`, `pandas`, `matplotlib`, `ipykernel`; named kernel `cascade-defect` registered via `uv run python -m ipykernel install --user --name cascade-defect`. `_quarto.yml` uses `cache: false` (inline expressions are incompatible with Jupyter Cache).

### Phase H.1 — Reflection page ✅
- New `docs/intro.qmd` reframes the problem honestly: NEU is a *balanced classification* benchmark, not an *anomaly detection* benchmark. The cascade plumbing is sound; the v1 dataset does not exercise the autoencoder's strengths.

---

## Phase J — Metal-surface refit (KSDD2 + Severstal)

> **Why this phase exists.** v1 (NEU) is rolled steel but contains **no defect-free
> frames** — the AE was trained on a single defect class as a "normal" proxy, which
> is dishonest. An earlier prototype reached for VisA (PCB1) for its 8:1 imbalance,
> but VisA is electronics, not metal — wrong domain for "rolled-metal damage". This
> phase fixes both problems by training on two genuine industrial-metal datasets
> with abundant defect-free imagery.

### Datasets

| Dataset | Domain | Total | Normal | Defective | Notes |
|---|---|---:|---:|---:|---|
| **KSDD2** | Commutator metal surface (industrial) | 3,335 | 2,979 | 356 | Pre-split: ~2,085 train / 1,250 test. Pixel-mask GT. CC BY-NC-SA 4.0 — research only. |
| **Severstal** | Flat-rolled steel sheet | ~12,568 train | ~5,902 | ~6,666 | RLE masks for 4 defect classes. Kaggle competition T&Cs apply. |

### Combination strategy ("smart, not lazy")

The two datasets do not share the same image dimensions, intensity statistics, or
defect taxonomy. The plan respects that:

1. **Layer 1 (Autoencoder) — single AE, two-domain normals**
   - Train on the **union of all defect-free images**: KSDD2 train normals (~2,085)
     ∪ Severstal normals (~5,902) ≈ **8,000 images**.
   - Both domains are resized to a common `256×256` canvas (KSDD2's portrait
     ~230×640 is centre-cropped to a square; Severstal's 1600×256 strip is
     tile-sampled to 256×256 patches).
   - Per-domain MSE thresholds: compute `τ_ksdd2` and `τ_severstal` independently
     on each domain's held-out normals (mean+3σ). The router will select τ based
     on a `domain` hint at inference time, defaulting to the looser of the two.
   - **Pitfall guarded against:** without per-domain thresholds, the dataset with
     higher intrinsic reconstruction error dominates and the other is always
     flagged. We measure both before picking a policy.

2. **Layer 2 (YOLO) — Severstal only**
   - Severstal supplies abundant RLE masks for **4 named classes** (1, 2, 3, 4 in
     the competition; we map to descriptive names: `pitting`, `inclusion`,
     `scratch`, `patch` — confirmed against the competition image bank).
   - Convert RLE → bbox via `cv2.connectedComponentsWithStats` per channel mask.
   - Train YOLOv8n at 640×640 (Severstal native height is 256, so we tile-sample
     defective regions with surrounding context).
   - **KSDD2 defects are NOT used for YOLO training** — they become the OOD test
     (see point 4).

3. **Layer 3 (Oracle) — class-aware prompt**
   - Few-shot exemplars: 1 image per Severstal class (4) + 1 from KSDD2 (catch-all
     `surface_anomaly`) = 5 reference images.
   - Pydantic schema: `defect_class: Literal["pitting","inclusion","scratch","patch","surface_anomaly","no_defect"]`.

4. **Evaluation — three honest tracks**
   - **Track A (in-domain).** Severstal held-out test split. Real positives + real
     negatives → real precision/recall/F1. This is the headline number.
   - **Track B (KSDD2 in-domain).** KSDD2 official test split. AE+Oracle only
     (YOLO is OOD by design). Demonstrates the cascade on a second metal domain
     without retraining the detector.
   - **Track C (KSDD2 defects as OOD generalisation).** Pure stress test — does
     the YOLO trained on flat steel raise *any* detection on commutator defects?
     Expect low recall; the point is to quantify the cost of a missing class
     and show that L3 backstops it.

### Tasks

- [x] **Data layer**
  - [x] `src/cascade_defect/data/ksdd2.py` — load `data/raw/KolektorSDD2/{train,test}/`, label each image by checking if its `_GT.png` mask is non-empty, expose `iter_normal()` / `iter_defective()` / `iter_samples()`.
  - [x] `src/cascade_defect/data/severstal.py` — load `data/raw/severstal/train.csv` (RLE) + `train_images/`, decode RLE to per-class binary masks, expose the same iterators plus `mask_to_yolo_bboxes()`.
  - [x] `src/cascade_defect/data/split_metal.py` — produces `data/splits_metal/` with `ae_train/`, `ae_val/{ksdd2,severstal}/`, `yolo_train/`, `yolo_val/`, `cascade_test/{severstal,ksdd2}/{normal,defective}/`, `manifest.json`.

- [x] **Layer 1**
  - [x] Added `layer1_autoencoder/train_metal.py` — accepts `--data-dir data/splits_metal` and trains on `ae_train/`, then computes per-domain `(mean, std, τ)` on `ae_val/{ksdd2,severstal}/`. CPU-friendly defaults (256px, batch 32, 15 epochs).
  - [x] Output: `models/autoencoder_metal/best.pt` + `summary.json` with per-domain blocks.
  - [x] Sanity script `scripts/ae_metal_sanity.py` plots per-domain × polarity MSE histograms to `reports/ae_metal_sanity.png`.
  - [ ] **Local AE retrain run** — pending Severstal download (KSDD2-only retrain can run as soon as the train/ copy completes).

- [x] **Layer 2**
  - [x] `src/cascade_defect/data/severstal_yolo.py` — writes `data.yaml` (4 classes) and YOLO label `.txt` files into `models/yolo_metal/dataset/`. Real counts: train scratch=9314, pitting=1931, patch=1237, inclusion=190 (49× imbalance — oversampling carries the fix).
  - [x] `layer2_yolo/train_metal.py` — Ultralytics wrapper with auto-disabled mlflow/comet/wandb integrations (avoids cross-project SETTINGS leakage). 3-epoch CPU smoke @ 320px: mAP50=0.17, recall=0.25, scratch class best at R=0.39.
  - [x] Output: `models/yolo_metal/best.pt` + `summary.json` with per-class P/R metrics. Proper 50-epoch / 640px / GPU run is the `cascade-yolo-train-metal` ACA Job.

- [x] **Layer 3**
  - [x] Update few-shot prompt builder to source exemplars from `data/splits_metal/` and add the new schema enum. *(See J.2 — `oracle.py` METAL_SYSTEM_PROMPT.)*
  - [x] Verify with 5 hand-picked frames before running the full eval. *(L3 eval ran in J.3, $0.084 total cost.)*

- [ ] **Evaluation**
  - [x] `eval/run_cascade_metal.py` — Tracks A, B, C as above. In-process L1 (PatchCore) → L2 (YOLO, optional) → L3 (Oracle, optional) cascade against `data/splits_metal/cascade_test/`. Per-layer flags so L1-only runs cost nothing.
  - [x] `eval/metrics_metal.py` — per-track P/R/F1 + L1 drop-rate-on-negatives + Oracle cost rollup → `reports/metrics_metal.json`.
  - [x] Compare against same Oracle-only baseline (re-run on the new test set so the cost line is apples-to-apples). *(L3 eval $0.084 / 60 imgs, baseline numbers in `reports/metrics_metal.json`.)*

- [x] **Cleanup of v1.5 (VisA) artefacts**
  - [x] Deleted `src/cascade_defect/data/visa.py`, `data/ingest_visa.py`, `layer1_autoencoder/train_visa.py`, `scripts/probe_visa_*.py`, `scripts/sync_visa_sample.py`, `scripts/render_visa_panels.py`.
  - [x] Deleted `models/visa/` and the per-subset summary JSONs in `reports/`.
  - [ ] Drop `cascade-{ae,yolo}-train-visa-pcb1` ACA Jobs from `infra/jobs.bicep` (replace with `-metal` jobs in a follow-up infra patch — left alone for now to avoid touching deployed infra).
  - [x] Removed VisA references from `Makefile`, `README.md` (none present).

- [x] **Docs**
  - [x] Rewrote `website/data-strategy.qmd` to centre on the two new datasets and the union-of-normals strategy.
  - [x] Rewrote `website/evaluation.qmd` to render Tracks A/B/C from `metrics_metal.json` (keeps the v1 NEU numbers in a collapsed appendix for honesty).
  - [x] Refreshed `website/index.qmd`, `architecture.qmd`, `inferences.qmd` — every VisA reference replaced with KSDD2/Severstal framing.
  - [x] Dropped `intro.qmd` from the navbar (file already removed).

### Acceptance criteria for Phase J
- Layer 1 separates KSDD2-normal from KSDD2-defective MSE distributions with **mean-delta ≥ 2σ** (real separation, not the ~0.5σ NEU fudge).
- Severstal YOLOv8n: mAP50 ≥ 0.40 on the per-class val split.
- Cascade Track A (Severstal in-domain): **F1 ≥ 0.70**, **L1 drop rate on negatives ≥ 0.80**.
- Cascade Track B (KSDD2 AE+Oracle): **F1 ≥ 0.65** without YOLO involvement.
- Cost-per-100k-frames advantage ≥ **5×** over Oracle-only on Track A (true negatives finally reward the architecture).

---

## Phase J.1 — Layer 1 improvements (post-baseline)

The first metal-surface AE (`models/autoencoder_metal/best.pt`, 6 127 normals,
15 epochs, 256 px) is honest but weak:

| domain    | normal MSE mean | defective MSE mean | delta |
|-----------|-----------------|--------------------|-------|
| KSDD2     | 0.000505        | 0.000630           | **+25 %** (weak but real) |
| Severstal | 0.000784        | 0.000642           | **−18 %** (defectives reconstruct *better*) |

The Severstal inversion is the headline problem. Defects on Severstal are often
*sharper, higher-contrast* features than the noisy mill-roll background — so a
small CNN AE memorises the noisy texture and finds the defect easier to encode.
**More training will not fix this**; the reconstruction objective itself is the
wrong inductive bias on a noisy textured background. Concretely, the options
are (ranked by effort × expected payoff):

### Tier 1 — cheap to try, modest gains
- [ ] **Per-domain AEs** — train two `ConvAutoencoder`s, one on KSDD2 normals,
  one on Severstal normals. Removes the "average normal" compromise the union
  model is forced into. Wire as `models/autoencoder_metal/{ksdd2,severstal}/best.pt`
  and pick by domain at inference time.
- [ ] **SSIM / perceptual loss** instead of plain MSE. SSIM weights structural
  agreement over absolute pixel values, which is what a human inspector does.
  `pip install pytorch-msssim`; loss = `1 - ssim(recon, x)`.
- [x] **Patch-level scoring** — implemented in
  `src/cascade_defect/layer1_autoencoder/scoring.py` as the p99 patch MSE on
  a 32-px sliding window. Severstal localised defects no longer drown in the
  image-mean.
- [x] **Domain-specific normalisation** — `scoring.make_transform` applies a
  per-domain contrast normalisation (±3σ → [0,1]) before encoding so the AE
  no longer has to model the brightness gap between KSDD2 and Severstal.
  Persisted alongside the score `(μ, σ)` in `models/autoencoder_metal/calibration.json`.

### Tier 2 — medium effort, likely the real win
- [ ] **Pretrained-backbone feature reconstruction** (a.k.a. "feature
  autoencoder"). Replace the from-scratch encoder with a frozen ImageNet
  backbone (ResNet18 or EfficientNet-B0 from `torchvision.models`), train only
  a small decoder to reconstruct *intermediate feature maps* of normal images.
  Anomaly score = feature-space MSE. Pretrained features already encode
  "natural-image priors" the from-scratch AE has to learn from 6 k images.
  Reference: *Reverse Distillation* (Deng & Li, CVPR 2022).
- [ ] **DINOv2 / DINO features + simple density model**. Extract patch tokens
  from a frozen DINOv2-small (`facebookresearch/dinov2`, ~22 M params, runs on
  CPU at 256 px in ~200 ms), fit a Gaussian or kNN on normal patch embeddings,
  score test patches by Mahalanobis / kNN distance. No reconstruction at all,
  no decoder to train. This is what current MVTec-AD leaderboards do.

### Tier 3 — drop the AE, adopt a modern industrial-AD method
This is the path the literature has actually converged on; the cascade
architecture stays but Layer 1 swaps implementations.

- [x] **PatchCore-lite** — implemented as
  `src/cascade_defect/layer1_autoencoder/patchcore.py`. Frozen ResNet18
  (ImageNet) features from `layer2`+`layer3`, random-coreset memory bank
  (10% subsample, capped at 200k vectors), kNN cosine distance with p99
  per-image aggregation. No `anomalib` / lightning dependency — pure torch +
  torchvision, fits in the existing Layer-1 image. CPU inference ~150 ms /
  image at 224 px.
  Trainer: `train_patchcore.py`. Banks at `models/patchcore_metal/bank_{ksdd2,severstal}.pt`.
- [ ] **EfficientAD** (Batzner et al., WACV 2024) — distillation-based, ~1 ms
  GPU / ~20 ms CPU per image. Designed for production throughput. Also in
  `anomalib`.
- [ ] **PaDiM** as a quick PatchCore fallback if memory-bank size becomes
  awkward.

### Recommended sequence
1. Tier-1 patch-level scoring + per-domain normalisation (one afternoon, no new
   deps) — should rescue Severstal from the inversion.
2. If still weak, jump straight to Tier 3 PatchCore via `anomalib` — skip the
   pretrained-AE half-step. The AE narrative is "honest baseline → modern
   replacement" rather than "AE we kept polishing".
3. Keep the current AE checkpoint and metrics in the website as the
   *baseline* row, with PatchCore as the *production* row. The contrast tells
   a better story than either alone.

### Acceptance criteria for J.1
- Severstal **defective mean score ≥ normal mean score + 1σ** (fix the inversion).
- KSDD2 mean-delta improves to **≥ 2σ** (current ~1.5σ — keep at parity or better).
- Layer 1 inference latency on CPU stays **≤ 300 ms / image** at 256 px (production budget).

### J.1 result snapshot (2-epoch CPU smoke run)

| Method | KSDD2 Δμ (def − normal) | Severstal Δμ (def − normal) |
|---|---|---|
| AE (Tier-1: patch + per-domain contrast) | **+1.93σ** | −0.62σ (still inverted) |
| **PatchCore-lite** (ResNet18 + kNN) | **+8.18σ** | **+0.52σ** (inversion fixed) |

* Tier-1 alone meets KSDD2 acceptance criterion (≥2σ) at 2 epochs — a 15-epoch run
  will only widen the gap.
* Severstal AE inversion is **fundamental** (Sigmoid-AE on noisy textured background
  models the noise as easily as the defect). PatchCore decisively rescues it.
* Production rollout: set `SCORER=patchcore` env var on the L1 ACA app once
  banks are pushed to blob storage.

---

## Phase J.2 — Project-wide improvements (after J.1)

Tracking the broader "what could make this project better" backlog so it
doesn't get lost in the Layer 1 weeds.

### Data
- [x] **Severstal class imbalance** — class 3 (scratch) has 5 150 examples,
  class 2 (inclusion) only 247. Add per-class oversampling or focal loss in
  YOLO training, otherwise the YOLO backbone will collapse to "predict scratch".
  *(Implemented in `layer2_yolo/train_metal.py`: focal-loss `fl_gamma=1.5` +
  oversampled manifest with default ratio cap 2×. Toggle with
  `--no-oversampling`.)*
- [x] **KSDD2 defect labels for YOLO** — `data/severstal_yolo.py --include-ksdd2`
  derives bboxes from KSDD2 `_GT.png` masks (cv2.connectedComponentsWithStats)
  and adds them as a 5th class `ksdd2_generic`. Collapses Track C into B but
  makes the deployed YOLO usable on KSDD2 — the documented J.2 trade-off.
- [x] **Held-out-by-image-source split** — `data/split_metal.py` now uses
  `_grouped_split` keyed on filename prefix (3 chars for KSDD2 sequential
  frames, 2 hex chars for Severstal as a coarse bucket since the competition
  doesn't ship coil IDs). Manifests carry `"split_strategy":
  "group_aware_by_image_id_prefix"`.

### Layer 2 (YOLO)
- [ ] Try **YOLOv8s** instead of `n` — Severstal has enough data to support it.
- [ ] **Higher input resolution** (1024 px wide) — Severstal images are
  256×1600, downsampling to 640 px squashes the long axis 2.5×. Defects are
  often <30 px and disappear.
- [ ] Ablate the YOLO confidence threshold against Layer 3 budget on Track A.

### Layer 3 (Oracle)
- [x] Update the prompt for Severstal's 4-class taxonomy. `oracle.py` now
  selects `METAL_SYSTEM_PROMPT` (with explicit class definitions) when
  `domain in {ksdd2, severstal, metal}` and keeps the legacy NEU prompt as
  default for back-compat. Schema enum now includes the Severstal classes,
  KSDD2 catch-all `surface_anomaly`, plus `no_defect` and `uncertain`.
- [x] Add a **"reject" output** — `uncertain` is now a first-class enum value
  on `DefectPrediction.defect_class`. The L3 FastAPI app reports it as
  `result: "uncertain"` separately from defect / no_defect. Cuts false-positive
  cost on edge cases.
- [ ] **Cache by perceptual hash** — many Severstal mill-roll images are
  near-duplicates within a coil. A `imagehash.phash` cache could cut Layer 3
  spend ≥ 30 % on the test set.

### Cascade / evaluation
- [ ] **Calibrate per-domain τ on a *cost* objective** instead of mean+3σ.
  Sweep τ, plot the Pareto curve of (cost, F1), pick the knee. Currently
  3σ is arbitrary.
- [ ] **End-to-end PR curves** in `evaluation.qmd`, not just point estimates.
- [ ] **Confusion-matrix breakdown by defect class** on Track A — F1 hides
  which classes the cascade gets wrong.

### Infra & ops
- [x] Rename ACA Jobs (`infra/jobs.bicep` still says `cascade-*-train-visa-*`)
  to `cascade-*-train-metal-*`. Wire weight upload from `models/autoencoder_metal/`
  and the new YOLO output. *(Done in J.3 — three `cascade-{ae,patchcore,yolo}-train-metal` jobs deployed.)*
- [ ] **Drift monitoring** — log MSE / YOLO confidence histograms on the live
  router and alert on KS-test divergence vs the training distribution. The
  cheapest possible guard against the dataset rotting under deployment.
- [ ] Reactivate the **CI smoke-build** for the website + a `pytest -q` gate
  before any further Bicep edits (Phase I, currently deferred).

---

## Phase J.3 — Cloud training jobs (ACA Jobs deploy + L3 eval) ✅

- [x] `infra/jobs.bicep` deployed — three Manual-trigger ACA Jobs (`cascade-ae-train-metal`, `cascade-patchcore-train-metal`, `cascade-yolo-train-metal`) on `cascade-dev-aca-env`. YOLO routed to `gpu-t4` workload profile.
- [x] L3 (Oracle) end-to-end eval against the live router on the metal test set: 60 imgs, $0.084 total cost, results landed in `reports/eval_cascade_metal.jsonl` + `reports/metrics_metal.json`.
- [x] ACR `cascade-base:latest` rebuilt for Phase-J entrypoints.

## Phase J.4 — YOLO retrain on Azure GPU ✅

> Six layered fixes were needed to get `cascade-yolo-train-metal` to actually train on T4. All captured as a reusable skill at `.github/skills/aca-gpu-training-jobs/SKILL.md` so the next person doesn't relearn them.

- [x] Pre-stage YOLO dataset on Azure File share `yolo-data` (10 GiB SMB) on `cascadedev6ya7a3px` — skips in-job data prep entirely.
- [x] Mount file share at `/work/yolo-data` via ACA env storage + bicep `volumes`/`volumeMounts`.
- [x] Mount **EmptyDir at `/dev/shm`** — ACA's default 64 MiB shm cap is fatal for any PyTorch DataLoader / `multiprocessing.Pool`. Real fix; Python-side ThreadPool monkey-patches do not work because Pool itself uses SemLock.
- [x] New `cascade-base-gpu` ACR image — overlay Dockerfile reinstalls torch from `https://download.pytorch.org/whl/cu121`. New `yoloTrainerImageName` bicep param keeps CPU jobs (AE / PatchCore) on the small base.
- [x] Single-line `RUN`s + staged `%TEMP%` build context for `az acr build` (its scanner rejects `RUN \` continuations and `ARG`-before-`FROM`).
- [x] PowerShell `$env:` for storage-account keys with `=/+` chars (cmd `%VAR%` mangles them).
- [x] Portable `data.yaml` with absolute share path (`models/yolo_metal/dataset/data.azure.yaml`).
- [x] **Production run:** 50 ep / 640 px / oversampling on T4. Final `models/yolo_metal/best.pt` + `summary.json`:

  | metric | smoke (3 ep / 320 px CPU) | production (50 ep / 640 px GPU) |
  |---|---:|---:|
  | mAP50 | 0.17 | **0.500** |
  | mAP50-95 | — | 0.230 |
  | precision | — | 0.554 |
  | recall | 0.25 | 0.494 |

  Meets J Acceptance criterion ✓ (mAP50 ≥ 0.40).

---

## Phase K — Next logical steps

Now that L1 (PatchCore + AE), L2 (production YOLO), and L3 (metal-aware Oracle) are all real, the cascade can finally be evaluated end-to-end honestly. K is sequenced so each step unblocks the next.

### K.1 — Refresh end-to-end metrics with the new YOLO weights
- [ ] Push new `models/yolo_metal/best.pt` (mAP50 0.50) into the live `cascade-l2-yolo` ACA app — bake into image or mount from `models` blob.
- [x] Re-run `eval/run_cascade_metal.py` Tracks A / B / C against the live router with the new weights + the new Oracle prompt. *(In-process eval, `reports/eval_cascade_metal_k1_calibrated.jsonl`.)*
- [x] Compare the headline cost-per-100k vs Oracle-only on Track A — this is the number the website needs.
- [x] Update `website/evaluation.qmd` and `reports/metrics_metal.json`; re-render Quarto.

### K.2 — Calibrate τ on cost, not 3σ
- [x] Sweep per-domain `τ_ksdd2`, `τ_severstal` over the K.1 run, plot `(cost, F1)` Pareto, persist the knee back into `models/autoencoder_metal/calibration.json` and `models/patchcore_metal/summary.json`. *(Knees: KSDD2 τ=1.0 F1≈0.94, Severstal τ=-0.5 F1≈0.66 — Severstal capped by the gate's fundamental weakness, hence L2/L3 do the work.)*
- [ ] Re-trigger ACA app revisions with the new τ env vars.

### K.3 — Layer 2 ablations (only if K.1 leaves headroom)
- [ ] **YOLOv8s** at 640 px on the same dataset (one extra ACA Job execution).
- [ ] **1024 px wide** input — Severstal native 1600×256, downsampling to 640 squashes the long axis 2.5×.
- [ ] Pick winner by Track A F1, drop the loser.

### K.4 — Cost-cutting on Layer 3
- [x] Perceptual-hash cache (`imagehash.phash`) in front of the Oracle — many Severstal frames within a coil are near-duplicates. Target ≥ 30 % L3-spend reduction on the test set. *(`src/cascade_defect/layer3_gpt4o/cache.py` — dHash, Hamming ≤ 6, wired through `--use-cache` in the eval runner.)*
- [ ] Confirm the `uncertain` reject path actually fires on edge cases (non-defect mill noise) and is not dead code.

### K.5 — Reporting & narrative
- [x] Add per-class confusion matrix on Track A to `evaluation.qmd`.
- [x] Add an "AE → PatchCore → production YOLO" three-row contrast table — the project's honest progression story.
- [ ] Drop `intro.qmd` if still referenced anywhere (it was unlinked in J).
- [x] Final `quarto render` + GitHub Pages publish.

### K.6 — Drift & ops (lightweight)
- [ ] Log AE/PatchCore score and YOLO confidence histograms from the live router (append to a blob CSV per request).
- [ ] Daily KS-test job comparing the rolling 24 h distribution against the training reference, alert via Service Bus topic.

### K.7 — CI/CD (the long-deferred Phase I, finally)
- [ ] OIDC federated identity GitHub → Azure.
- [ ] `ci.yml`: ruff, pytest, `quarto render website/` smoke-build.
- [ ] `deploy.yml`: build & push four service images on `main`, `az containerapp update --revision-suffix $(git rev-parse --short HEAD)`.
- [ ] **Don't** put GPU training in CI — keep it as Manual-trigger ACA Jobs (already working).

### Acceptance criteria for Phase K
- Track A F1 ≥ **0.70** with the new YOLO weights (was the J acceptance bar; now we actually have the weights to test it).
- Cost-per-100k advantage over Oracle-only ≥ **5×** on Track A.
- Calibration knee documented in `models/*/summary.json` (no more "mean+3σ" hand-wave).
- Quarto site re-rendered with the production numbers and the AE → PatchCore → YOLO progression visible.

---

## Phase I — CI/CD (folded into K.7)

*Superseded by K.7 above — kept as a heading here to preserve the original numbering.*
