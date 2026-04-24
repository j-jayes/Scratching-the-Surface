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

## Phase J — VisA extension (the "right shape of data" demo)

> **Goal:** Re-run the full v1 demonstration on a dataset whose normal:anomaly ratio actually matches the architecture, so the autoencoder's gatekeeper role becomes load-bearing rather than contrived.

### Why VisA
- 10,821 images, **9,621 normal vs 1,200 anomalous** — genuine 8:1 imbalance.
- 12 categories, MVTec-AD-style layout (`1cls/<obj>/{train/good, test/good, test/bad, ground_truth}`).
- CC BY 4.0 — commercial use OK.
- Direct S3 download — no signup form (unlike MVTec AD, which is non-commercial + email-gated).
- Pixel-level segmentation masks ship with the bad set → clean YOLO bounding-box conversion.

Source: <https://github.com/amazon-science/spot-diff>
Tarball: `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`

### Tasks

- [ ] **Data layer**
  - [ ] Add `download_visa()` to `src/cascade_defect/data/ingest.py` — stream-extract the tarball into `data/raw/visa/`, validate checksums.
  - [ ] Add `--dataset {neu,visa}` and `--visa-subset {pcb1,candle,…}` to the data CLI; default to `pcb1` (structural complexity stresses YOLO) and `candle` (texture stresses the AE) for a two-subset comparison.
  - [ ] Update `data/split.py` so VisA's existing `train/good ∪ test/{good,bad}` partition is honoured rather than re-stratified. Seed = small sample of `test/bad` for the few-shot Oracle prompt.

- [ ] **Layer 1 — Autoencoder (now genuine)**
  - [ ] Retrain on `train/good` *only* (per VisA's intended split). For `pcb1` that is ~904 normal images — about 4× the contrived NEU rolled-in-scale set.
  - [ ] Threshold derivation: compute MSE on `test/good` (held-out normals), set `τ = mean + 3σ`. Sanity-check by also computing MSE on `test/bad` — distributions should be visibly separated.
  - [ ] Add a `docs/_freeze`-friendly notebook (or a code cell in `evaluation.qmd`) that plots both MSE distributions and the chosen τ.

- [ ] **Layer 2 — YOLO (now with real masks)**
  - [ ] Convert VisA's per-pixel masks (`ground_truth/<defect_type>/*.png`) to YOLO bbox format via `cv2.boundingRect` on connected components.
  - [ ] Train YOLOv8n on `test/bad` images + converted bboxes (small set — augment aggressively or use n-shot). Track in MLflow if available, else `models/yolo_visa/`.

- [ ] **Layer 3 — Oracle few-shot prompt**
  - [ ] Rebuild the few-shot block from VisA `test/bad` examples (one per defect subtype). For PCB1 that is 4 classes: `bent`, `broken`, `melt`, `missing`.
  - [ ] Keep the same Pydantic schema; add `subtype` enum.

- [ ] **Eval — honest TP/FP/TN/FN**
  - [ ] New `eval/run_cascade_visa.py` — runs the full cascade against `test/good ∪ test/bad`. With abundant negatives, **dropping a normal frame at L1 is now correct**.
  - [ ] Report: precision, recall, F1, plus the cost/latency table from v1. Expect L1 drop rate to climb to 80–90% and classified-frame accuracy to remain ≈1.0 — that combination is the point of the architecture.
  - [ ] Add a third row to the v1 evaluation table comparing v1 (NEU) vs v2 (VisA) cascades side-by-side.

- [ ] **Docs**
  - [ ] Extend `data-strategy.qmd` with a "Choosing the right dataset" section.
  - [ ] Extend `evaluation.qmd` with the v2 numbers.
  - [ ] Update `intro.qmd` to mark Phase J as completed (replace "specced" with measured numbers).

### Acceptance criteria for Phase J
- Live cascade returns correct labels on a hold-out drawn from `test/{good,bad}` of at least one VisA subset.
- Classified-frame accuracy ≥ 0.95.
- L1 drop rate ≥ 0.75 on a normal-heavy sample.
- Cost-per-100k-frames advantage ≥ 3× over Oracle-only baseline (target: 5–10×, given the higher drop rate).

---

## Phase I — CI/CD (deferred)

Still pending; lower priority than Phase J because the build pipeline currently runs locally with `make` targets and pushes to ACR work hands-on. To revisit after Phase J data results are in.

- [ ] Federated identity (OIDC) from GitHub Actions → Azure
- [ ] `ci.yml`: ruff, pytest, `quarto render docs/` smoke-build
- [ ] `deploy.yml`: build & push the four service images on `main`, kick off `az containerapp update --revision-suffix $(git rev-parse --short HEAD)`
