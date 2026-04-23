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

- [ ] Download NEU Metal Surface Defects Database from Kaggle (`data/raw/`)
- [ ] Write `src/cascade_defect/data/split.py` — stratified split logic
  - Few-Shot Seed   : 3 images × 6 classes  =  18 images  (1%)
  - Unlabeled Pool  : ~1,420 images          (79%)
  - Golden Test Set : 360 images             (20%)
- [ ] Upload split data to **Azure Data Lake Storage Gen2** (`az storage` commands)
- [ ] Register dataset in **Azure ML Workspace**

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
