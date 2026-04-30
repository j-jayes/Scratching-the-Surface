# Project Cascade Defect 🏭

> A proof-of-concept ML portfolio project: a cost-effective **Cascade Architecture** for real-time rolled-metal surface defect detection.

[![Copilot Setup Steps](https://github.com/j-jayes/Scratching-the-Surface/actions/workflows/copilot-setup-steps.yml/badge.svg)](https://github.com/j-jayes/Scratching-the-Surface/actions/workflows/copilot-setup-steps.yml)

## What is this?

Instead of running every factory camera frame through an expensive Multimodal LLM, this system routes frames through three progressively more powerful — and expensive — layers:

| Layer | Model | Task | Median latency | Cost/frame |
|-------|-------|------|---------------:|-----------:|
| 1 — Gatekeeper | **PatchCore-lite** (frozen ResNet18 + kNN, per-patch p99 score) | Binary: anomalous vs. clean | ~250 ms (CPU) | ~$0 |
| 2 — Specialist | **YOLOv8n** (50 ep / 640 px / T4) on Severstal defectives | Classify + localise defect | ~300 ms (CPU eval) | ~$0 |
| 3 — Oracle | **gpt-4.1-mini** via Azure OpenAI, few-shot, dHash-cached | Edge-case reasoning + final class | ~3,400 ms | ~$0.001 |

**Headline numbers** (calibrated, Phase K, 60 frames/track):

| Track | Domain | F1 | Cost / 100k frames |
|---|---|---:|---:|
| A | Severstal in-distribution | 0.65 | $16.84 |
| B | KSDD2 in-distribution | 0.89 | $52.53 |
| C | KSDD2 OOD stress (Severstal-trained YOLO) | 0.86 | $40.74 |

A pure-Oracle baseline on the same frames runs ~$320 / 100k — a 6–20× cost saving while keeping competitive F1, with the perceptual-hash cache pushing Track C savings further (47% L3 hit-rate on near-duplicate KSDD2 frames).

## Portfolio Website

📖 **[View the Quarto website →](https://j-jayes.github.io/Scratching-the-Surface/)**

Covers:
- [System Architecture](website/architecture.qmd) — Mermaid.js diagrams of the full Azure infrastructure
- [Data Strategy](website/data-strategy.qmd) — KSDD2 + Severstal splits, weak-label workflow
- [Evaluation](website/evaluation.qmd) — Phase K calibration, cost/F1 Pareto, AE→PatchCore→YOLO progression, per-class confusion
- [Inference walkthroughs](website/inferences.qmd) — three end-to-end frames (L1 drop / L2 catch / L3 backstop) with input + PatchCore heatmap + YOLO overlay + router trace + Oracle reasoning

## Quick Start (DevContainer / Codespaces)

```bash
# Open in GitHub Codespaces or VS Code DevContainer
# All tools (uv, Azure CLI, Quarto) are pre-installed

# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Preview the Quarto website
quarto preview docs/
```

## Project Structure

```
.
├── .agents/skills/          # Copilot domain-knowledge instructions
│   ├── azure_container_apps.md
│   ├── uv_python.md
│   └── mermaid_syntax.md
├── .claude/plans/           # Project implementation plan (marked progress)
├── .devcontainer/           # VS Code / GitHub Codespaces container config
├── .github/workflows/       # Copilot setup steps + CI
├── .pre-commit-config.yaml  # nbstripout + ruff hooks
├── website/                 # Quarto portfolio source (renders to docs/)
│   ├── _quarto.yml
│   ├── index.qmd
│   ├── architecture.qmd
│   ├── data-strategy.qmd
│   ├── evaluation.qmd
│   └── inferences.qmd
├── docs/                    # Rendered website (GitHub Pages)
├── src/cascade_defect/      # Python source package
│   ├── data/                # split_metal, severstal_yolo, ksdd2, severstal
│   ├── layer1_autoencoder/  # Conv AE + PatchCore-lite + FastAPI app
│   ├── layer2_yolo/         # YOLOv8 train_metal + FastAPI app
│   ├── layer3_gpt4o/        # Azure OpenAI oracle + dHash cache + FastAPI app
│   ├── eval/                # run_cascade_metal, metrics_metal, threshold_sweep
│   └── router.py            # End-to-end cascade orchestrator
├── scripts/                 # ae_metal_sanity, render_inference_panels_metal
├── infra/                   # Bicep — apps, jobs, storage, ACR, OpenAI
├── docker/                  # Base + per-layer Dockerfiles (incl. CUDA overlay)
├── reports/                 # metrics_metal.json, threshold_sweep.json
├── tests/                   # pytest unit tests
└── pyproject.toml           # uv-managed project config
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| `uv` | Python package manager (Python 3.11) |
| `ruff` | Linter + formatter |
| `nbstripout` | Strip notebook outputs before commit |
| `pre-commit` | Git hooks for code quality |
| PyTorch + Ultralytics YOLOv8 | Layer 2 detector (50 ep / 640 px on T4) |
| Frozen ResNet18 + kNN | PatchCore-lite L1 anomaly gate |
| Azure Container Apps Jobs | GPU training (CUDA 12.1 overlay, Azure Files mount) |
| Azure Container Apps | Serverless inference (per-layer scale-to-zero) |
| Azure Service Bus + KEDA | Event-driven autoscaling |
| Azure Blob Storage | Datasets + model weights + pseudo-labels |
| Azure OpenAI (`gpt-4.1-mini`) | Few-shot edge-case classification with dHash cache |
| Quarto | Portfolio website with Mermaid.js diagrams |

## Reproduce the Phase K headline numbers

```powershell
# In-process eval (same code paths as the live ACA router) with calibrated
# per-domain τ and the perceptual-hash Oracle cache.
uv run python -m cascade_defect.eval.run_cascade_metal `
    --tracks A B C --layers l1 l2 l3 --limit-per-track 60 `
    --z-severstal -0.5 --z-ksdd2 1.0 --use-cache `
    --out reports/eval_cascade_metal_k1_calibrated.jsonl

Copy-Item reports/eval_cascade_metal_k1_calibrated.jsonl `
    reports/eval_cascade_metal.jsonl -Force
uv run python -m cascade_defect.eval.metrics_metal
uv run python scripts/render_inference_panels_metal.py
cd website; quarto render
```

## Gotchas

- **GPU quota:** ACA T4 GPU quota defaults to 0. Open an Azure support ticket immediately (allow 24–48 h).
- **Cold-start:** Scaling from zero with a T4 takes 30–90 s. Document this separately from inference latency.
- **West Europe A100:** Not available for ACA. Use `Consumption-GPU-NC8as-T4` instead.
- **ACA ephemeral storage:** Mount EmptyDir at `/dev/shm` for any DataLoader using `num_workers > 0` — the default 64 MiB cap will Bus-error YOLO training silently.
- **AOAI structured output:** Always use `client.beta.chat.completions.parse()` with a Pydantic model to enforce the defect-class schema.
- **PatchCore on Severstal:** Δμ caps at +0.52σ regardless of τ — the gate is fundamentally weak on rolled-coil texture, by design carried by L2 + L3.

## Licence

MIT