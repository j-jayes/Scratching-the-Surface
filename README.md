# Project Cascade Defect 🏭

> A proof-of-concept ML portfolio project: a cost-effective **Cascade Architecture** for real-time rolled-metal surface defect detection.

[![Copilot Setup Steps](https://github.com/j-jayes/Scratching-the-Surface/actions/workflows/copilot-setup-steps.yml/badge.svg)](https://github.com/j-jayes/Scratching-the-Surface/actions/workflows/copilot-setup-steps.yml)

## What is this?

Instead of running every factory camera frame through an expensive Multimodal LLM (GPT-4o), this system routes frames through three progressively more powerful — and expensive — layers:

| Layer | Model | Task | Latency | Cost/frame |
|-------|-------|------|---------|-----------|
| 1 — Gatekeeper | Convolutional Autoencoder | Binary: Defect vs. No Defect (MSE) | ~5 ms | ~$0.000001 |
| 2 — Specialist | YOLOv8n (T4 GPU, Azure Container Apps) | Classify + Localise defect | ~15 ms | ~$0.000024 |
| 3 — Oracle | GPT-4o (Azure OpenAI) | Edge-case reasoning (few-shot) | ~3,000 ms | ~$0.009 |

**Result:** ~99% cost reduction vs. pure MLLM, with comparable accuracy.

## Portfolio Website

📖 **[View the Quarto website →](https://j-jayes.github.io/Scratching-the-Surface/)**

Covers:
- [System Architecture](docs/architecture.qmd) — Mermaid.js diagrams of the full Azure infrastructure
- [Data Strategy](docs/data-strategy.qmd) — NEU dataset splits + GPT-4o pseudo-labelling
- [Evaluation](docs/evaluation.qmd) — Latency, cost, and accuracy benchmarks

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
├── docs/                    # Quarto portfolio website
│   ├── _quarto.yml
│   ├── index.qmd
│   ├── architecture.qmd
│   ├── data-strategy.qmd
│   └── evaluation.qmd
├── src/cascade_defect/      # Python source package
│   ├── data/split.py        # NEU dataset split utility
│   ├── layer1_autoencoder/  # Conv AE model, training, FastAPI app
│   ├── layer2_yolo/         # YOLOv8 inference FastAPI app
│   └── layer3_gpt4o/        # GPT-4o oracle FastAPI app
├── tests/                   # pytest unit tests
└── pyproject.toml           # uv-managed project config
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| `uv` | Python package manager (replaces pip) |
| `ruff` | Linter + formatter |
| `nbstripout` | Strip notebook outputs before commit |
| `pre-commit` | Git hooks for code quality |
| Azure ML | Model training on `Standard_NC6s_v3` GPU |
| Azure Container Apps | Serverless GPU inference (T4, West Europe) |
| Azure Service Bus + KEDA | Scale-to-zero event-driven autoscaling |
| Azure Data Lake Gen2 | Raw data + pseudo-label storage |
| Azure OpenAI (GPT-4o) | Few-shot edge-case classification |
| Quarto | Portfolio website with Mermaid.js diagrams |

## Gotchas

- **GPU quota:** ACA T4 GPU quota defaults to 0. Open an Azure support ticket immediately (allow 24–48 h).
- **Cold-start:** Scaling from zero with a T4 takes 30–90 s. Document this separately from inference latency.
- **West Europe A100:** Not available for ACA. Use `Consumption-GPU-NC8as-T4` instead.
- **GPT-4o JSON:** Always use `client.beta.chat.completions.parse()` with a Pydantic model to enforce schema.

## Licence

MIT