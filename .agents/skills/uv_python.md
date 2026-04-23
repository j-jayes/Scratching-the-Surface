# Skill: Python Package Management with `uv`

## Core Rule
**Always use `uv` instead of `pip` for all Python dependency management.**
Never suggest `pip install`, `pip freeze`, or `virtualenv` commands in this project.

---

## Common `uv` Commands

### Add a runtime dependency
```bash
uv add <package>
# e.g.
uv add ultralytics          # YOLOv8
uv add torch torchvision    # PyTorch (CPU)
uv add "torch==2.3.0+cu121" --index-url https://download.pytorch.org/whl/cu121  # CUDA
uv add openai               # Azure OpenAI SDK
uv add pydantic             # Data validation
uv add fastapi uvicorn      # Inference API
uv add azure-storage-blob azure-identity  # Azure SDKs
```

### Add a development dependency
```bash
uv add --dev ruff pytest pytest-cov nbstripout pre-commit
```

### Sync dependencies (install from lockfile)
```bash
uv sync
```

### Run a script inside the managed environment
```bash
uv run python src/cascade_defect/layer1_autoencoder/train.py
uv run pytest tests/
uv run ruff check src/
```

### Export a `requirements.txt` (for Docker builds only)
```bash
uv export --no-dev --format requirements-txt > requirements.txt
```

---

## `pyproject.toml` Structure

When generating or editing `pyproject.toml`, always follow this structure:

```toml
[project]
name = "cascade-defect"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # add packages here via `uv add`
]

[dependency-groups]
dev = [
    "ruff>=0.4",
    "pytest>=8",
    "nbstripout>=0.7",
    "pre-commit>=3",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

---

## Docker Integration

In `Dockerfile`s, always install from the uv-generated lockfile:

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
CMD ["uv", "run", "uvicorn", "cascade_defect.layer2_yolo.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Never Do This
```bash
# WRONG — do not suggest these:
pip install torch
pip freeze > requirements.txt
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
conda install pytorch
```
