# Skill: Training ResNet50 Locally on M2 MPS

## Entry point

```bash
uv run python -m cascade_defect.classical.train [OPTIONS]
```

Source: `src/cascade_defect/classical/train.py`  
Model definition: `src/cascade_defect/classical/resnet50.py`  
Data loader: `src/cascade_defect/classical/data.py`

---

## Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--train-csv` | `data/splits_metal_v2/train_labels.csv` | CSV with `image_path` and `label` columns |
| `--epochs` | 10 | Max epochs to run **in this session** |
| `--batch-size` | 32 | Works well on 16 GB M2 |
| `--lr` | `3e-4` | Starting LR for a fresh run; use `1e-4` when resuming |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--num-workers` | 2 | DataLoader workers |
| `--out` | `models/resnet50_severstal.pt` | Where the best checkpoint is saved |
| `--history` | `reports/resnet50_train_history.json` | Training history (appended on resume) |
| `--device` | auto-detect | Override with `cpu`, `mps`, or `cuda` |
| `--resume` | None | Path to an existing `state_dict` to fine-tune from |
| `--time-limit-hours` | None | Hard wall-clock stop before each epoch begins |

---

## Fresh training (from ImageNet weights)

```bash
uv run python -m cascade_defect.classical.train \
  --epochs 10 \
  --lr 3e-4
```

- Saves best-val-acc checkpoint to `models/resnet50_severstal.pt`
- Writes full history to `reports/resnet50_train_history.json`
- ~3 min/epoch on M2 MPS; 10 epochs ≈ 30 min

---

## Resume / continue training

Use `--resume` to load a saved checkpoint and keep going. History is **appended**, and epoch numbers continue from where the previous run left off (e.g. epoch 11, 12, 13...).

Use a **lower LR** when resuming to avoid overshooting a good minimum:

```bash
uv run python -m cascade_defect.classical.train \
  --resume models/resnet50_severstal.pt \
  --epochs 200 \
  --lr 1e-4 \
  --time-limit-hours 8.0
```

- Cosine LR anneals from `1e-4` → 0 over `--epochs` steps
- Training stops automatically when the wall-clock exceeds `--time-limit-hours`
- At ~3 min/epoch on M2 MPS, 8 h ≈ 160 epochs

---

## Device auto-detection order

1. MPS (Apple Silicon) — used by default on M2
2. CUDA — if available
3. CPU — fallback

Override explicitly with `--device cpu` when debugging DataLoader issues.

---

## History file format

`reports/resnet50_train_history.json`:

```json
{
  "epochs": [
    {
      "epoch": 11,
      "train_loss": 0.65,
      "val_loss": 0.59,
      "val_acc": 0.741,
      "val_per_class_acc": {"no_defect": 0.78, "pitting": 0.70, ...},
      "lr": 9.9e-05,
      "secs": 181.3
    },
    ...
  ],
  "best_val_acc": 0.7398,
  "device": "mps",
  "classes": ["no_defect", "pitting", "inclusion", "scratch", "patch"],
  "args": { ... }
}
```

When `--resume` is used, existing epochs are **preserved** and new epochs are appended.

---

## Checking training progress

```bash
# Last 3 epochs
python3 -c "
import json
h = json.load(open('reports/resnet50_train_history.json'))
for e in h['epochs'][-3:]:
    print(f\"epoch={e['epoch']} val_acc={e['val_acc']:.4f} lr={e['lr']:.2e} ({e['secs']:.0f}s)\")
print(f\"Best so far: {h['best_val_acc']:.4f}\")
"
```

---

## Class weights

Computed automatically from the training CSV to handle class imbalance.  
Approximate weights for the v2 Severstal split:

| Class | Weight |
|---|---|
| no_defect | 0.43 |
| scratch | 0.51 |
| pitting | 3.10 |
| patch | 3.41 |
| inclusion | 11.91 |

---

## Baseline results (10-epoch fresh run, v2 split)

| Epoch | val_acc | best |
|---|---|---|
| 1 | 0.699 | |
| 4 | 0.735 | ← best |
| 10 | 0.719 | |

- macro-F1 on test set: **0.598**
- Binary F1 (defect/no-defect): **0.858**
