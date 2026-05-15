"""Train ResNet50 on the v2 Severstal split (M2 MPS by default).

Usage::

    uv run python -m cascade_defect.classical.train \
        --train-csv data/splits_metal_v2/train_labels.csv \
        --epochs 10 --batch-size 32 --out models/resnet50_severstal.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import class_weights_from_csv, make_loaders
from .resnet50 import CLASSES, build_model

ROOT = Path(__file__).resolve().parents[3]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device,
             criterion: nn.Module) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    per_class_correct = [0] * len(CLASSES)
    per_class_total = [0] * len(CLASSES)
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        for c in range(len(CLASSES)):
            mask = y == c
            per_class_total[c] += int(mask.sum().item())
            per_class_correct[c] += int(((pred == y) & mask).sum().item())
    per_cls_acc = {
        CLASSES[c]: (per_class_correct[c] / per_class_total[c])
        if per_class_total[c] else None
        for c in range(len(CLASSES))
    }
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "per_class_acc": per_cls_acc,
        "n": total,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv",
                   default="data/splits_metal_v2/train_labels.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default="models/resnet50_severstal.pt")
    p.add_argument("--history", default="reports/resnet50_train_history.json")
    p.add_argument("--device", default=None,
                   help="Override device (cpu/mps/cuda).")
    p.add_argument("--resume", default=None,
                   help="Path to a saved state_dict to resume from.")
    p.add_argument("--time-limit-hours", type=float, default=None,
                   help="Stop training after this many hours (wall-clock).")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Device: {device}")

    train_csv = ROOT / args.train_csv
    out_path = ROOT / args.out
    history_path = ROOT / args.history
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(
        train_csv, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    weights = class_weights_from_csv(train_csv).to(device)
    print(f"Class weights: {dict(zip(CLASSES, weights.tolist()))}")

    model = build_model(pretrained=True).to(device)
    if args.resume:
        resume_path = ROOT / args.resume
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed weights from {resume_path}")
    optim = AdamW(model.parameters(), lr=args.lr,
                  weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history: list[dict] = []
    best_val_acc = -1.0
    best_state: dict | None = None

    # Load existing history if resuming so we can append to it
    if args.resume and history_path.exists():
        prev = json.loads(history_path.read_text())
        history = prev.get("epochs", [])
        best_val_acc = prev.get("best_val_acc", -1.0)
        print(f"Loaded {len(history)} prior epochs; best_val_acc so far = {best_val_acc:.4f}")

    time_limit_secs = args.time_limit_hours * 3600 if args.time_limit_hours else None
    wall_start = time.time()
    epoch_offset = len(history)  # so epoch numbers continue from prior run

    for epoch in range(1, args.epochs + 1):
        # Check wall-clock time limit before starting each epoch
        if time_limit_secs is not None:
            elapsed = time.time() - wall_start
            if elapsed >= time_limit_secs:
                print(f"Time limit of {args.time_limit_hours:.1f}h reached "
                      f"after {elapsed/3600:.2f}h. Stopping.")
                break
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / max(seen, 1)
        sched.step()

        val = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        rec = {
            "epoch": epoch_offset + epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_per_class_acc": val["per_class_acc"],
            "lr": optim.param_groups[0]["lr"],
            "secs": dt,
        }
        history.append(rec)
        elapsed_h = (time.time() - wall_start) / 3600
        print(f"[{epoch_offset + epoch:03d}] "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
              f"({dt:.1f}s, wall {elapsed_h:.2f}h)")

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save(best_state, out_path)
            print(f"  ↳ new best val_acc={best_val_acc:.4f}, saved to {out_path}")

    history_path.write_text(json.dumps(
        {"epochs": history, "best_val_acc": best_val_acc,
         "device": str(device), "classes": CLASSES,
         "args": vars(args)}, indent=2))
    print(f"Wrote history to {history_path}")


if __name__ == "__main__":
    main()
