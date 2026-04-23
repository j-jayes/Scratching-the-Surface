"""Tests for the NEU dataset split utility."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from cascade_defect.data.split import NEU_CLASSES, split_dataset


@pytest.fixture()
def fake_raw_dir(tmp_path: Path) -> Path:
    """Create a minimal fake NEU dataset structure."""
    rng = random.Random(0)
    for class_name in NEU_CLASSES:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(20):
            img = class_dir / f"{class_name}_{i:03d}.jpg"
            # Write a tiny fake JPEG (1×1 pixel)
            img.write_bytes(b"\xff\xd8\xff\xd9")
    return tmp_path


def test_split_creates_three_dirs(fake_raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "splits"
    split_dataset(fake_raw_dir, out)
    assert (out / "seed").exists()
    assert (out / "unlabelled").exists()
    assert (out / "test").exists()


def test_split_counts_are_correct(fake_raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "splits"
    counts = split_dataset(fake_raw_dir, out, seed_per_class=3, test_fraction=0.20)
    total_raw = sum(1 for c in NEU_CLASSES for _ in (fake_raw_dir / c).glob("*.jpg"))
    assert counts["seed"] + counts["unlabelled"] + counts["test"] == total_raw


def test_seed_has_correct_count_per_class(fake_raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "splits"
    split_dataset(fake_raw_dir, out, seed_per_class=3)
    for class_name in NEU_CLASSES:
        seed_class_dir = out / "seed" / class_name
        assert seed_class_dir.exists(), f"Missing seed dir for {class_name}"
        imgs = list(seed_class_dir.glob("*.jpg"))
        assert len(imgs) == 3, f"Expected 3 seed images for {class_name}, got {len(imgs)}"


def test_no_overlap_between_splits(fake_raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "splits"
    split_dataset(fake_raw_dir, out)

    def get_names(split: str) -> set[str]:
        return {p.name for p in (out / split).rglob("*.jpg")}

    seed_names = get_names("seed")
    unlabelled_names = get_names("unlabelled")
    test_names = get_names("test")

    assert seed_names.isdisjoint(unlabelled_names), "Overlap between seed and unlabelled"
    assert seed_names.isdisjoint(test_names), "Overlap between seed and test"
    assert unlabelled_names.isdisjoint(test_names), "Overlap between unlabelled and test"


def test_missing_class_dir_is_skipped(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    # Only create one class
    (raw / "crazing").mkdir()
    for i in range(10):
        (raw / "crazing" / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    out = tmp_path / "splits"
    counts = split_dataset(raw, out)
    assert counts["seed"] == 3
    assert counts["test"] > 0
