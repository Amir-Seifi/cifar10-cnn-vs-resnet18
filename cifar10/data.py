"""Reproducible CIFAR-10 data preparation for CNN vs ResNet18.

Run with Python 3.10+, PyTorch, torchvision, and NumPy.
No test images are used for training, hyperparameter selection, or normalization.
"""

from __future__ import annotations

import hashlib
import json
import random
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for repeatable experiments."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and random for each DataLoader worker."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stratified_split(
    targets: list[int] | np.ndarray,
    *,
    val_per_class: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint train/validation indices with equal validation count/class."""
    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("targets must be a 1D array")
    if val_per_class <= 0:
        raise ValueError("val_per_class must be positive")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for cls in np.unique(labels):
        cls_indices = np.flatnonzero(labels == cls)
        if len(cls_indices) <= val_per_class:
            raise ValueError(f"Class {cls}: need > {val_per_class} samples")
        shuffled = rng.permutation(cls_indices)
        val_parts.append(shuffled[:val_per_class])
        train_parts.append(shuffled[val_per_class:])

    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(labels)
    return train_idx, val_idx


def _target_fingerprint(targets: list[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(targets, dtype=np.int64).tobytes()).hexdigest()


def get_or_create_split(
    targets: list[int] | np.ndarray,
    artifact_dir: str | Path,
    *,
    seed: int = 42,
    val_per_class: int = 500,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Save a stable split; reject incompatible or incomplete previous splits."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_file = artifact_dir / f"cifar10_split_seed{seed}_val{val_per_class}.npz"
    split_meta_file = split_file.with_suffix(".json")
    labels = np.asarray(targets, dtype=np.int64)
    expected = {
        "seed": seed,
        "val_per_class": val_per_class,
        "num_samples": len(labels),
        "targets_sha256": _target_fingerprint(labels),
    }
    expected_train, expected_val = stratified_split(labels, seed=seed, val_per_class=val_per_class)
    if split_file.exists() or split_meta_file.exists():
        if not (split_file.exists() and split_meta_file.exists()):
            raise RuntimeError("Incomplete split files. Remove both and retry.")
        saved_meta = json.loads(split_meta_file.read_text(encoding="utf-8"))
        if saved_meta != expected:
            raise RuntimeError("Saved split metadata does not match this dataset/config.")
        with np.load(split_file, allow_pickle=False) as arr:
            train_idx, val_idx = arr["train_indices"], arr["val_indices"]
        # Validate membership and order. Shuffle order affects seeded training.
        for saved, reference in ((train_idx, expected_train), (val_idx, expected_val)):
            if saved.dtype.kind not in "iu" or not np.array_equal(saved, reference):
                raise RuntimeError(
                    "Saved split indices disagree with the stratified split for this dataset and seed."
                )
    else:
        train_idx, val_idx = expected_train, expected_val
        np.savez_compressed(split_file, train_indices=train_idx, val_indices=val_idx)
        split_meta_file.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return train_idx, val_idx, split_file


def published_split_id(validation_indices: np.ndarray) -> str:
    """Map the ordered validation IDs to the original experiment's split identifier.

    Call after get_or_create_split validates both index arrays against the seeded
    split. Hashing indices avoids dependence on how NumPy compresses an NPZ.
    """
    raw = np.asarray(validation_indices, dtype="<i8").tobytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "e205673867bbc0ac46124bb02c6c3500c9fb380ea1a76a41c9bac10ba2ef5745"
    ):
        raise ValueError("Validation IDs differ from the published CIFAR-10 split.")
    return "ed196ad4967911730b6e8fb52d0270ae3228e7a7ce0ee900b3c87857658a3221"


def training_mean_std(
    raw_train_images: np.ndarray,
    train_indices: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> tuple[list[float], list[float]]:
    """Compute per-RGB-channel normalization statistics from TRAIN ONLY.

    raw_train_images: uint8 array of shape [N, H, W, 3].
    """
    if raw_train_images.ndim != 4 or raw_train_images.shape[-1] != 3:
        raise ValueError("Expected RGB images of shape [N, H, W, 3]")
    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for start in range(0, len(train_indices), chunk_size):
        ids = train_indices[start : start + chunk_size]
        batch = raw_train_images[ids].astype(np.float64) / 255.0
        sums += batch.sum(axis=(0, 1, 2))
        sq_sums += np.square(batch).sum(axis=(0, 1, 2))
        n_pixels += batch.shape[0] * batch.shape[1] * batch.shape[2]
    if n_pixels == 0:
        raise ValueError("No training pixels")
    means = sums / n_pixels
    stds = np.sqrt(np.maximum(sq_sums / n_pixels - means**2, 0))
    return means.tolist(), stds.tolist()


def cnn_transforms(mean, std):
    from torchvision import transforms

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


def build_dataloaders(
    *,
    data_root: str | Path = "data",
    artifact_dir: str | Path = "artifacts/splits",
    batch_size: int = 128,
    num_workers: int = 2,
    include_test: bool = False,
    image_view: str = "cnn",
    augment: bool = True,
    seed: int = 42,
    val_per_class: int = 500,
) -> tuple[dict[str, DataLoader], dict]:
    """Create CIFAR-10 loaders and metadata for a fair model comparison.

    Full train: 50,000 -> stratified 45,000 train + 5,000 val.
    Official 10,000-image test set stays untouched.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    seed_everything(seed)
    data_root = Path(data_root)
    artifact_dir = Path(artifact_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_train = datasets.CIFAR10(root=data_root, train=True, download=True)
    train_idx, val_idx, split_path = get_or_create_split(
        raw_train.targets, artifact_dir, seed=seed, val_per_class=val_per_class
    )

    stats_path = artifact_dir / f"train_stats_seed{seed}_val{val_per_class}.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if (
            stats.get("split_file") != split_path.name
            or stats.get("split_sha256") != hashlib.sha256(split_path.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Normalization stats do not match the saved split.")
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        if (
            mean.shape != (3,)
            or std.shape != (3,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError(
                "Cached RGB statistics must contain three finite means and positive standard deviations."
            )
        mean, std = mean.tolist(), std.tolist()
    else:
        mean, std = training_mean_std(raw_train.data, train_idx)
        stats = {
            "mean": mean,
            "std": std,
            "source": "training split only, original uint8 RGB pixels / 255",
            "split_file": split_path.name,
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        }
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    if image_view == "cnn":
        train_tf, eval_tf = cnn_transforms(mean, std)
    elif image_view == "resnet":
        from .transfer import resnet_transforms

        train_tf, eval_tf = resnet_transforms()
    else:
        raise ValueError(f"Unknown image view: {image_view}")
    if not augment:
        train_tf = eval_tf

    # Share the pixel array. Each subset owns its transform.
    val_view = copy(raw_train)
    raw_train.transform = train_tf
    val_view.transform = eval_tf
    train_set = Subset(raw_train, train_idx.tolist())
    val_set = Subset(val_view, val_idx.tolist())

    def loader(dataset, *, shuffle: bool, offset: int) -> DataLoader:
        g = torch.Generator().manual_seed(seed + offset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=g,
            drop_last=False,
        )

    loaders = {
        "train": loader(train_set, shuffle=True, offset=0),
        "val": loader(val_set, shuffle=False, offset=1),
    }
    if include_test:
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)
        loaders["test"] = loader(test_set, shuffle=False, offset=2)
    metadata = {
        "seed": seed,
        "classes": raw_train.classes,
        "num_classes": len(raw_train.classes),
        "mean": mean,
        "std": std,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set) if include_test else None,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "split_path": str(split_path),
        "stats_path": str(stats_path),
    }
    return loaders, metadata


def split_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def check_split(config, classes, seed, ids):
    if config["seed"] != seed or config["classes"] != list(classes):
        raise ValueError("Checkpoint seed or class order differs from the dataset")
    if "legacy_split" in config:
        matches = config["legacy_split"] == published_split_id(ids)
    else:
        matches = config["validation_ids_sha256"] == split_digest(ids)
    if not matches:
        raise ValueError("Checkpoint belongs to a different validation split")


def dataset_lock(root):
    """Serialize the first download and split-cache creation across local runs."""
    from filelock import FileLock

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(root / ".dataset.lock")
