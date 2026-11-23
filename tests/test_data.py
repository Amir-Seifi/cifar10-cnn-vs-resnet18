"""No internet/data download needed for these sanity tests."""
from typing import ClassVar

import numpy as np

from cifar10.data import get_or_create_split, stratified_split, training_mean_std


def test_stratified_split_deterministic_balanced():
    targets = np.repeat(np.arange(10), 20)
    a_train, a_val = stratified_split(targets, val_per_class=5, seed=42)
    b_train, b_val = stratified_split(targets, val_per_class=5, seed=42)
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_val, b_val)
    assert len(a_train) == 150 and len(a_val) == 50
    assert not (set(a_train) & set(a_val))
    assert all((targets[a_val] == i).sum() == 5 for i in range(10))


def test_saved_split_roundtrip(tmp_path):
    targets = np.repeat(np.arange(10), 20)
    first = get_or_create_split(targets, tmp_path, val_per_class=5)
    second = get_or_create_split(targets, tmp_path, val_per_class=5)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_stats_training_only():
    images = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    images[:2, :, :, 0] = 255
    images[2:, :, :, :] = 255
    means, stds = training_mean_std(images, np.array([0, 1]))
    np.testing.assert_allclose(means, [1, 0, 0])
    np.testing.assert_allclose(stds, [0, 0, 0])


def test_build_dataloaders_with_synthetic_cifar(monkeypatch, tmp_path):
    from PIL import Image
    from torchvision import datasets

    from cifar10 import data

    loaded_partitions = []

    class FakeCIFAR10:
        classes: ClassVar[list[str]] = [f"class_{i}" for i in range(10)]

        def __init__(self, root, train=True, download=False, transform=None):
            loaded_partitions.append(train)
            self.transform = transform
            n_per_class = 20 if train else 4
            self.targets = np.repeat(np.arange(10), n_per_class).tolist()
            rng = np.random.default_rng(99 if train else 100)
            self.data = rng.integers(0, 256, size=(len(self.targets), 32, 32, 3), dtype=np.uint8)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            img = Image.fromarray(self.data[idx])
            if self.transform:
                img = self.transform(img)
            return img, self.targets[idx]

    monkeypatch.setattr(datasets, "CIFAR10", FakeCIFAR10)
    loaders, info = data.build_dataloaders(
        data_root=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts/splits",
        batch_size=8,
        num_workers=0,
        include_test=True,
        seed=42,
        val_per_class=5,
    )
    assert info["train_size"] == 150
    assert info["val_size"] == 50
    assert info["test_size"] == 40
    assert all(s > 0 for s in info["std"])
    images, labels = next(iter(loaders["train"]))
    assert tuple(images.shape) == (8, 3, 32, 32)
    assert tuple(labels.shape) == (8,)
    # Repeated runs reuse the same saved split and stats files.
    _, info2 = data.build_dataloaders(
        data_root=tmp_path / "data", artifact_dir=tmp_path / "artifacts/splits",
        batch_size=8, num_workers=0, seed=42, val_per_class=5,
    )
    assert np.array_equal(info["train_indices"], info2["train_indices"])
    assert info["mean"] == info2["mean"]
    loaded_partitions.clear()
    training_only, meta = data.build_dataloaders(
        data_root=tmp_path / "data", artifact_dir=tmp_path / "artifacts/splits",
        val_per_class=5, num_workers=0,
    )
    assert loaded_partitions == [True]
    assert set(training_only) == {"train", "val"} and meta["test_size"] is None
    assert training_only["train"].dataset.dataset.data is training_only["val"].dataset.dataset.data


def test_reject_duplicate_training_index(tmp_path):
    targets = np.repeat(np.arange(3), 15)
    train, val, path = get_or_create_split(targets, tmp_path, val_per_class=3)
    train[1] = train[0]
    np.savez_compressed(path, train_indices=train, val_indices=val)
    with np.testing.assert_raises(RuntimeError):
        get_or_create_split(targets, tmp_path, val_per_class=3)


def test_reject_changed_validation_order(tmp_path):
    targets = np.repeat(np.arange(3), 15)
    train, val, path = get_or_create_split(targets, tmp_path, val_per_class=3)
    np.savez_compressed(path, train_indices=train, val_indices=val[::-1])
    with np.testing.assert_raises(RuntimeError):
        get_or_create_split(targets, tmp_path, val_per_class=3)
