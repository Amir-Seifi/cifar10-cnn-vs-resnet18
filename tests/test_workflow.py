"""Exercise the CLI, checkpoint recovery and held-out evaluation on small CPU tensors."""

from functools import partial
from typing import ClassVar
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision
from PIL import Image
from torchvision import datasets

from cifar10 import data, evaluate, train, transfer
from cifar10.checkpoints import build_model, load_selected


class TinyCIFAR:
    classes: ClassVar[list[str]] = [f"class-{i}" for i in range(10)]
    reads: ClassVar[list[bool]] = []

    def __init__(self, root, train, download, transform=None):
        self.reads.append(train)
        self.transform = transform
        self.targets = np.repeat(range(10), 3 if train else 1).tolist()
        rng = np.random.default_rng(3 if train else 5)
        self.data = rng.integers(0, 256, (len(self.targets), 32, 32, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        return (self.transform(image) if self.transform else image), self.targets[index]


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_training_resume_and_paired_evaluation(tmp_path, monkeypatch, cpu_threads):
    monkeypatch.setattr(datasets, "CIFAR10", TinyCIFAR)
    monkeypatch.setattr(evaluate, "CIFAR10", TinyCIFAR)
    monkeypatch.setattr(
        train, "build_dataloaders", partial(data.build_dataloaders, val_per_class=1)
    )
    monkeypatch.setattr(
        evaluate, "stratified_split", partial(data.stratified_split, val_per_class=1)
    )
    monkeypatch.setattr(
        transfer, "resnet18", lambda weights: torchvision.models.resnet18(weights=None)
    )
    TinyCIFAR.reads.clear()
    common = ["--device", "cpu", "--batch-size", "10", "--data-dir", str(tmp_path / "data")]

    def fit(stage, folder, epochs=1, extra=()):
        train.main(
            [stage, "--run-dir", str(tmp_path / folder), "--epochs", str(epochs), *common, *extra]
        )
        return tmp_path / folder / "best.pt"

    cnn = fit("cnn", "cnn")
    fit("cnn", "cnn", 2, ["--resume"])
    full = fit("cnn", "uninterrupted", 2)
    one = torch.load(tmp_path / "cnn" / "last.pt", weights_only=True)
    two = torch.load(full.parent / "last.pt", weights_only=True)
    assert all(torch.equal(v, two["model"][k]) for k, v in one["model"].items())
    assert one["best"]["epoch"] == two["best"]["epoch"]
    del one, two
    fit("cnn", "no-aug", extra=["--no-augmentation"])
    frozen = fit("frozen", "frozen")
    tuned = fit("tuned", "tuned", extra=["--parent", str(frozen)])
    parent_model, _, parent_digest, _ = load_selected(frozen, "frozen")
    child_model, cfg, _, _ = load_selected(tuned, "tuned")
    assert cfg["parent_sha256"] == parent_digest
    for name, value in parent_model.encoder.named_buffers():
        assert torch.equal(value, dict(child_model.encoder.named_buffers())[name])
    del parent_model, child_model
    assert all(TinyCIFAR.reads)
    models = ["--cnn", str(cnn), "--frozen", str(frozen), "--tuned", str(tuned)]
    val = tmp_path / "validation"
    evaluate.main([*common, *models, "--output", str(val)])
    test = tmp_path / "test"
    args = [*common, *models, "--split", "test", "--validation", str(val), "--output", str(test)]
    evaluate.main(args)
    summary = pd.read_csv(test / "summary.csv")
    assert summary.n.tolist() == [10, 10, 10]
    assert (test / "paired_changes.png").is_file()
    assert TinyCIFAR.reads.count(False) == 1
    with (val / "predictions.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="exact checkpoints"):
        evaluate.main([*args[:-1], str(tmp_path / "rejected")])
    assert TinyCIFAR.reads.count(False) == 1


@pytest.mark.parametrize("as_archive", [False, True])
def test_legacy_ablation_stops_before_test_data(tmp_path, as_archive):
    config = {
        "model": "CustomCNN",
        "class_names": ["test"],
        "seed": 42,
        "split_sha256": "original",
        "mean": [0.5] * 3,
        "std": [0.2] * 3,
        "augmentation": "none",
    }
    path = tmp_path / "best_no_aug.pt"
    state = {
        "config": config,
        "epoch": 10,
        "model_state_dict": build_model("cnn", pretrained=False).state_dict(),
    }
    torch.save(state, path)
    if as_archive:
        archive = tmp_path / "ablation.zip"
        with ZipFile(archive, "w") as f:
            f.write(path, "training/best_no_aug.pt")
        path = archive
    assert load_selected(path, "cnn")[1]["augment"] is False
    with pytest.raises(ValueError, match="validation only"):
        evaluate.main(
            [
                "--split",
                "test",
                "--cnn",
                str(path),
                "--frozen",
                "unused.pt",
                "--tuned",
                "unused.pt",
                "--validation",
                "unused",
                "--output",
                str(tmp_path / "test"),
                "--device",
                "cpu",
            ]
        )
