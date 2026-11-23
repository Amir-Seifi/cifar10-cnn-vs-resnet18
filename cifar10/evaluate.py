"""Evaluate selected models on aligned validation or test images."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from .checkpoints import load_selected
from .data import check_split, cnn_transforms, dataset_lock, stratified_split
from .reporting import write_scores
from .transfer import WEIGHTS


class ImageViews(Dataset):
    def __init__(self, dataset, ids, transforms):
        self.dataset = dataset
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, position):
        image_id = int(self.ids[position])
        image, label = self.dataset[image_id]
        return (
            image_id,
            label,
            {key: transform(image) for key, transform in self.transforms.items()},
        )


@torch.inference_mode()
def predict(dataset, ids, transforms, models, device, batch_size):
    loader = DataLoader(ImageViews(dataset, ids, transforms), batch_size=batch_size)
    frames = []
    for image_ids, labels, views in loader:
        batch = {"image_id": image_ids.numpy(), "true_id": labels.numpy()}
        inputs = {key: images.to(device) for key, images in views.items()}
        for key, model in models.items():
            scores = model(inputs["cnn" if key == "cnn" else "resnet"])
            if (
                scores.shape != (len(labels), len(dataset.classes))
                or not torch.isfinite(scores).all()
            ):
                raise ValueError(f"Invalid logits from {key}")
            batch[key] = scores.argmax(dim=1).cpu().numpy()
        frames.append(pd.DataFrame(batch))
    return pd.concat(frames, ignore_index=True)


def check_validation_replay(directory, digests, raw, ids, transforms, models, device, batch_size):
    manifest = json.loads((directory / "protocol.json").read_text())
    csv_path = directory / "predictions.csv"
    if (
        manifest["split"] != "validation"
        or manifest["checkpoint_sha256"] != digests
        or hashlib.sha256(csv_path.read_bytes()).hexdigest() != manifest["predictions_sha256"]
    ):
        raise ValueError("Supply the validation output from these exact checkpoints")
    frame = pd.read_csv(csv_path).sort_values("image_id")
    if not np.array_equal(frame.image_id, sorted(ids)):
        raise ValueError("Validation image IDs differ from the training split")
    sample = frame.groupby("true_id", sort=False).head(2).reset_index(drop=True)
    replay = predict(raw, sample.image_id.to_numpy(), transforms, models, device, batch_size)
    if not replay.equals(sample):
        raise ValueError("Validation predictions changed before test evaluation")


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(f"Choose an unused output directory: {args.output}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models, configs, digests, epochs = {}, {}, {}, {}
    for key in ("cnn", "frozen", "tuned"):
        path = getattr(args, key)
        if path is not None:
            model, configs[key], digests[key], epochs[key] = load_selected(path, key)
            if key == "cnn" and args.split == "test" and not configs[key]["augment"]:
                raise ValueError("The augmentation ablation is evaluated on validation only")
            models[key] = model.to(device).eval()
    if "tuned" in configs and configs["tuned"]["parent_sha256"] != digests["frozen"]:
        raise ValueError("The tuned checkpoint and the selected frozen parent do not match")
    cnn = configs["cnn"]
    transforms = {"cnn": cnn_transforms(cnn["mean"], cnn["std"])[1]}
    if "frozen" in models:
        transforms["resnet"] = WEIGHTS.transforms()
    with dataset_lock(args.data_dir):
        raw = CIFAR10(args.data_dir, train=True, download=True)
    _, ids = stratified_split(raw.targets, seed=cnn["seed"])
    for config in configs.values():
        check_split(config, raw.classes, cnn["seed"], ids)
    if args.split == "test":
        check_validation_replay(
            args.validation, digests, raw, ids, transforms, models, device, args.batch_size
        )
        del raw
        with dataset_lock(args.data_dir):
            raw = CIFAR10(args.data_dir, train=False, download=True)
        if list(raw.classes) != cnn["classes"]:
            raise ValueError("Test class order differs from the training checkpoint")
        ids = np.arange(len(raw))
    frame = predict(raw, ids, transforms, models, device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        report = Path(temp) / "report"
        report.mkdir()
        csv_path = report / "predictions.csv"
        frame.to_csv(csv_path, index=False)
        summary = write_scores(frame, list(raw.classes), list(models), report)
        protocol = {
            "split": args.split,
            "seed": cnn["seed"],
            "classes": raw.classes,
            "checkpoint_sha256": digests,
            "selected_epoch": epochs,
            "device": str(device),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "predictions_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        }
        (report / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        report.rename(args.output)
    print(summary.to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--cnn", required=True, type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--tuned", type=Path)
    parser.add_argument(
        "--validation", type=Path, help="Matching validation report for test evaluation"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    if bool(args.frozen) != bool(args.tuned):
        parser.error("supply --frozen and --tuned together")
    if args.split == "test" and (not args.validation or not args.tuned):
        parser.error("test evaluation requires all three selected models and --validation")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
