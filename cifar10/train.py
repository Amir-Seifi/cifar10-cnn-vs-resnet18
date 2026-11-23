"""Train, resume and select CIFAR-10 classifiers."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch
from filelock import FileLock

from .checkpoints import build_model, load_selected, save_atomic
from .data import (
    build_dataloaders,
    check_split,
    dataset_lock,
    seed_everything,
    split_digest,
)
from .reporting import plot_history
from .training import run_epoch


def new_optimizer(model, stage):
    if stage == "tuned":
        return torch.optim.Adam(
            [
                {"params": model.encoder.layer4.parameters(), "lr": 1e-5},
                {"params": model.head.parameters(), "lr": 1e-4},
            ]
        )
    params = model.parameters() if stage == "cnn" else model.head.parameters()
    return torch.optim.Adam(params, lr=1e-3)


def selected_state(model, config, epoch, loss, accuracy):
    return {
        "format": "cifar10-selected-v1",
        "config": config,
        "epoch": epoch,
        "val_loss": loss,
        "val_accuracy": accuracy,
        "model": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
    }


def train(args):
    stage, output = args.stage, args.run_dir
    last_path = output / "last.pt"
    if not args.resume and (output / "best.pt").exists():
        raise FileExistsError(f"Run directory already contains selected weights: {output}")
    if args.resume != last_path.exists():
        raise ValueError("Use --resume for an existing run, or choose a new --run-dir")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with dataset_lock(args.data_dir):
        loaders, info = build_dataloaders(
            data_root=args.data_dir,
            artifact_dir=args.data_dir / "splits",
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=0,
            include_test=False,
            image_view="cnn" if stage == "cnn" else "resnet",
            augment=not args.no_augmentation,
        )
    config = {
        "stage": stage,
        "classes": list(info["classes"]),
        "seed": args.seed,
        "validation_ids_sha256": split_digest(info["val_indices"]),
        "batch_size": args.batch_size,
        "augment": not args.no_augmentation,
        "parent_sha256": None,
        "mean": info["mean"],
        "std": info["std"],
    }
    seed_everything(args.seed)
    model = build_model(stage, pretrained=stage == "frozen" and not args.resume)
    if stage == "tuned":
        parent, parent_config, digest, _ = load_selected(args.parent, "frozen")
        check_split(parent_config, info["classes"], args.seed, info["val_indices"])
        config["parent_sha256"] = digest
        model.load_state_dict(parent.state_dict())
        del parent

    model.to(device)
    optimizer = new_optimizer(model, stage)
    criterion = torch.nn.CrossEntropyLoss()
    history, best = [], None
    if args.resume:
        state = torch.load(last_path, weights_only=True, map_location="cpu")
        if state["config"] != config:
            raise ValueError("Resume settings differ from the saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best = state["history"], state["best"]
        torch.set_rng_state(state["rng"])
        for name, loader in loaders.items():
            loader.generator.set_state(state["loader_rng"][name])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        del state
    elif stage == "tuned":
        baseline = run_epoch(model, loaders["val"], criterion, device)
        best = selected_state(model, config, 0, baseline["loss"], baseline["accuracy"])
        loaders["val"].generator.manual_seed(args.seed + 1)

    if args.epochs < len(history):
        raise ValueError("--epochs cannot be lower than the number of completed epochs")
    for epoch in range(len(history) + 1, args.epochs + 1):
        started = perf_counter()
        training = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer)
        val = run_epoch(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": training["loss"],
            "train_accuracy": training["accuracy"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "train_seen": training["seen"],
            "val_seen": val["seen"],
            "elapsed_seconds": round(perf_counter() - started, 2),
        }
        history.append(row)
        if best is None or val["loss"] < best["val_loss"]:
            best = selected_state(model, config, epoch, val["loss"], val["accuracy"])
        save_atomic(
            {
                "config": config,
                "history": history,
                "best": best,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "loader_rng": {k: v.generator.get_state() for k, v in loaders.items()},
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
            last_path,
        )
        print(
            f"Epoch {epoch} | train {training['accuracy']:.2%} | "
            f"validation {val['accuracy']:.2%} | validation loss {val['loss']:.4f}",
            flush=True,
        )

    save_atomic(best, output / "best.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output / "learning_curves.png")
    print(f"Selected epoch {best['epoch']} | {output / 'best.pt'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cnn", "frozen", "tuned"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--parent", type=Path, help="Selected frozen checkpoint for fine-tuning")
    parser.add_argument("--epochs", type=int, help="Total epochs (default 10, or 3 for tuned)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true", help="CNN ablation")
    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = 3 if args.stage == "tuned" else 10
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if bool(args.parent) != (args.stage == "tuned"):
        parser.error("supply --parent only and always for the tuned stage")
    if args.no_augmentation and args.stage != "cnn":
        parser.error("--no-augmentation applies to the CNN")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(args.run_dir / ".training.lock", timeout=0):
        train(args)


if __name__ == "__main__":
    main()
