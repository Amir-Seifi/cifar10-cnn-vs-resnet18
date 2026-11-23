"""Selected weights and resumable state for the CIFAR-10 experiments."""

import hashlib
import io
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import torch

from .models import CustomCNN
from .transfer import WEIGHTS, FrozenResNet18, Layer4ResNet18


def build_model(stage, *, pretrained):
    if stage == "cnn":
        return CustomCNN()
    model_class = {"frozen": FrozenResNet18, "tuned": Layer4ResNet18}[stage]
    return model_class(weights=WEIGHTS if pretrained else None)


def save_atomic(payload, destination):
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_selected(path, stage):
    """Load native weights or the selected member of an original training archive."""
    members = {
        "cnn": ("best.pt", "best_no_aug.pt"),
        "frozen": ("best_head.pt",),
        "tuned": ("selected.pt",),
    }
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            matches = [
                p
                for p in archive.namelist()
                if Path(p).name in members[stage] and "__MACOSX" not in Path(p).parts
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one selected {stage} checkpoint in {path}")
            raw = archive.read(matches[0])
    else:
        raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if state.get("format") == "cifar10-selected-v1":
        config = state["config"]
        if config["stage"] != stage:
            raise ValueError(f"Expected a {stage} checkpoint, got {config['stage']}")
        model = build_model(stage, pretrained=False)
        model.load_state_dict(state["model"])
        return model, config, digest, state["epoch"]

    cfg = state["config"]
    if stage == "cnn" and cfg.get("model") == "CustomCNN":
        model = build_model(stage, pretrained=False)
        model.load_state_dict(state["model_state_dict"])
        source = cfg
    elif stage == "frozen" and cfg.get("stage") == "frozen_resnet18_head":
        model = build_model(stage, pretrained=True)
        model.head.load_state_dict(state["head_state_dict"])
        source = cfg
    elif stage == "tuned" and cfg.get("stage") == "resnet18_layer4_and_head":
        model = build_model(stage, pretrained=True)
        model.head.load_state_dict(state["tuned_weights"]["head"])
        model.encoder.layer4.load_state_dict(state["tuned_weights"]["layer4"])
        source = cfg["parent_config"]
    else:
        raise ValueError(f"Unrecognized {stage} checkpoint: {path}")
    if stage != "cnn" and source["weights"] != WEIGHTS.name:
        raise ValueError("Pretrained encoder weights differ from IMAGENET1K_V1")
    config = {
        "stage": stage,
        "classes": source.get("classes", source.get("class_names")),
        "seed": source["seed"],
        "legacy_split": source["split_sha256"],
        "parent_sha256": cfg.get("parent_sha256"),
    }
    if stage == "cnn":
        # The original augmented CNN did not record this field. The ablation stored "none".
        if cfg.get("augmentation") not in (None, "none"):
            raise ValueError("Unknown legacy CNN augmentation setting")
        config.update(mean=cfg["mean"], std=cfg["std"], augment=cfg.get("augmentation") != "none")
    return model, config, digest, state["epoch"]
