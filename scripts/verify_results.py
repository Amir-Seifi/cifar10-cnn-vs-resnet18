"""Recalculate the published metrics from per-image predictions using Python alone."""

import csv
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
MODELS = ("cnn", "frozen", "tuned")
HISTORIES = {
    "cnn": "custom_cnn.csv",
    "frozen": "resnet18_frozen.csv",
    "tuned": "resnet18_finetuned.csv",
}


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(value, expected, description):
    check(math.isclose(float(value), expected, rel_tol=1e-10, abs_tol=1e-12), description)


def indexed(rows, *keys):
    mapping = {tuple(row[k] for k in keys): row for row in rows}
    check(len(rows) == len(mapping), f"Duplicate row for {keys}")
    return mapping


def metrics(truth, guesses, classes):
    labels = range(len(classes))
    check(all(v in labels for v in guesses), "Prediction outside class range")
    matrix = Counter(zip(truth, guesses))
    actual = Counter(truth)
    predicted = Counter(guesses)
    hits = sum(matrix[i, i] for i in labels)
    per_class = {}
    for i, name in enumerate(classes):
        tp = matrix[i, i]
        per_class[name] = {
            "support": actual[i], "recall": tp / actual[i],
            "precision": tp / predicted[i] if predicted[i] else 0.0,
            "f1": 2 * tp / (actual[i] + predicted[i]),
        }
    macro = sum(row["f1"] for row in per_class.values()) / len(classes)
    return hits, macro, per_class, matrix


def selected_epoch(path, epochs):
    history = read(path)
    check([int(r["epoch"]) for r in history] == list(range(1, epochs + 1)), f"Epoch sequence in {path.name}")
    for row in history:
        check((int(row["train_seen"]), int(row["val_seen"])) == (45_000, 5_000), f"Split sizes in {path.name}")
        for prefix in ("train", "val"):
            loss = float(row[f"{prefix}_loss"])
            acc = float(row[f"{prefix}_accuracy"])
            check(math.isfinite(loss) and loss >= 0 and 0 <= acc <= 1, f"Epoch metrics in {path.name}")
    return min(history, key=lambda r: float(r["val_loss"]))


def change_type(frozen_correct, tuned_correct):
    return {(True, True): "both_correct", (False, False): "both_wrong",
            (False, True): "corrected", (True, False): "regressed"}[frozen_correct, tuned_correct]


def verify_split(results, split, protocol, selected):
    directory = results / split
    is_test = split == "test"
    prefix = "test_" if is_test else ""
    predictions = read(directory / ("test_predictions.csv" if is_test else "paired_predictions.csv"))
    classes = protocol["class_names"]
    n = 10_000 if is_test else 5_000
    ids = [int(row["image_id"]) for row in predictions]
    truth = [int(row["true_id"]) for row in predictions]
    check(len(ids) == len(set(ids)) == n, f"Invalid {split} image IDs")
    check(Counter(truth) == {i: n // 10 for i in range(10)}, f"Unbalanced {split} labels")
    if is_test:
        check(ids == list(range(n)), "Official test order differs")
    else:
        check(all(0 <= idx < 50_000 for idx in ids), "Validation IDs outside official training set")
        signature = hashlib.sha256(b"".join(struct.pack("<q", i) for i in ids)).hexdigest()
        check(signature == "e205673867bbc0ac46124bb02c6c3500c9fb380ea1a76a41c9bac10ba2ef5745", "Validation order differs")
    summaries = indexed(read(directory / f"{split}_summary.csv"), "model")
    class_rows = indexed(read(directory / f"{prefix}per_class.csv"), "model", "class")
    cm_rows = indexed(read(directory / f"{prefix}confusion_counts.csv"), "model", "true_class", "predicted_class")
    check((len(summaries), len(class_rows), len(cm_rows)) == (3, 30, 300), f"Incomplete {split} report")
    for key in MODELS:
        guesses = [int(r[key]) for r in predictions]
        hits, macro, levels, counts = metrics(truth, guesses, classes)
        row = summaries[key,]
        cp = protocol["checkpoints"][key]
        check(row["split"] == split and int(row["n"]) == n and int(row["correct"]) == hits, f"{split}/{key} counts")
        check(int(row["selected_epoch"]) == cp["selected_epoch"] == int(selected[key]["epoch"]), f"{key} epoch")
        close(cp["validation_loss"], float(selected[key]["val_loss"]), f"{key} selected loss")
        close(cp["validation_accuracy"], float(selected[key]["val_accuracy"]), f"{key} logged accuracy")
        close(row["accuracy"], hits / n, f"{split}/{key} accuracy")
        close(row["macro_f1"], macro, f"{split}/{key} macro-F1")
        if is_test:
            close(row["validation_accuracy"], cp["validation_accuracy"], "Validation reference")
            close(row["test_minus_validation_accuracy_pp"], 100 * (hits / n - cp["validation_accuracy"]), "Accuracy gap")
        else:
            check(int(row["replay_delta_images"]) == 0 and hits == round(cp["validation_accuracy"] * n), "Validation replay")
        for i, name in enumerate(classes):
            for metric, expected in levels[name].items():
                close(class_rows[key, name][metric], expected, f"{split}/{key}/{name} {metric}")
            for j, guess in enumerate(classes):
                check(int(cm_rows[key, name, guess]["count"]) == counts[i, j], f"{split}/{key} confusion count")
        print(f"{split:10s} {key:7s} {hits:5d}/{n}   accuracy {hits / n:.2%}   macro-F1 {macro:.2%}")
    pairs = Counter()
    for row in predictions:
        change = change_type(row["frozen"] == row["true_id"], row["tuned"] == row["true_id"])
        pairs[classes[int(row["true_id"])], change] += 1
        if not is_test:
            check(row["change"] == change, "Validation paired prediction label")
    changes = indexed(read(directory / f"{prefix}paired_changes.csv"), "true_class")
    check(len(changes) == 10, "Incomplete paired class report")
    for name in classes:
        for change in ("corrected", "regressed", "both_correct", "both_wrong"):
            check(int(changes[name,][change]) == pairs[name, change], f"{split}/{name} {change}")
        check(int(changes[name,]["net_gain"]) == pairs[name, "corrected"] - pairs[name, "regressed"], "Paired class gain")
    return predictions


def verify_ablation(results, validation, classes, augmented_epoch):
    folder = results / "augmentation"
    no_aug = indexed(read(folder / "no_aug_validation_predictions.csv"), "image_id")
    check(len(no_aug) == len(validation), "Ablation sample size")
    ids = [row["image_id"] for row in validation]
    truth = [int(row["true_id"]) for row in validation]
    without = []
    for idx, label in zip(ids, truth):
        row = no_aug[idx,]
        check(int(row["true_id"]) == label, f"Ablation label for image {idx}")
        without.append(int(row["cnn_no_aug"]))
    with_aug = [int(row["cnn"]) for row in validation]
    conditions = indexed(read(folder / "augmentation_comparison.csv"), "condition")
    check(len(conditions) == 2, "Ablation conditions")
    no_aug_epoch = selected_epoch(folder / "no_aug_history.csv", 10)
    for name, predictions, checkpoint in [("Crop + flip", with_aug, augmented_epoch),
                                          ("No augmentation", without, no_aug_epoch)]:
        hits, macro, _, _ = metrics(truth, predictions, classes)
        row = conditions[name,]
        check(int(row["correct"]) == hits and int(row["selected_epoch"]) == int(checkpoint["epoch"]), "Ablation selection")
        close(row["val_loss"], float(checkpoint["val_loss"]), "Ablation loss")
        close(checkpoint["val_accuracy"], hits / len(truth), "Ablation checkpoint accuracy")
        close(row["accuracy"], hits / len(truth), "Ablation accuracy")
        close(row["macro_f1"], macro, "Ablation macro-F1")
    augmented_only = sum(a == y and b != y for a, b, y in zip(with_aug, without, truth))
    plain_only = sum(a != y and b == y for a, b, y in zip(with_aug, without, truth))
    print(f"Validation ablation   crop + flip only correct: {augmented_only}   unaugmented only correct: {plain_only}")


def verify(results=ROOT):
    protocol = json.loads((results / "test/protocol.json").read_text(encoding="utf-8"))
    check(protocol["split"] == "test" and protocol["n_images"] == 10_000 and protocol["training_seed"] == 42, "Evaluation protocol")
    check(len(set(protocol["class_names"])) == 10, "Class names")
    digest = hashlib.sha256((results / "test/test_predictions.csv").read_bytes()).hexdigest()
    check(digest == protocol["test_predictions_sha256"], "Test prediction digest differs")
    selected = {model: selected_epoch(results / "training" / name, 3 if model == "tuned" else 10)
                for model, name in HISTORIES.items()}
    check(float(selected["tuned"]["val_loss"]) < float(selected["frozen"]["val_loss"]), "Fine-tuning starting-point selection")
    validation = verify_split(results, "validation", protocol, selected)
    verify_split(results, "test", protocol, selected)
    verify_ablation(results, validation, protocol["class_names"], selected["cnn"])


if __name__ == "__main__":
    verify()
