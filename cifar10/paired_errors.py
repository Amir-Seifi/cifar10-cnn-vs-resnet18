"""Summarize paired test predictions and plot corrections by class."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .reporting import accuracy_interval, paired_summary, plot_changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=Path("results/test/test_predictions.csv")
    )
    parser.add_argument("--protocol", type=Path, default=Path("results/test/protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/paired_changes.png"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    names = protocol.get("class_names") or protocol["classes"]
    frame = pd.read_csv(args.predictions)
    if not frame.image_id.is_unique:
        raise ValueError("Each image ID must occur once")
    counts = paired_summary(frame, names)
    low, high = accuracy_interval(counts)
    print(counts.to_string())
    print(f"Accuracy gain: {100 * counts.net.sum() / counts.n.sum():.2f} percentage points")
    print(f"95% class-stratified paired bootstrap interval: [{low:.2f}, {high:.2f}]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_changes(counts, args.output)


if __name__ == "__main__":
    main()
