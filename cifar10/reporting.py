"""Metrics and figures for aligned classification predictions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def paired_summary(predictions, class_names):
    before = predictions["frozen"].eq(predictions["true_id"])
    after = predictions["tuned"].eq(predictions["true_id"])
    groups = pd.DataFrame(
        {
            "class_id": predictions["true_id"],
            "corrected": ~before & after,
            "regressed": before & ~after,
        }
    ).groupby("class_id", sort=True)
    counts = groups[["corrected", "regressed"]].sum()
    counts["n"] = groups.size()
    counts.index = pd.Index([class_names[i] for i in counts.index], name="class")
    counts["net"] = counts["corrected"] - counts["regressed"]
    return counts


def accuracy_interval(counts, repeats=20_000, seed=42):
    # Bootstrap paired correctness differences within class, keeping each class size fixed.
    rng = np.random.default_rng(seed)
    net = np.zeros(repeats, dtype=np.int64)
    for row in counts.itertuples():
        outcomes = np.array([row.corrected, row.regressed, row.n - row.corrected - row.regressed])
        draws = rng.multinomial(row.n, outcomes / row.n, size=repeats)
        net += draws[:, 0] - draws[:, 1]
    return np.percentile(100 * net / counts["n"].sum(), [2.5, 97.5])


def plot_changes(counts, destination):
    ranked = counts.sort_values("net", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.65))
    positions = np.arange(len(ranked))
    ax.barh(positions, -ranked.regressed, height=0.56, color="#a6573f")
    ax.barh(positions, ranked.corrected, height=0.56, color="#176864")
    ax.axvline(0, lw=0.8, color="#637478")
    for y, row in enumerate(ranked.itertuples()):
        ax.text(-row.regressed - 2.5, y, str(row.regressed), va="center", ha="right", size=8.5)
        ax.text(row.corrected + 2.5, y, str(row.corrected), va="center", ha="left", size=8.5)
    left = max(1.0, ranked.regressed.max() * 1.32)
    right = max(1.0, ranked.corrected.max() * 1.20)
    ax.set(
        yticks=positions,
        yticklabels=[label.capitalize() for label in ranked.index],
        xlim=(-left, right),
        ylim=(len(ranked) - 0.4, -0.6),
    )
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=8, labelsize=9)
    ax.spines[:].set_visible(False)
    ax.text(
        -left * 0.5,
        -0.85,
        "Correct → wrong",
        ha="center",
        va="bottom",
        size=9,
        weight="bold",
        color="#954732",
    )
    ax.text(
        right * 0.5,
        -0.85,
        "Wrong → correct",
        ha="center",
        va="bottom",
        size=9,
        weight="bold",
        color="#145d59",
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.025, top=0.91)
    fig.savefig(destination, dpi=210, facecolor="white")
    plt.close(fig)


def write_scores(frame, names, keys, output):
    matrices, scores, per_class = [], [], []
    for key in keys:
        matrix = np.zeros((len(names), len(names)), dtype=int)
        np.add.at(matrix, (frame.true_id.to_numpy(), frame[key].to_numpy()), 1)
        matrices.append(matrix)
        hits = matrix.diagonal()
        support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
        f1 = 2 * hits / np.maximum(support + predicted, 1)
        scores.append(
            {
                "model": key,
                "n": len(frame),
                "correct": int(hits.sum()),
                "accuracy": hits.sum() / len(frame),
                "macro_f1": f1.mean(),
            }
        )
        for i, name in enumerate(names):
            per_class.append(
                {
                    "model": key,
                    "true_class": name,
                    "support": support[i],
                    "precision": hits[i] / max(predicted[i], 1),
                    "recall": hits[i] / max(support[i], 1),
                    "f1": f1[i],
                }
            )
        pd.DataFrame(matrix, index=names, columns=names).to_csv(
            output / f"confusion_{key}.csv", index_label="true_class"
        )
    summary = pd.DataFrame(scores)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "per_class.csv", index=False)
    fig, axes = plt.subplots(
        1, len(keys), figsize=(4.8 * len(keys), 4.6), squeeze=False, constrained_layout=True
    )
    for key, matrix, ax in zip(keys, matrices, axes.flat):
        image = ax.imshow(
            matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1
        )
        ax.set(
            xticks=range(len(names)),
            yticks=range(len(names)),
            xticklabels=names,
            yticklabels=names,
            xlabel="Predicted class",
            ylabel="True class",
            title=key,
        )
        ax.tick_params(axis="x", rotation=75)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Fraction of true class", shrink=0.75)
    fig.savefig(output / "confusions.png", dpi=150)
    plt.close(fig)
    if "tuned" in keys:
        counts = paired_summary(frame, names)
        counts.to_csv(output / "paired_changes.csv")
        plot_changes(counts, output / "paired_changes.png")
    return summary


def plot_history(history, destination):
    frame = pd.DataFrame(history).set_index("epoch")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for metric, ax in zip(("loss", "accuracy"), axes):
        for split in ("train", "val"):
            ax.plot(frame.index, frame[f"{split}_{metric}"], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
