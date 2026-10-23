"""Persist token metrics, descriptive statistics and distribution plots."""

import csv
import json
import math
from pathlib import Path
import statistics


METRICS = ("kl", "cross_entropy", "teacher_entropy", "student_entropy",
           "teacher_nll", "student_nll", "top1_agreement", "teacher_accuracy", "student_accuracy")


def describe(values):
    values = sorted(values)
    if not values:
        raise ValueError("Нет значений для статистики")
    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"count": len(values), "mean": statistics.mean(values),
            "std": statistics.pstdev(values), "min": values[0], "max": values[-1],
            "median": statistics.median(values), "p05": percentile(.05), "p95": percentile(.95)}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_report(report, output, plots=True):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = output.parent / (output.stem + "_artifacts")
    folder.mkdir(parents=True, exist_ok=True)
    summary_rows, example_rows, token_rows = [], [], []
    for model in report["models"]:
        tokens = [t for example in model["per_example"] for t in example["per_token"]]
        model["token_statistics"] = {
            metric: describe([t[metric] for t in tokens]) for metric in METRICS
        }
        summary_rows.append({"model": model["name"], "path": model["path"],
                             "parameters": model["parameters"], **model["summary"]})
        for example in model["per_example"]:
            example_rows.append({"model": model["name"], **{k: v for k, v in example.items() if k != "per_token"}})
            token_rows.extend({"model": model["name"], "example_index": example["index"], **t}
                              for t in example["per_token"])
    write_csv(folder / "summary.csv", summary_rows)
    write_csv(folder / "examples.csv", example_rows)
    write_csv(folder / "tokens.csv", token_rows)
    report["artifacts_directory"] = str(folder)
    # Save the numerical results first, even if plotting fails.
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if plots:
        plot_report(report, folder)
    return folder


def plot_report(report, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric in zip(axes.flat, METRICS[:6]):
        series = [[t[metric] for example in m["per_example"] for t in example["per_token"]]
                  for m in report["models"]]
        # One set of bin boundaries for all models.
        low, high = min(min(s) for s in series), max(max(s) for s in series)
        if low == high:
            low, high = low - .5, high + .5
        bins = [low + (high - low) * i / 40 for i in range(41)]
        for model, values in zip(report["models"], series):
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.8, label=model["name"])
        ax.set(title=metric, xlabel="Nats per answer token", ylabel="Density")
        ax.legend()
    fig.suptitle("Distribution of token metrics on the test set")
    fig.tight_layout()
    fig.savefig(folder / "token_metric_histograms.png", dpi=160)
    plt.close(fig)

    for model_number, model in enumerate(report["models"]):
        snapshots = model["distribution_snapshots"]
        if not snapshots:
            continue
        fig, axes = plt.subplots(len(snapshots), 1, figsize=(16, 4 * len(snapshots)), squeeze=False)
        for ax, snap in zip(axes.flat, snapshots):
            rows = snap["tokens"]
            labels = [f"{t['id']}: {t['text']!r}" for t in rows] + ["Other vocabulary"]
            teacher = [t["teacher_probability"] for t in rows] + [snap["teacher_other_mass"]]
            student = [t["student_probability"] for t in rows] + [snap["student_other_mass"]]
            x = list(range(len(labels)))
            ax.bar([i - .2 for i in x], teacher, width=.4, label="Teacher")
            ax.bar([i + .2 for i in x], student, width=.4, label=model["name"])
            ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
            ax.set(ylabel="Probability", title=f"Example {snap['example_index']}, position {snap['position']}, gold {snap['target_text']!r}")
            ax.legend()
        fig.tight_layout()
        fig.savefig(folder / f"model_{model_number}_token_probabilities.png", dpi=160)
        plt.close(fig)
