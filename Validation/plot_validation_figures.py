from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_binary_scatter(rows: list[dict], out: Path) -> None:
    x_75 = [float(r["p7_5"]) for r in rows if float(r["label_hz"]) == 7.5]
    y_75 = [float(r["p12_5"]) for r in rows if float(r["label_hz"]) == 7.5]
    x_125 = [float(r["p7_5"]) for r in rows if float(r["label_hz"]) == 12.5]
    y_125 = [float(r["p12_5"]) for r in rows if float(r["label_hz"]) == 12.5]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_75, y_75, alpha=0.7, label="label 7.5", s=28)
    ax.scatter(x_125, y_125, alpha=0.7, label="label 12.5", s=28)
    lim = max(max(x_75 + x_125), max(y_75 + y_125)) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("PSD power at 7.5 Hz")
    ax.set_ylabel("PSD power at 12.5 Hz")
    ax.set_title("Binary PSD Scatter (2_fre_1 + 2_freq_2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_binary_confusion(rows: list[dict], out: Path) -> None:
    labels = ["7.5", "12.5"]
    idx = {k: i for i, k in enumerate(labels)}
    cm = np.zeros((2, 2), dtype=int)
    for r in rows:
        true_label = "12.5" if float(r["label_hz"]) > 10 else "7.5"
        pred_label = "12.5" if float(r["p12_5"]) > float(r["p7_5"]) else "7.5"
        cm[idx[true_label], idx[pred_label]] += 1

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(2), labels=labels)
    ax.set_yticks(range(2), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Binary Simple PSD Confusion")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_per_channel(report: dict, out: Path) -> None:
    summary = report["per_channel_summary"]
    ch = sorted(int(k) for k in summary.keys())
    acc = [float(summary[str(i)]["simple_rule_accuracy"]) for i in ch]
    gap = [float(summary[str(i)]["ratio_gap_7p5_minus_12p5"]) for i in ch]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(ch, acc)
    axes[0].axhline(0.5, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Simple rule accuracy")
    axes[0].set_title("Per-Channel Binary PSD Diagnostics")

    axes[1].bar(ch, gap)
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Ratio gap (7.5 - 12.5)")
    axes[1].set_xlabel("Channel index")

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_5freq_run_bars(report: dict, out: Path) -> None:
    target_freqs = [str(f).rstrip("0").rstrip(".") for f in report["config"]["target_freqs_hz"]]
    run_metrics = report["run_metrics"]
    runs = list(run_metrics.keys())
    vals = [[run_metrics[r]["dominance_mean_by_label"][f] for f in target_freqs] for r in runs]

    x = np.arange(len(target_freqs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - width / 2, vals[0], width, label=runs[0])
    ax.bar(x + width / 2, vals[1], width, label=runs[1])
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(x, labels=target_freqs)
    ax.set_xlabel("True target frequency (Hz)")
    ax.set_ylabel("Dominance: P(true) / mean P(other)")
    ax.set_title("5-Frequency Run Comparison (Dominance by Label)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_5freq_confusions(trial_rows: list[dict], out: Path) -> None:
    labels = ["7.5", "8.3333", "10", "12.5", "15"]
    idx = {k: i for i, k in enumerate(labels)}
    run_to_cm: dict[str, np.ndarray] = {
        "5_fre_1": np.zeros((5, 5), dtype=int),
        "5_fre_2": np.zeros((5, 5), dtype=int),
    }
    for r in trial_rows:
        run = r["run"]
        if run not in run_to_cm:
            continue
        t = r["label_hz"]
        p = r["pred_hz_simple_psd"]
        if t in idx and p in idx:
            run_to_cm[run][idx[t], idx[p]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, run in zip(axes, ["5_fre_1", "5_fre_2"]):
        cm = run_to_cm[run]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(run)
        ax.set_xticks(range(5), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(5), labels=labels)
        ax.set_xlabel("Predicted")
        if run == "5_fre_1":
            ax.set_ylabel("True")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("5-Frequency Simple PSD Confusion")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    rows_binary = _read_csv(BASE / "psd_trial_level.csv")
    report_ch = json.loads((BASE / "psd_per_channel_report.json").read_text(encoding="utf-8"))
    report_5 = json.loads((BASE / "psd_compare_5freq_report.json").read_text(encoding="utf-8"))
    rows_5 = _read_csv(BASE / "psd_compare_5freq_trials.csv")

    plot_binary_scatter(rows_binary, BASE / "fig_binary_scatter.png")
    plot_binary_confusion(rows_binary, BASE / "fig_binary_confusion.png")
    plot_per_channel(report_ch, BASE / "fig_per_channel_summary.png")
    plot_5freq_run_bars(report_5, BASE / "fig_5freq_run_dominance.png")
    plot_5freq_confusions(rows_5, BASE / "fig_5freq_confusion.png")

    print("wrote figures:")
    for name in [
        "fig_binary_scatter.png",
        "fig_binary_confusion.png",
        "fig_per_channel_summary.png",
        "fig_5freq_run_dominance.png",
        "fig_5freq_confusion.png",
    ]:
        print((BASE / name).as_posix())


if __name__ == "__main__":
    main()

