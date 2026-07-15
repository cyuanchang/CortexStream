from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.recording_io import load_recording


@dataclass(frozen=True)
class Trial:
    run_name: str
    label_hz: float
    start_s: float
    end_s: float


def _fmt_freq(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def load_trials_from_markers(markers_path: Path, accepted_hz: tuple[float, ...]) -> list[Trial]:
    events = [
        json.loads(line)
        for line in markers_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starts: dict[int, tuple[float, float]] = {}
    used: set[int] = set()
    out: list[Trial] = []
    accepted = set(float(v) for v in accepted_hz)

    for event in events:
        name = str(event.get("event", "")).strip()
        trial_idx = int(event.get("trial_index", -1))
        if trial_idx < 0:
            continue
        mono = float(event.get("monotonic_s", 0.0))
        stream_start = float(event.get("stream_started_monotonic", 0.0))
        freq = float(event.get("frequency_hz", math.nan))
        if not np.isfinite(freq) or freq not in accepted:
            continue
        if stream_start <= 0.0 or mono <= 0.0:
            continue
        rel_s = mono - stream_start
        if rel_s <= 0.0:
            continue

        if name == "trial_start":
            starts[trial_idx] = (rel_s, freq)
        elif name == "trial_end":
            if trial_idx in used or trial_idx not in starts:
                continue
            start_s, start_hz = starts[trial_idx]
            if start_hz != freq:
                continue
            if rel_s <= start_s:
                continue
            out.append(
                Trial(
                    run_name=markers_path.parent.name,
                    label_hz=freq,
                    start_s=float(start_s),
                    end_s=float(rel_s),
                )
            )
            used.add(trial_idx)
    return out


def pick_window(
    signal: np.ndarray,
    fs: int,
    start_s: float,
    end_s: float,
    onset_trim_s: float,
    window_s: float,
) -> np.ndarray | None:
    start_idx = int(round((start_s + onset_trim_s) * fs))
    end_idx = start_idx + int(round(window_s * fs))
    trial_end_idx = int(round(end_s * fs))
    if start_idx < 0 or end_idx > signal.shape[-1] or end_idx > trial_end_idx:
        return None
    return signal[:, start_idx:end_idx]


def band_power(avg_epoch: np.ndarray, fs: int, target_hz: float, band_half_width_hz: float) -> float:
    spec = np.abs(np.fft.rfft(avg_epoch)) ** 2
    freqs = np.fft.rfftfreq(avg_epoch.shape[0], d=1.0 / fs)
    mask = (freqs >= (target_hz - band_half_width_hz)) & (freqs <= (target_hz + band_half_width_hz))
    if not np.any(mask):
        return 0.0
    return float(np.mean(spec[mask]))


def run_binary_sweep(
    run_dirs: list[Path],
    onset_grid_s: list[float],
    window_s: float,
    fs_expected: int = 125,
    band_half_width_hz: float = 0.5,
) -> dict:
    target_freqs = (7.5, 12.5)
    per_onset = {}

    for onset_trim_s in onset_grid_s:
        rows = []
        for run_dir in run_dirs:
            recording = load_recording(run_dir)
            if recording.sample_rate_hz != fs_expected:
                raise RuntimeError(f"Unexpected fs in {run_dir}: {recording.sample_rate_hz}")
            trials = load_trials_from_markers(run_dir / "ssvep_markers.jsonl", accepted_hz=target_freqs)
            for trial in trials:
                epoch = pick_window(recording.eeg_matrix, recording.sample_rate_hz, trial.start_s, trial.end_s, onset_trim_s, window_s)
                if epoch is None:
                    continue
                avg = np.mean(epoch, axis=0)
                p75 = band_power(avg, recording.sample_rate_hz, 7.5, band_half_width_hz)
                p125 = band_power(avg, recording.sample_rate_hz, 12.5, band_half_width_hz)
                pred = 12.5 if p125 > p75 else 7.5
                rows.append(
                    {
                        "label": trial.label_hz,
                        "pred": pred,
                        "ratio_75_125": (p75 + 1e-12) / (p125 + 1e-12),
                    }
                )
        if not rows:
            per_onset[str(onset_trim_s)] = {"n": 0, "acc": 0.0, "mean_ratio_7p5": 0.0, "mean_ratio_12p5": 0.0}
            continue
        n = len(rows)
        acc = float(np.mean([int(r["label"] == r["pred"]) for r in rows]))
        mean_ratio_7p5 = float(np.mean([r["ratio_75_125"] for r in rows if r["label"] == 7.5]))
        mean_ratio_12p5 = float(np.mean([r["ratio_75_125"] for r in rows if r["label"] == 12.5]))
        per_onset[str(onset_trim_s)] = {
            "n": n,
            "acc": acc,
            "mean_ratio_7p5": mean_ratio_7p5,
            "mean_ratio_12p5": mean_ratio_12p5,
        }
    return per_onset


def run_5freq_sweep(
    run_dirs: list[Path],
    onset_grid_s: list[float],
    window_s: float,
    fs_expected: int = 125,
    band_half_width_hz: float = 0.5,
) -> dict:
    target_freqs = (7.5, 8.3333, 10.0, 12.5, 15.0)
    per_onset = {}

    for onset_trim_s in onset_grid_s:
        rows = []
        for run_dir in run_dirs:
            recording = load_recording(run_dir)
            if recording.sample_rate_hz != fs_expected:
                raise RuntimeError(f"Unexpected fs in {run_dir}: {recording.sample_rate_hz}")
            trials = load_trials_from_markers(run_dir / "ssvep_markers.jsonl", accepted_hz=target_freqs)
            for trial in trials:
                epoch = pick_window(recording.eeg_matrix, recording.sample_rate_hz, trial.start_s, trial.end_s, onset_trim_s, window_s)
                if epoch is None:
                    continue
                avg = np.mean(epoch, axis=0)
                powers = {f: band_power(avg, recording.sample_rate_hz, f, band_half_width_hz) for f in target_freqs}
                pred = max(powers.keys(), key=lambda f: powers[f])
                rows.append({"label": trial.label_hz, "pred": pred})
        n = len(rows)
        acc = float(np.mean([int(r["label"] == r["pred"]) for r in rows])) if n else 0.0
        per_onset[str(onset_trim_s)] = {"n": n, "acc": acc}
    return per_onset


def plot_sweep(binary: dict, fivefreq: dict, out: Path) -> None:
    x = sorted(float(k) for k in binary.keys())
    b_acc = [binary[str(v)]["acc"] for v in x]
    f_acc = [fivefreq[str(v)]["acc"] for v in x]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(x, b_acc, marker="o", label="2-freq simple PSD acc")
    axes[0].axhline(0.5, linestyle="--", linewidth=1, label="binary chance")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_title("Window-Position Sweep (onset trim, 2.0s window)")

    axes[1].plot(x, f_acc, marker="o", label="5-freq simple PSD acc")
    axes[1].axhline(0.2, linestyle="--", linewidth=1, label="5-class chance")
    axes[1].set_xlabel("Onset trim (s)")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    onset_grid_s = [0.2, 0.6, 1.0, 1.4, 1.8]
    window_s = 2.0

    binary_runs = [
        REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "2_fre_1",
        REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "2_freq_2",
    ]
    fivefreq_runs = [
        REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "5_fre_1",
        REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "5_fre_2",
    ]

    binary = run_binary_sweep(binary_runs, onset_grid_s, window_s)
    fivefreq = run_5freq_sweep(fivefreq_runs, onset_grid_s, window_s)

    out_dir = Path(__file__).resolve().parent
    report = {
        "config": {
            "onset_grid_s": onset_grid_s,
            "window_s": window_s,
            "binary_runs": [str(p) for p in binary_runs],
            "fivefreq_runs": [str(p) for p in fivefreq_runs],
        },
        "binary_sweep": binary,
        "fivefreq_sweep": fivefreq,
    }
    report_path = out_dir / "psd_window_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig_path = out_dir / "fig_psd_window_sweep.png"
    plot_sweep(binary, fivefreq, fig_path)

    print(f"wrote {report_path}")
    print(f"wrote {fig_path}")
    print("binary sweep:", {k: round(v["acc"], 3) for k, v in binary.items()})
    print("5freq sweep:", {k: round(v["acc"], 3) for k, v in fivefreq.items()})


if __name__ == "__main__":
    main()

