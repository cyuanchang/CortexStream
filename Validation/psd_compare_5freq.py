from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

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


def main() -> None:
    run_dirs = [
        REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "5_fre_1",
        REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "5_fre_2",
    ]
    target_freqs = (7.5, 8.3333, 10.0, 12.5, 15.0)
    fs_expected = 125
    onset_trim_s = 0.2
    window_s = 2.0
    band_half_width_hz = 0.5

    analysis_rows: list[dict] = []
    run_metrics: dict[str, dict] = {}

    for run_dir in run_dirs:
        recording = load_recording(run_dir)
        if recording.sample_rate_hz != fs_expected:
            raise RuntimeError(f"Unexpected fs in {run_dir}: {recording.sample_rate_hz}")
        trials = load_trials_from_markers(run_dir / "ssvep_markers.jsonl", accepted_hz=target_freqs)
        per_freq_scores: dict[str, list[float]] = {_fmt_freq(f): [] for f in target_freqs}
        simple_correct = 0
        valid_trials = 0

        for trial in trials:
            epoch = pick_window(
                signal=recording.eeg_matrix,
                fs=recording.sample_rate_hz,
                start_s=trial.start_s,
                end_s=trial.end_s,
                onset_trim_s=onset_trim_s,
                window_s=window_s,
            )
            if epoch is None:
                continue
            avg = np.mean(epoch, axis=0)
            powers = {f: band_power(avg, fs=recording.sample_rate_hz, target_hz=f, band_half_width_hz=band_half_width_hz) for f in target_freqs}
            true_f = float(trial.label_hz)
            true_power = powers[true_f]
            other_mean = float(np.mean([v for k, v in powers.items() if k != true_f]))
            dominance = (true_power + 1e-12) / (other_mean + 1e-12)
            per_freq_scores[_fmt_freq(true_f)].append(dominance)
            pred_f = max(powers.keys(), key=lambda f: powers[f])
            simple_correct += int(pred_f == true_f)
            valid_trials += 1

            row = {
                "run": run_dir.name,
                "label_hz": _fmt_freq(true_f),
                "pred_hz_simple_psd": _fmt_freq(pred_f),
                "dominance_true_over_other_mean": dominance,
            }
            for f in target_freqs:
                row[f"p_{_fmt_freq(f)}"] = powers[f]
            analysis_rows.append(row)

        run_summary = {
            "n_trials_used": valid_trials,
            "simple_psd_id_accuracy": (simple_correct / valid_trials) if valid_trials else 0.0,
            "dominance_mean_by_label": {
                label: (float(np.mean(vals)) if vals else 0.0)
                for label, vals in per_freq_scores.items()
            },
            "dominance_std_by_label": {
                label: (float(np.std(vals)) if vals else 0.0)
                for label, vals in per_freq_scores.items()
            },
        }
        run_metrics[run_dir.name] = run_summary

    out_dir = Path(__file__).resolve().parent
    report = {
        "config": {
            "runs": [str(p) for p in run_dirs],
            "target_freqs_hz": list(target_freqs),
            "sample_rate_hz": fs_expected,
            "onset_trim_s": onset_trim_s,
            "window_s": window_s,
            "band_half_width_hz": band_half_width_hz,
        },
        "run_metrics": run_metrics,
    }
    report_path = out_dir / "psd_compare_5freq_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if analysis_rows:
        import csv

        csv_path = out_dir / "psd_compare_5freq_trials.csv"
        fieldnames = list(analysis_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analysis_rows)
        print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    for run_name, stats in run_metrics.items():
        print(
            f"run={run_name} n={stats['n_trials_used']} "
            f"simple_psd_id_acc={stats['simple_psd_id_accuracy']:.3f}"
        )
        by_label = stats["dominance_mean_by_label"]
        print("  dominance_means:", {k: round(v, 3) for k, v in by_label.items()})


if __name__ == "__main__":
    main()

