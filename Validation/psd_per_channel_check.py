from __future__ import annotations

import csv
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


def band_power_1d(channel_epoch: np.ndarray, fs: int, target_hz: float, band_half_width_hz: float) -> float:
    spec = np.abs(np.fft.rfft(channel_epoch)) ** 2
    freqs = np.fft.rfftfreq(channel_epoch.shape[0], d=1.0 / fs)
    mask = (freqs >= (target_hz - band_half_width_hz)) & (freqs <= (target_hz + band_half_width_hz))
    if not np.any(mask):
        return 0.0
    return float(np.mean(spec[mask]))


def main() -> None:
    run_dirs = [
        REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "2_fre_1",
        REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "2_freq_2",
    ]
    accepted_hz = (7.5, 12.5)
    fs_expected = 125
    onset_trim_s = 0.2
    window_s = 2.0
    band_half_width_hz = 0.5

    rows: list[dict] = []
    for run_dir in run_dirs:
        recording = load_recording(run_dir)
        if recording.sample_rate_hz != fs_expected:
            raise RuntimeError(f"Unexpected fs in {run_dir}: {recording.sample_rate_hz}")
        trials = load_trials_from_markers(run_dir / "ssvep_markers.jsonl", accepted_hz=accepted_hz)

        for trial in trials:
            epoch = pick_window(
                recording.eeg_matrix,
                fs=recording.sample_rate_hz,
                start_s=trial.start_s,
                end_s=trial.end_s,
                onset_trim_s=onset_trim_s,
                window_s=window_s,
            )
            if epoch is None:
                continue

            for ch_idx in range(epoch.shape[0]):
                ch = epoch[ch_idx]
                p75 = band_power_1d(ch, fs=recording.sample_rate_hz, target_hz=7.5, band_half_width_hz=band_half_width_hz)
                p125 = band_power_1d(ch, fs=recording.sample_rate_hz, target_hz=12.5, band_half_width_hz=band_half_width_hz)
                pred = 12.5 if p125 > p75 else 7.5
                rows.append(
                    {
                        "run": trial.run_name,
                        "label_hz": trial.label_hz,
                        "channel_index": ch_idx,
                        "p7_5": p75,
                        "p12_5": p125,
                        "ratio_7p5_over_12p5": (p75 + 1e-12) / (p125 + 1e-12),
                        "simple_pred_hz": pred,
                        "correct_simple_pred": int(pred == trial.label_hz),
                    }
                )

    if not rows:
        raise RuntimeError("No usable per-channel PSD rows.")

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "psd_per_channel.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "label_hz",
                "channel_index",
                "p7_5",
                "p12_5",
                "ratio_7p5_over_12p5",
                "simple_pred_hz",
                "correct_simple_pred",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float]] = {}
    for ch_idx in sorted(set(int(r["channel_index"]) for r in rows)):
        ch_rows = [r for r in rows if int(r["channel_index"]) == ch_idx]
        n = len(ch_rows)
        acc = float(np.mean([r["correct_simple_pred"] for r in ch_rows])) if n else 0.0
        mean_ratio_75_label = float(
            np.mean([r["ratio_7p5_over_12p5"] for r in ch_rows if float(r["label_hz"]) == 7.5])
        )
        mean_ratio_125_label = float(
            np.mean([r["ratio_7p5_over_12p5"] for r in ch_rows if float(r["label_hz"]) == 12.5])
        )
        summary[str(ch_idx)] = {
            "samples": float(n),
            "simple_rule_accuracy": acc,
            "mean_ratio_7p5_label": mean_ratio_75_label,
            "mean_ratio_12p5_label": mean_ratio_125_label,
            "ratio_gap_7p5_minus_12p5": mean_ratio_75_label - mean_ratio_125_label,
        }

    sorted_channels = sorted(summary.items(), key=lambda kv: kv[1]["simple_rule_accuracy"], reverse=True)
    top_channels = [
        {"channel_index": int(ch), **metrics}
        for ch, metrics in sorted_channels[:8]
    ]

    report = {
        "config": {
            "runs": [str(p) for p in run_dirs],
            "accepted_hz": list(accepted_hz),
            "sample_rate_hz": fs_expected,
            "onset_trim_s": onset_trim_s,
            "window_s": window_s,
            "band_half_width_hz": band_half_width_hz,
        },
        "per_channel_summary": summary,
        "top_channels_by_simple_rule_accuracy": top_channels,
        "interpretation": (
            "Useful channels should have simple_rule_accuracy > 0.5 and "
            "mean_ratio_7p5_label noticeably different from mean_ratio_12p5_label."
        ),
    }
    report_path = out_dir / "psd_per_channel_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    for entry in top_channels:
        print(
            f"ch={entry['channel_index']} "
            f"acc={entry['simple_rule_accuracy']:.3f} "
            f"ratio7.5={entry['mean_ratio_7p5_label']:.3f} "
            f"ratio12.5={entry['mean_ratio_12p5_label']:.3f} "
            f"gap={entry['ratio_gap_7p5_minus_12p5']:.3f}"
        )


if __name__ == "__main__":
    main()

