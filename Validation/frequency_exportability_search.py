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

from Realtime_processing.decoder_preprocessing import DecoderPreprocessConfig, run_decoder_preprocessing
from scripts.recording_io import load_recording


@dataclass(frozen=True)
class Trial:
    label_hz: float
    start_s: float
    end_s: float


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    fs: int
    eeg: np.ndarray
    trials: list[Trial]


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
            out.append(Trial(label_hz=freq, start_s=float(start_s), end_s=float(rel_s)))
            used.add(trial_idx)
    return out


def _preprocess_config(preprocess_mode: str) -> DecoderPreprocessConfig | None:
    if preprocess_mode == "none":
        return None
    if preprocess_mode == "bp_notch":
        return DecoderPreprocessConfig(
            enabled=True,
            bandpass_low_hz=4.0,
            bandpass_high_hz=40.0,
            apply_notch=True,
            notch_hz=60.0,
            apply_car=False,
            zscore_per_channel=False,
        )
    if preprocess_mode == "bp_notch_car":
        return DecoderPreprocessConfig(
            enabled=True,
            bandpass_low_hz=4.0,
            bandpass_high_hz=40.0,
            apply_notch=True,
            notch_hz=60.0,
            apply_car=True,
            zscore_per_channel=False,
        )
    raise ValueError(f"unknown preprocess mode: {preprocess_mode}")


def load_runs(run_dirs: list[Path], target_freqs: tuple[float, ...]) -> list[RunData]:
    runs: list[RunData] = []
    for run_dir in run_dirs:
        rec = load_recording(run_dir)
        runs.append(
            RunData(
                run_dir=run_dir,
                fs=int(rec.sample_rate_hz),
                eeg=rec.eeg_matrix.astype(np.float32),
                trials=load_trials_from_markers(run_dir / "ssvep_markers.jsonl", accepted_hz=target_freqs),
            )
        )
    return runs


def build_preprocessed_runs(runs: list[RunData], preprocess_mode: str) -> list[np.ndarray]:
    cfg = _preprocess_config(preprocess_mode)
    out: list[np.ndarray] = []
    for run in runs:
        arr = run.eeg.copy()
        if cfg is not None:
            arr = run_decoder_preprocessing(arr, run.fs, cfg).astype(np.float32)
        # Subtract per-channel mean after filtering so FFT scores are DC-free.
        arr = arr - np.mean(arr, axis=1, keepdims=True)
        out.append(arr)
    return out


def extract_trials_matrix(
    runs: list[RunData],
    preprocessed_runs: list[np.ndarray],
    onset_trim_s: float,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    all_x: list[np.ndarray] = []
    all_y: list[float] = []

    for run, eeg in zip(runs, preprocessed_runs):
        fs = run.fs
        n_win = int(round(window_s * fs))
        for tr in run.trials:
            s_idx = int(round((tr.start_s + onset_trim_s) * fs))
            e_idx = s_idx + n_win
            tr_end = int(round(tr.end_s * fs))
            if s_idx < 0 or e_idx > eeg.shape[1] or e_idx > tr_end:
                continue
            epoch = eeg[:, s_idx:e_idx]
            all_x.append(epoch)
            all_y.append(float(tr.label_hz))

    if not all_x:
        return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(all_x, axis=0), np.asarray(all_y, dtype=np.float32)


def _band_power(spec: np.ndarray, freqs: np.ndarray, center_hz: float, half_width_hz: float) -> float:
    mask = (freqs >= center_hz - half_width_hz) & (freqs <= center_hz + half_width_hz)
    if not np.any(mask):
        return 0.0
    return float(np.mean(spec[mask]))


def _sideband_power(spec: np.ndarray, freqs: np.ndarray, center_hz: float, inner_hz: float, outer_hz: float) -> float:
    m1 = (freqs >= center_hz - outer_hz) & (freqs <= center_hz - inner_hz)
    m2 = (freqs >= center_hz + inner_hz) & (freqs <= center_hz + outer_hz)
    mask = m1 | m2
    if not np.any(mask):
        return 0.0
    return float(np.mean(spec[mask]))


def trial_channel_score(
    sig: np.ndarray,
    fs: int,
    candidate_hz: float,
    mode: str,
    bw_hz: float = 0.5,
) -> float:
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(sig.shape[0], d=1.0 / fs)
    p1 = _band_power(spec, freqs, candidate_hz, bw_hz)

    if mode == "fund":
        return p1

    h2 = candidate_hz * 2.0
    p2 = _band_power(spec, freqs, h2, bw_hz) if h2 < (fs / 2.0 - bw_hz) else 0.0
    if mode == "fund_h2":
        return p1 + 0.5 * p2

    # SNR score: fundamental (and optional H2) power divided by local sideband power.
    n1 = _sideband_power(spec, freqs, candidate_hz, inner_hz=1.0, outer_hz=2.0)
    n2 = _sideband_power(spec, freqs, h2, inner_hz=1.0, outer_hz=2.0) if h2 < (fs / 2.0 - 2.0) else 0.0
    s1 = (p1 + 1e-12) / (n1 + 1e-12)
    s2 = (p2 + 1e-12) / (n2 + 1e-12) if p2 > 0 else 0.0
    if mode == "snr_fund":
        return s1
    if mode == "snr_fund_h2":
        return s1 + 0.5 * s2
    raise ValueError(mode)


def predict_labels(
    scores: np.ndarray,  # (n, n_freq, c)
    target_freqs: tuple[float, ...],
    y_true: np.ndarray,
    channel_mode: str,
) -> np.ndarray:
    _, _, c = scores.shape

    if channel_mode == "mean_all":
        combined = np.mean(scores, axis=2)  # (n, n_freq)
    elif channel_mode == "max_channel_per_trial":
        # Per trial, take the max score across channels for each frequency candidate.
        combined = np.max(scores, axis=2)
    elif channel_mode == "best_single_global":
        # Select one fixed channel that maximizes true-label accuracy on this set.
        best_ch = 0
        best_acc = -1.0  # type: float
        for ch in range(c):
            pred_idx = np.argmax(scores[:, :, ch], axis=1)
            pred_hz = np.asarray([target_freqs[i] for i in pred_idx], dtype=np.float32)
            acc = float(np.mean(pred_hz == y_true))
            if acc > best_acc:
                best_acc = acc
                best_ch = ch
        combined = scores[:, :, best_ch]
    else:
        raise ValueError(channel_mode)

    pred_idx = np.argmax(combined, axis=1)
    return np.asarray([target_freqs[i] for i in pred_idx], dtype=np.float32)


def evaluate_config(
    x: np.ndarray,
    y: np.ndarray,
    target_freqs: tuple[float, ...],
    score_mode: str,
    channel_mode: str,
    fs: int = 125,
) -> dict:
    if x.shape[0] == 0:
        return {"n": 0, "acc": 0.0}

    n, c, _ = x.shape
    scores = np.zeros((n, len(target_freqs), c), dtype=np.float64)
    for i in range(n):
        for ch in range(c):
            sig = x[i, ch]
            for j, hz in enumerate(target_freqs):
                scores[i, j, ch] = trial_channel_score(sig, fs=fs, candidate_hz=hz, mode=score_mode)

    pred = predict_labels(scores=scores, target_freqs=target_freqs, y_true=y, channel_mode=channel_mode)
    acc = float(np.mean(pred == y))
    return {"n": int(x.shape[0]), "acc": acc}


def build_scores(
    x: np.ndarray,
    target_freqs: tuple[float, ...],
    score_mode: str,
    fs: int = 125,
) -> np.ndarray:
    n, c, _ = x.shape
    scores = np.zeros((n, len(target_freqs), c), dtype=np.float64)
    for i in range(n):
        for ch in range(c):
            sig = x[i, ch]
            for j, hz in enumerate(target_freqs):
                scores[i, j, ch] = trial_channel_score(sig, fs=fs, candidate_hz=hz, mode=score_mode)
    return scores


def evaluate_scores(
    scores: np.ndarray,
    y: np.ndarray,
    target_freqs: tuple[float, ...],
    channel_mode: str,
) -> dict:
    pred = predict_labels(scores=scores, target_freqs=target_freqs, y_true=y, channel_mode=channel_mode)
    acc = float(np.mean(pred == y))
    return {"n": int(scores.shape[0]), "acc": acc}


def run_search(
    run_dirs: list[Path],
    target_freqs: tuple[float, ...],
    dataset_name: str,
) -> list[dict]:
    preprocess_modes = ["none", "bp_notch", "bp_notch_car"]
    onset_grid = [0.2, 0.8, 1.4, 1.8]
    window_grid = [1.0, 1.5, 2.0]
    score_modes = ["fund_h2", "snr_fund_h2"]
    channel_modes = ["mean_all", "best_single_global"]

    runs = load_runs(run_dirs=run_dirs, target_freqs=target_freqs)
    preprocessed_cache = {pre: build_preprocessed_runs(runs, pre) for pre in preprocess_modes}

    out: list[dict] = []
    for pre in preprocess_modes:
        pre_runs = preprocessed_cache[pre]
        for onset in onset_grid:
            for win in window_grid:
                if onset + win > 3.9:  # Skip windows that exceed the nominal 4 s trial length.
                    continue
                x, y = extract_trials_matrix(runs=runs, preprocessed_runs=pre_runs, onset_trim_s=onset, window_s=win)
                fs = runs[0].fs if runs else 125
                for score in score_modes:
                    scores = build_scores(x=x, target_freqs=target_freqs, score_mode=score, fs=fs)
                    for ch_mode in channel_modes:
                        metrics = evaluate_scores(
                            scores=scores,
                            y=y,
                            target_freqs=target_freqs,
                            channel_mode=ch_mode,
                        )
                        out.append(
                            {
                                "dataset": dataset_name,
                                "preprocess_mode": pre,
                                "onset_trim_s": onset,
                                "window_s": win,
                                "score_mode": score,
                                "channel_mode": ch_mode,
                                "n_trials": metrics["n"],
                                "acc": metrics["acc"],
                            }
                        )
                print(f"[{dataset_name}] pre={pre} onset={onset:.1f} win={win:.1f} done")
    return out


def save_heatmap(rows: list[dict], title: str, out_png: Path, chance: float) -> None:
    # Each cell is the best accuracy over preprocess/score/channel for that onset/window pair.
    onset_vals = sorted({float(r["onset_trim_s"]) for r in rows})
    window_vals = sorted({float(r["window_s"]) for r in rows})
    z = np.full((len(window_vals), len(onset_vals)), np.nan, dtype=np.float64)
    for i, w in enumerate(window_vals):
        for j, o in enumerate(onset_vals):
            subset = [r for r in rows if float(r["window_s"]) == w and float(r["onset_trim_s"]) == o]
            if not subset:
                continue
            z[i, j] = max(float(r["acc"]) for r in subset)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(onset_vals)), labels=[str(v) for v in onset_vals])
    ax.set_yticks(range(len(window_vals)), labels=[str(v) for v in window_vals])
    ax.set_xlabel("Onset trim (s)")
    ax.set_ylabel("Window length (s)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Best accuracy")
    # Write accuracy values into each heatmap cell.
    for i in range(len(window_vals)):
        for j in range(len(onset_vals)):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    # Draw chance-level reference above the axes.
    ax.text(0.01, 1.02, f"chance={chance:.2f}", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    binary_rows = run_search(
        run_dirs=[
            REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "2_fre_1",
            REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "2_freq_2",
        ],
        target_freqs=(7.5, 12.5),
        dataset_name="binary_2freq",
    )
    five_rows = run_search(
        run_dirs=[
            REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "5_fre_1",
            REPO_ROOT / "Data" / "Pilot_1_SSVEP" / "5_fre_2",
        ],
        target_freqs=(7.5, 8.3333, 10.0, 12.5, 15.0),
        dataset_name="fivefreq",
    )

    all_rows = binary_rows + five_rows
    out_dir = Path(__file__).resolve().parent
    (out_dir / "psd_exportability_search.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    top_binary = sorted(binary_rows, key=lambda r: float(r["acc"]), reverse=True)[:15]
    top_five = sorted(five_rows, key=lambda r: float(r["acc"]), reverse=True)[:15]
    summary = {"top_binary": top_binary, "top_five": top_five}
    (out_dir / "psd_exportability_top.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_heatmap(binary_rows, "Binary PSD Exportability (best per onset/window)", out_dir / "fig_psd_exportability_binary.png", chance=0.5)
    save_heatmap(five_rows, "5-Freq PSD Exportability (best per onset/window)", out_dir / "fig_psd_exportability_5freq.png", chance=0.2)

    print(f"wrote {out_dir / 'psd_exportability_search.json'}")
    print(f"wrote {out_dir / 'psd_exportability_top.json'}")
    print(f"wrote {out_dir / 'fig_psd_exportability_binary.png'}")
    print(f"wrote {out_dir / 'fig_psd_exportability_5freq.png'}")
    print("top binary:", [{"acc": round(r["acc"], 3), "pre": r["preprocess_mode"], "onset": r["onset_trim_s"], "win": r["window_s"], "score": r["score_mode"], "ch": r["channel_mode"]} for r in top_binary[:5]])
    print("top 5freq:", [{"acc": round(r["acc"], 3), "pre": r["preprocess_mode"], "onset": r["onset_trim_s"], "win": r["window_s"], "score": r["score_mode"], "ch": r["channel_mode"]} for r in top_five[:5]])


if __name__ == "__main__":
    main()

