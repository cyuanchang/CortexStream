from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Realtime_processing.decoder_preprocessing import run_decoder_preprocessing
from decoder.config import DecoderTaskType, resolve_decoder_stack_config
from scripts.labels_io import TrialLabel, load_trials
from scripts.recording_io import load_recording


def _trial_window_indices(trial: TrialLabel, fs: int, window_samples: int, stride_samples: int) -> list[tuple[int, int]]:
    start = max(int(round(trial.start_s * fs)), 0)
    stop = max(int(round(trial.end_s * fs)), 0)
    if stop - start < window_samples:
        return []
    indices: list[tuple[int, int]] = []
    for end_idx in range(start + window_samples, stop + 1, max(stride_samples, 1)):
        indices.append((end_idx - window_samples, end_idx))
    return indices


def build_epochs(
    recording_dir: Path,
    trials_file: Path,
    task_type: DecoderTaskType,
    model_path: str,
    manifest_path: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    recording = load_recording(recording_dir)
    trials = load_trials(trials_file)
    config = resolve_decoder_stack_config(
        task_type=task_type,
        deployment_mode="dev",
        model_path=model_path,
        manifest_path=manifest_path,
    )
    window_samples = int(config.eegnet.samples)
    stride_samples = int(config.runtime.stride_samples)
    allowed_labels = {label: idx for idx, label in enumerate(config.eegnet.class_labels)}

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    for trial in trials:
        if trial.task != task_type:
            continue
        label_idx = allowed_labels.get(trial.label)
        if label_idx is None:
            continue
        for start_idx, end_idx in _trial_window_indices(
            trial=trial,
            fs=recording.sample_rate_hz,
            window_samples=window_samples,
            stride_samples=stride_samples,
        ):
            if end_idx > recording.eeg_matrix.shape[1]:
                continue
            raw_window = recording.eeg_matrix[:, start_idx:end_idx]
            prepared = run_decoder_preprocessing(
                window=raw_window,
                sample_rate_hz=recording.sample_rate_hz,
                config=config.preprocess,
            )
            x_list.append(prepared.astype(np.float32))
            y_list.append(label_idx)

    if not x_list:
        return (
            np.zeros((0, recording.eeg_matrix.shape[0], window_samples), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            {
                "task_type": task_type,
                "recording_dir": str(recording_dir),
                "window_samples": window_samples,
                "stride_samples": stride_samples,
                "class_labels": list(config.eegnet.class_labels),
                "count": 0,
            },
        )

    x = np.stack(x_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    meta = {
        "task_type": task_type,
        "recording_dir": str(recording_dir),
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "class_labels": list(config.eegnet.class_labels),
        "count": int(x.shape[0]),
    }
    return x, y, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training epochs from labeled trials")
    parser.add_argument("--recording-dir", required=True, type=str)
    parser.add_argument("--trials-file", required=True, type=str)
    parser.add_argument("--task-type", required=True, choices=("ssvep", "mi"))
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--manifest", default="", type=str)
    args = parser.parse_args()

    x, y, meta = build_epochs(
        recording_dir=Path(args.recording_dir),
        trials_file=Path(args.trials_file),
        task_type=args.task_type,
        model_path=args.model_path,
        manifest_path=args.manifest,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", x)
    np.save(out_dir / "y.npy", y)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"epochs={x.shape[0]} output_dir={out_dir}")


if __name__ == "__main__":
    main()
