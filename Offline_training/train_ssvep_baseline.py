from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Realtime_processing.decoder_preprocessing import DecoderPreprocessConfig, run_decoder_preprocessing
from decoder.backends.eegmodels_reference import build_reference_eegnet
from decoder.model_manifest import MANIFEST_SCHEMA_VERSION
from scripts.recording_io import load_recording


@dataclass(frozen=True)
class Trial:
    run_id: str
    start_s: float
    end_s: float
    label: str


@dataclass(frozen=True)
class TrialSample:
    run_id: str
    label: str
    trial_start_s: float
    trial_end_s: float
    window_start_idx: int
    window_end_idx: int
    data: np.ndarray


def _load_marker_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _fmt_label(freq_hz: float) -> str:
    text = f"{float(freq_hz):.4f}".rstrip("0").rstrip(".")
    return text


def build_trials_from_markers(
    markers_path: Path,
    output_trials_path: Path,
    accepted_labels: set[str],
) -> list[Trial]:
    events = _load_marker_events(markers_path)
    if not events:
        raise RuntimeError(f"no marker events in {markers_path}")

    starts_by_idx: dict[int, tuple[float, str]] = {}
    done_idx: set[int] = set()
    trials: list[Trial] = []

    for event in events:
        name = str(event.get("event", "")).strip()
        stream_start = float(event.get("stream_started_monotonic", 0.0))
        mono = float(event.get("monotonic_s", 0.0))
        if stream_start <= 0.0 or mono <= 0.0:
            continue
        rel_s = mono - stream_start
        trial_index = int(event.get("trial_index", -1))
        run_id = str(event.get("session_id", "")).strip()
        if trial_index < 0 or not run_id:
            continue
        freq = event.get("frequency_hz", None)
        if freq is None:
            continue
        label = _fmt_label(float(freq))
        if label not in accepted_labels:
            continue

        if name == "trial_start":
            starts_by_idx[trial_index] = (rel_s, label)
        elif name == "trial_end":
            start_pack = starts_by_idx.get(trial_index)
            if start_pack is None or trial_index in done_idx:
                continue
            start_s, start_label = start_pack
            if start_label != label:
                continue
            if rel_s <= start_s:
                continue
            trials.append(
                Trial(
                    run_id=run_id,
                    start_s=float(start_s),
                    end_s=float(rel_s),
                    label=label,
                )
            )
            done_idx.add(trial_index)

    if not trials:
        raise RuntimeError(f"no valid trials reconstructed from {markers_path}")

    output_trials_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in trials:
        rows.append(
            {
                "start_s": trial.start_s,
                "end_s": trial.end_s,
                "label": trial.label,
                "task": "ssvep",
            }
        )
    output_trials_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return trials


def _extract_trial_samples(
    recording_dir: Path,
    trials: Iterable[Trial],
    preprocess_config: DecoderPreprocessConfig,
    window_seconds: float,
    onset_trim_seconds: float,
) -> list[TrialSample]:
    recording = load_recording(recording_dir)
    fs = recording.sample_rate_hz
    # Apply preprocessing once on the continuous recording to avoid per-epoch
    # IIR edge artifacts from repeatedly filtering short windows.
    continuous_pre = run_decoder_preprocessing(recording.eeg_matrix, fs, preprocess_config).astype(np.float32)
    window_samples = max(int(round(window_seconds * fs)), 1)
    onset_trim_samples = max(int(round(onset_trim_seconds * fs)), 0)
    out: list[TrialSample] = []

    for trial in trials:
        start_idx = max(int(round(trial.start_s * fs)) + onset_trim_samples, 0)
        end_idx = start_idx + window_samples
        if end_idx > continuous_pre.shape[1]:
            continue
        if end_idx > int(round(trial.end_s * fs)):
            continue
        pre = continuous_pre[:, start_idx:end_idx]
        out.append(
            TrialSample(
                run_id=trial.run_id,
                label=trial.label,
                trial_start_s=trial.start_s,
                trial_end_s=trial.end_s,
                window_start_idx=start_idx,
                window_end_idx=end_idx,
                data=pre,
            )
        )
    return out


def _stratified_split(samples: list[TrialSample], train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    by_key: dict[tuple[str, str], list[int]] = {}
    for idx, sample in enumerate(samples):
        key = (sample.run_id, sample.label)
        by_key.setdefault(key, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for key, indices in sorted(by_key.items()):
        _ = key
        local = indices[:]
        rng.shuffle(local)
        cut = int(round(len(local) * train_ratio))
        cut = max(min(cut, len(local)), 0)
        train_idx.extend(local[:cut])
        test_idx.extend(local[cut:])
    return sorted(train_idx), sorted(test_idx)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _prepare_tensor(x: np.ndarray) -> np.ndarray:
    # EEGModels EEGNet uses (batch, chans, samples, 1).
    return x[..., np.newaxis].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal SSVEP EEGNet baseline and package manifest")
    parser.add_argument("--run-dir", action="append", required=True, help="Path to run folder containing raw_frames.csv and ssvep_markers.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory for model package")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--onset-trim-seconds", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--f1", type=int, default=8, help="EEGNet temporal filters")
    parser.add_argument("--d", type=int, default=2, help="EEGNet spatial depth multiplier")
    parser.add_argument("--kern-length", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--apply-car", dest="apply_car", action="store_true", help="Enable CAR preprocessing")
    parser.add_argument("--no-apply-car", dest="apply_car", action="store_false", help="Disable CAR preprocessing")
    parser.set_defaults(apply_car=True)
    parser.add_argument("--zscore-per-channel", dest="zscore_per_channel", action="store_true", help="Enable per-channel zscore")
    parser.add_argument("--no-zscore-per-channel", dest="zscore_per_channel", action="store_false", help="Disable per-channel zscore")
    parser.set_defaults(zscore_per_channel=True)
    args = parser.parse_args()

    if args.train_ratio <= 0.0 or args.train_ratio >= 1.0:
        raise SystemExit("train-ratio must be between 0 and 1")
    run_dirs = [Path(p) for p in args.run_dir]
    labels = ("7.5", "12.5")
    label_to_idx = {label: i for i, label in enumerate(labels)}

    preprocess_config = DecoderPreprocessConfig(
        enabled=True,
        bandpass_low_hz=4.0,
        bandpass_high_hz=40.0,
        apply_notch=True,
        notch_hz=60.0,
        apply_car=bool(args.apply_car),
        zscore_per_channel=bool(args.zscore_per_channel),
    )

    trials_root = Path(args.output_dir) / "intermediate" / "trials"
    all_samples: list[TrialSample] = []
    for run_dir in run_dirs:
        markers = run_dir / "ssvep_markers.jsonl"
        if not markers.exists():
            raise FileNotFoundError(f"missing markers file: {markers}")
        trials_path = trials_root / f"{run_dir.name}_trials.jsonl"
        trials = build_trials_from_markers(markers, trials_path, accepted_labels=set(labels))
        samples = _extract_trial_samples(
            recording_dir=run_dir,
            trials=trials,
            preprocess_config=preprocess_config,
            window_seconds=float(args.window_seconds),
            onset_trim_seconds=float(args.onset_trim_seconds),
        )
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("no usable samples after trial extraction")

    train_idx, test_idx = _stratified_split(all_samples, train_ratio=float(args.train_ratio), seed=int(args.split_seed))
    if not train_idx or not test_idx:
        raise RuntimeError("split produced empty train or test set")

    x = np.stack([s.data for s in all_samples], axis=0)
    y = np.asarray([label_to_idx[s.label] for s in all_samples], dtype=np.int64)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_train_tf = _prepare_tensor(x_train)
    x_test_tf = _prepare_tensor(x_test)
    y_train_oh = _one_hot(y_train, len(labels))
    y_test_oh = _one_hot(y_test, len(labels))

    model = build_reference_eegnet(
        nb_classes=len(labels),
        chans=x.shape[1],
        samples=x.shape[2],
        dropoutRate=float(args.dropout_rate),
        kernLength=int(args.kern_length),
        F1=int(args.f1),
        D=int(args.d),
        F2=int(args.f1) * int(args.d),
        dropoutType="Dropout",
    )

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [EarlyStopping(monitor="val_accuracy", patience=max(int(args.patience), 1), restore_best_weights=True)]
    history = model.fit(
        x_train_tf,
        y_train_oh,
        validation_data=(x_test_tf, y_test_oh),
        batch_size=max(int(args.batch_size), 1),
        epochs=max(int(args.epochs), 1),
        callbacks=callbacks,
        verbose=2,
    )
    eval_loss, eval_acc = model.evaluate(x_test_tf, y_test_oh, verbose=0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    model.save(model_path)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task_type": "ssvep",
        "model_path": "model.keras",
        "class_labels": list(labels),
        "samples": int(x.shape[2]),
        "stride_samples": 16,
        "expected_channels": int(x.shape[1]),
        "input_layout": "channels_last",
        "deployment_mode": "live",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "offline_ssvep_baseline_v1",
        "preprocess": asdict(preprocess_config),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    split_manifest = {
        "train_indices": train_idx,
        "test_indices": test_idx,
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "split_seed": int(args.split_seed),
        "train_ratio": float(args.train_ratio),
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    train_report = {
        "labels": list(labels),
        "window_seconds": float(args.window_seconds),
        "onset_trim_seconds": float(args.onset_trim_seconds),
        "f1": int(args.f1),
        "d": int(args.d),
        "f2": int(args.f1) * int(args.d),
        "kern_length": int(args.kern_length),
        "dropout_rate": float(args.dropout_rate),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "preprocess": asdict(preprocess_config),
        "test_loss": float(eval_loss),
        "test_accuracy": float(eval_acc),
        "history": {k: [float(vv) for vv in vals] for k, vals in history.history.items()},
        "run_dirs": [str(p) for p in run_dirs],
    }
    (out_dir / "training_report.json").write_text(json.dumps(train_report, indent=2), encoding="utf-8")
    np.save(out_dir / "X_train.npy", x_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", x_test)
    np.save(out_dir / "y_test.npy", y_test)

    print(
        "training_complete "
        f"train={len(train_idx)} test={len(test_idx)} "
        f"acc={eval_acc:.4f} model={model_path}"
    )


if __name__ == "__main__":
    main()

