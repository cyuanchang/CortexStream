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
from decoder.backends.eegnet_adapter import EEGNetBackend, EEGNetBackendConfig
from decoder.config import (
    DecoderTaskType,
    DecoderStackConfig,
    build_decoder_stack_from_manifest,
    resolve_decoder_stack_config,
)
from decoder.model_manifest import load_model_manifest
from decoder.preflight import run_decoder_preflight
from decoder.types import DecoderChunkMeta
from scripts.recording_io import load_recording

EXIT_OK = 0
EXIT_PREFLIGHT_FAILED = 1
EXIT_MOCK_NOT_ALLOWED = 2
EXIT_RUNTIME_FAILED = 3
EXIT_ACCEPTANCE_FAILED = 4


def _resolve_stack_config(
    task_type: DecoderTaskType,
    model_path: str,
    manifest_path: str,
    allow_mock: bool,
) -> DecoderStackConfig:
    if manifest_path.strip():
        manifest = load_model_manifest(manifest_path)
        config = build_decoder_stack_from_manifest(manifest, manifest_path)
        if allow_mock:
            eegnet = EEGNetBackendConfig(
                model_path=config.eegnet.model_path,
                class_labels=config.eegnet.class_labels,
                samples=config.eegnet.samples,
                expected_channels=config.eegnet.expected_channels,
                input_layout=config.eegnet.input_layout,
                fallback_to_mock=True,
            )
            return DecoderStackConfig(runtime=config.runtime, preprocess=config.preprocess, eegnet=eegnet)
        return config
    return resolve_decoder_stack_config(
        task_type=task_type,
        deployment_mode="dev" if allow_mock else "live",
        model_path=model_path,
        manifest_path="",
    )


def run_replay(
    recording_dir: Path,
    config: DecoderStackConfig,
    allow_mock: bool,
    min_inferences: int,
    max_failures: int,
    min_mean_confidence: float,
    report_json: str,
) -> int:
    recording = load_recording(recording_dir)
    preflight = run_decoder_preflight(
        config=config,
        stream_sample_rate_hz=recording.sample_rate_hz,
        stream_channel_count=recording.eeg_matrix.shape[0],
    )
    if not preflight.ok:
        _emit_report(
            report_json=report_json,
            payload={
                "ok": False,
                "exit_code": EXIT_PREFLIGHT_FAILED,
                "stage": "preflight",
                "errors": list(preflight.errors),
                "warnings": list(preflight.warnings),
            },
        )
        print("preflight failed:", "; ".join(preflight.errors))
        return EXIT_PREFLIGHT_FAILED

    backend = EEGNetBackend(config.eegnet)
    input_spec = backend.required_input_spec(
        stream_sample_rate_hz=recording.sample_rate_hz,
        stream_channel_count=recording.eeg_matrix.shape[0],
    )
    backend.load(input_spec)
    if backend.backend_mode == "mock" and not allow_mock:
        backend.close()
        _emit_report(
            report_json=report_json,
            payload={
                "ok": False,
                "exit_code": EXIT_MOCK_NOT_ALLOWED,
                "stage": "backend_mode",
                "errors": ["backend resolved to mock mode"],
                "warnings": [],
            },
        )
        print("replay failed: backend resolved to mock mode")
        return EXIT_MOCK_NOT_ALLOWED

    stride = max(int(config.runtime.stride_samples), 1)
    window_samples = int(input_spec.samples)
    if recording.eeg_matrix.shape[1] < window_samples:
        backend.close()
        _emit_report(
            report_json=report_json,
            payload={
                "ok": False,
                "exit_code": EXIT_RUNTIME_FAILED,
                "stage": "windowing",
                "errors": [
                    (
                        "recording too short: "
                        f"{recording.eeg_matrix.shape[1]} samples, need at least {window_samples}"
                    )
                ],
                "warnings": [],
            },
        )
        print(
            "replay failed: recording too short: "
            f"{recording.eeg_matrix.shape[1]} samples, need at least {window_samples}"
        )
        return EXIT_RUNTIME_FAILED

    inference_count = 0
    failures = 0
    labels: dict[str, int] = {}
    confidences: list[float] = []
    for end_idx in range(window_samples, recording.eeg_matrix.shape[1] + 1, stride):
        try:
            window = recording.eeg_matrix[:, end_idx - window_samples : end_idx]
            prepared = run_decoder_preprocessing(window, recording.sample_rate_hz, config.preprocess)
            output = backend.infer(
                prepared,
                DecoderChunkMeta(
                    sequence_id=inference_count,
                    host_timestamp=float(end_idx / recording.sample_rate_hz),
                    device_timestamp_start=float((end_idx - window_samples) / recording.sample_rate_hz),
                    device_timestamp_end=float(end_idx / recording.sample_rate_hz),
                ),
            )
            inference_count += 1
            labels[output.label] = labels.get(output.label, 0) + 1
            confidences.append(output.confidence)
        except Exception:
            failures += 1

    mode = backend.backend_mode
    backend.close()
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    print(f"mode={mode} model_path={config.eegnet.model_path or '<mock>'}")
    print(f"inferences={inference_count} failures={failures} mean_conf={mean_conf:.4f}")
    print(f"label_counts={labels}")
    ok = (
        inference_count >= min_inferences
        and failures <= max_failures
        and (mean_conf >= min_mean_confidence or inference_count == 0)
    )
    exit_code = EXIT_OK if ok else EXIT_ACCEPTANCE_FAILED
    _emit_report(
        report_json=report_json,
        payload={
            "ok": ok,
            "exit_code": exit_code,
            "stage": "acceptance",
            "backend_mode": mode,
            "model_path": config.eegnet.model_path,
            "inference_count": inference_count,
            "failures": failures,
            "mean_confidence": mean_conf,
            "label_counts": labels,
            "min_inferences": min_inferences,
            "max_failures": max_failures,
            "min_mean_confidence": min_mean_confidence,
            "recording_dir": str(recording_dir),
        },
    )
    if not ok:
        print("replay acceptance failed")
    return exit_code


def _emit_report(report_json: str, payload: dict) -> None:
    if not report_json.strip():
        return
    report_path = Path(report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline decoder replay validator")
    parser.add_argument("--recording-dir", required=True, type=str)
    parser.add_argument("--task-type", default="ssvep", choices=("ssvep", "mi"))
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--manifest", default="", type=str)
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--min-inferences", default=10, type=int)
    parser.add_argument("--max-failures", default=0, type=int)
    parser.add_argument("--min-mean-confidence", default=0.0, type=float)
    parser.add_argument("--report-json", default="", type=str)
    args = parser.parse_args()
    config = _resolve_stack_config(
        task_type=args.task_type,
        model_path=args.model_path,
        manifest_path=args.manifest,
        allow_mock=args.allow_mock,
    )
    exit_code = run_replay(
        recording_dir=Path(args.recording_dir),
        config=config,
        allow_mock=args.allow_mock,
        min_inferences=max(args.min_inferences, 0),
        max_failures=max(args.max_failures, 0),
        min_mean_confidence=max(args.min_mean_confidence, 0.0),
        report_json=args.report_json,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
