from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from Realtime_processing.decoder_preprocessing import DecoderPreprocessConfig
from decoder.types import DecoderInputLayout

DecoderTaskType = Literal["ssvep", "mi"]
DeploymentMode = Literal["dev", "live"]
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DecoderModelManifest:
    schema_version: int
    task_type: DecoderTaskType
    model_path: str
    class_labels: tuple[str, ...]
    samples: int
    stride_samples: int
    expected_channels: int | None
    input_layout: DecoderInputLayout
    preprocess: DecoderPreprocessConfig
    deployment_mode: DeploymentMode = "live"
    version: str = ""
    created_at: str = ""
    checksum: str = ""
    subscriber_name: str = "decoder"
    queue_maxsize: int = 512

    def resolved_model_path(self, manifest_path: Path) -> str:
        raw = self.model_path.strip()
        if not raw:
            return ""
        path = Path(raw)
        if path.is_absolute():
            return str(path)
        return str((manifest_path.parent / path).resolve())


def load_model_manifest(path: str | Path) -> DecoderModelManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    return _parse_manifest_dict(payload, manifest_path)


def _parse_manifest_dict(payload: dict[str, Any], manifest_path: Path) -> DecoderModelManifest:
    schema_version = int(payload.get("schema_version", MANIFEST_SCHEMA_VERSION))
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={schema_version}, expected {MANIFEST_SCHEMA_VERSION}"
        )
    task_type = _as_task_type(payload.get("task_type", ""))
    input_layout = _as_layout(payload.get("input_layout", "channels_last"))
    class_labels = tuple(str(v) for v in payload.get("class_labels", []))
    if not class_labels:
        raise ValueError("manifest.class_labels must contain at least one label")
    preprocess_payload = payload.get("preprocess", {})
    if not isinstance(preprocess_payload, dict):
        raise ValueError("manifest.preprocess must be an object")

    preprocess = DecoderPreprocessConfig(
        enabled=bool(preprocess_payload.get("enabled", True)),
        bandpass_low_hz=float(preprocess_payload.get("bandpass_low_hz", 4.0)),
        bandpass_high_hz=float(preprocess_payload.get("bandpass_high_hz", 40.0)),
        apply_notch=bool(preprocess_payload.get("apply_notch", True)),
        notch_hz=float(preprocess_payload.get("notch_hz", 60.0)),
        apply_car=bool(preprocess_payload.get("apply_car", False)),
        zscore_per_channel=bool(preprocess_payload.get("zscore_per_channel", True)),
        zscore_eps=float(preprocess_payload.get("zscore_eps", 1e-6)),
        channel_order=tuple(int(v) for v in preprocess_payload.get("channel_order", [])),
    )

    expected_channels_raw = payload.get("expected_channels", None)
    expected_channels = None if expected_channels_raw is None else int(expected_channels_raw)

    manifest = DecoderModelManifest(
        schema_version=schema_version,
        task_type=task_type,
        model_path=str(payload.get("model_path", "")).strip(),
        class_labels=class_labels,
        samples=max(int(payload.get("samples", 1)), 1),
        stride_samples=max(int(payload.get("stride_samples", 1)), 1),
        expected_channels=expected_channels,
        input_layout=input_layout,
        preprocess=preprocess,
        deployment_mode=_as_deployment_mode(payload.get("deployment_mode", "live")),
        version=str(payload.get("version", "")),
        created_at=str(payload.get("created_at", "")),
        checksum=str(payload.get("checksum", "")),
        subscriber_name=str(payload.get("subscriber_name", "decoder")),
        queue_maxsize=max(int(payload.get("queue_maxsize", 512)), 1),
    )
    _validate_manifest(manifest, manifest_path)
    return manifest


def _validate_manifest(manifest: DecoderModelManifest, manifest_path: Path) -> None:
    if manifest.samples <= 0:
        raise ValueError("manifest.samples must be > 0")
    if manifest.stride_samples <= 0:
        raise ValueError("manifest.stride_samples must be > 0")
    if manifest.expected_channels is not None and manifest.expected_channels <= 0:
        raise ValueError("manifest.expected_channels must be > 0")
    if manifest.preprocess.bandpass_low_hz <= 0.0:
        raise ValueError("manifest.preprocess.bandpass_low_hz must be > 0")
    if manifest.preprocess.bandpass_high_hz <= manifest.preprocess.bandpass_low_hz:
        raise ValueError("manifest.preprocess.bandpass_high_hz must be greater than bandpass_low_hz")
    if manifest.preprocess.channel_order and manifest.expected_channels is not None:
        if len(manifest.preprocess.channel_order) != manifest.expected_channels:
            raise ValueError("manifest.preprocess.channel_order length must match expected_channels")
    if manifest.deployment_mode == "live" and not manifest.model_path.strip():
        raise ValueError("manifest.model_path is required in live mode")
    if manifest.model_path.strip():
        resolved = Path(manifest.resolved_model_path(manifest_path))
        if not resolved.exists() and manifest.deployment_mode == "live":
            raise ValueError(f"live mode model path does not exist: {resolved}")


def _as_task_type(value: Any) -> DecoderTaskType:
    raw = str(value).strip().lower()
    if raw not in ("ssvep", "mi"):
        raise ValueError("manifest.task_type must be one of: ssvep, mi")
    return raw  # type: ignore[return-value]


def _as_layout(value: Any) -> DecoderInputLayout:
    raw = str(value).strip().lower()
    if raw not in ("channels_first", "channels_last"):
        raise ValueError("manifest.input_layout must be channels_first or channels_last")
    return raw  # type: ignore[return-value]


def _as_deployment_mode(value: Any) -> DeploymentMode:
    raw = str(value).strip().lower()
    if raw not in ("dev", "live"):
        raise ValueError("manifest.deployment_mode must be one of: dev, live")
    return raw  # type: ignore[return-value]
