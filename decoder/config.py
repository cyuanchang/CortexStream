from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from Realtime_processing.decoder_preprocessing import DecoderPreprocessConfig
from decoder.backends.eegnet_adapter import EEGNetBackend, EEGNetBackendConfig
from decoder.model_manifest import DecoderModelManifest, DeploymentMode, load_model_manifest
from decoder.runtime import DecoderRuntime

DecoderTaskType = Literal["ssvep", "mi"]


@dataclass(frozen=True)
class DecoderRuntimeConfig:
    subscriber_name: str = "decoder"
    queue_maxsize: int = 512
    stride_samples: int = 16
    strict_start: bool = False


@dataclass(frozen=True)
class DecoderStackConfig:
    runtime: DecoderRuntimeConfig = field(default_factory=DecoderRuntimeConfig)
    preprocess: DecoderPreprocessConfig = field(default_factory=DecoderPreprocessConfig)
    eegnet: EEGNetBackendConfig = field(default_factory=EEGNetBackendConfig)


def build_decoder_runtime(config: DecoderStackConfig | None = None) -> DecoderRuntime:
    effective = DecoderStackConfig() if config is None else config
    backend = EEGNetBackend(effective.eegnet)
    return DecoderRuntime(
        backend=backend,
        subscriber_name=effective.runtime.subscriber_name,
        queue_maxsize=effective.runtime.queue_maxsize,
        stride_samples=effective.runtime.stride_samples,
        strict_start=effective.runtime.strict_start,
        preprocess_config=effective.preprocess,
    )


def get_task_profile(task_type: DecoderTaskType) -> DecoderStackConfig:
    if task_type == "ssvep":
        return DecoderStackConfig(
            runtime=DecoderRuntimeConfig(stride_samples=16, strict_start=False),
            preprocess=DecoderPreprocessConfig(
                enabled=True,
                bandpass_low_hz=4.0,
                bandpass_high_hz=40.0,
                apply_notch=True,
                notch_hz=60.0,
                apply_car=False,
                zscore_per_channel=True,
            ),
            eegnet=EEGNetBackendConfig(
                model_path="",
                class_labels=("idle", "target"),
                samples=256,
                expected_channels=16,
                input_layout="channels_last",
                fallback_to_mock=True,
            ),
        )
    if task_type == "mi":
        return DecoderStackConfig(
            runtime=DecoderRuntimeConfig(stride_samples=16, strict_start=False),
            preprocess=DecoderPreprocessConfig(
                enabled=True,
                bandpass_low_hz=8.0,
                bandpass_high_hz=30.0,
                apply_notch=True,
                notch_hz=60.0,
                apply_car=True,
                zscore_per_channel=True,
            ),
            eegnet=EEGNetBackendConfig(
                model_path="",
                class_labels=("left", "right"),
                samples=256,
                expected_channels=16,
                input_layout="channels_last",
                fallback_to_mock=True,
            ),
        )
    raise ValueError(f"unsupported decoder task type: {task_type}")


def build_decoder_runtime_for_task(
    task_type: DecoderTaskType,
    model_path: str = "",
    fallback_to_mock: bool = True,
    deployment_mode: DeploymentMode = "dev",
) -> DecoderRuntime:
    config = get_task_profile(task_type)
    eegnet = EEGNetBackendConfig(
        model_path=model_path.strip(),
        class_labels=config.eegnet.class_labels,
        samples=config.eegnet.samples,
        expected_channels=config.eegnet.expected_channels,
        input_layout=config.eegnet.input_layout,
        fallback_to_mock=fallback_to_mock if deployment_mode == "dev" else False,
    )
    runtime = DecoderRuntimeConfig(
        subscriber_name=config.runtime.subscriber_name,
        queue_maxsize=config.runtime.queue_maxsize,
        stride_samples=config.runtime.stride_samples,
        strict_start=deployment_mode == "live",
    )
    return build_decoder_runtime(DecoderStackConfig(runtime=runtime, preprocess=config.preprocess, eegnet=eegnet))


def build_decoder_stack_from_manifest(
    manifest: DecoderModelManifest,
    manifest_path: str | Path,
) -> DecoderStackConfig:
    manifest_file = Path(manifest_path)
    model_path = manifest.resolved_model_path(manifest_file)
    runtime = DecoderRuntimeConfig(
        subscriber_name=manifest.subscriber_name,
        queue_maxsize=manifest.queue_maxsize,
        stride_samples=manifest.stride_samples,
        strict_start=manifest.deployment_mode == "live",
    )
    eegnet = EEGNetBackendConfig(
        model_path=model_path,
        class_labels=manifest.class_labels,
        samples=manifest.samples,
        expected_channels=manifest.expected_channels,
        input_layout=manifest.input_layout,
        fallback_to_mock=manifest.deployment_mode != "live",
    )
    return DecoderStackConfig(runtime=runtime, preprocess=manifest.preprocess, eegnet=eegnet)


def build_decoder_runtime_from_manifest(manifest_path: str | Path) -> DecoderRuntime:
    manifest = load_model_manifest(manifest_path)
    return build_decoder_runtime(build_decoder_stack_from_manifest(manifest, manifest_path))


def resolve_decoder_stack_config(
    task_type: DecoderTaskType,
    deployment_mode: DeploymentMode,
    model_path: str = "",
    manifest_path: str = "",
) -> DecoderStackConfig:
    if manifest_path.strip():
        manifest = load_model_manifest(manifest_path)
        return build_decoder_stack_from_manifest(manifest, manifest_path)

    base = get_task_profile(task_type)
    runtime = DecoderRuntimeConfig(
        subscriber_name=base.runtime.subscriber_name,
        queue_maxsize=base.runtime.queue_maxsize,
        stride_samples=base.runtime.stride_samples,
        strict_start=deployment_mode == "live",
    )
    eegnet = EEGNetBackendConfig(
        model_path=model_path.strip(),
        class_labels=base.eegnet.class_labels,
        samples=base.eegnet.samples,
        expected_channels=base.eegnet.expected_channels,
        input_layout=base.eegnet.input_layout,
        fallback_to_mock=deployment_mode != "live",
    )
    return DecoderStackConfig(runtime=runtime, preprocess=base.preprocess, eegnet=eegnet)
