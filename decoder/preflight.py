from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from decoder.backends.eegnet_adapter import EEGNetBackend
from decoder.config import DecoderStackConfig
from decoder.types import DecoderInputSpec


@dataclass(frozen=True)
class DecoderPreflightReport:
    ok: bool
    backend_mode: str
    backend_loaded: bool
    input_spec: DecoderInputSpec | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def run_decoder_preflight(
    config: DecoderStackConfig,
    stream_sample_rate_hz: int | None = None,
    stream_channel_count: int | None = None,
) -> DecoderPreflightReport:
    errors: list[str] = []
    warnings: list[str] = []
    sample_rate = 125 if stream_sample_rate_hz is None else max(int(stream_sample_rate_hz), 1)

    if stream_channel_count is not None:
        stream_channels = max(int(stream_channel_count), 1)
    elif config.eegnet.expected_channels is not None:
        stream_channels = max(int(config.eegnet.expected_channels), 1)
    else:
        errors.append("stream channel count is required when expected_channels is not set")
        stream_channels = 1

    if config.runtime.stride_samples <= 0:
        errors.append("runtime.stride_samples must be > 0")
    if config.eegnet.samples <= 0:
        errors.append("eegnet.samples must be > 0")
    if config.eegnet.expected_channels is not None and config.eegnet.expected_channels <= 0:
        errors.append("eegnet.expected_channels must be > 0")
    if len(config.eegnet.class_labels) == 0:
        errors.append("eegnet.class_labels must not be empty")

    model_path = config.eegnet.model_path.strip()
    if not model_path:
        if not config.eegnet.fallback_to_mock:
            errors.append("model_path is required when fallback_to_mock is disabled")
    else:
        model_file = Path(model_path)
        if not model_file.exists() and not config.eegnet.fallback_to_mock:
            errors.append(f"model file does not exist: {model_file}")

    if errors:
        return DecoderPreflightReport(
            ok=False,
            backend_mode="unloaded",
            backend_loaded=False,
            input_spec=None,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    backend = EEGNetBackend(config.eegnet)
    input_spec = backend.required_input_spec(sample_rate, stream_channels)
    try:
        backend.load(input_spec)
    except Exception as exc:
        errors.append(str(exc))
        return DecoderPreflightReport(
            ok=False,
            backend_mode=backend.backend_mode,
            backend_loaded=False,
            input_spec=input_spec,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    if input_spec.channels != stream_channels:
        warnings.append(
            f"runtime channels ({stream_channels}) differ from backend spec ({input_spec.channels})"
        )
    mode = backend.backend_mode
    if mode == "mock":
        warnings.append("backend is running in mock mode")
    backend.close()

    return DecoderPreflightReport(
        ok=True,
        backend_mode=mode,
        backend_loaded=True,
        input_spec=input_spec,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
