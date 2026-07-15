from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DecoderChunkMeta:
    sequence_id: int
    host_timestamp: float
    device_timestamp_start: float
    device_timestamp_end: float


DecoderInputLayout = Literal["channels_first", "channels_last"]


@dataclass(frozen=True)
class DecoderInputSpec:
    channels: int
    samples: int
    sample_rate_hz: int
    layout: DecoderInputLayout = "channels_last"


@dataclass(frozen=True)
class DecoderOutput:
    label: str
    confidence: float
    scores: tuple[float, ...]
    inference_ms: float
    chunk_sequence_end: int
    device_timestamp_end: float
    host_timestamp: float


@dataclass(frozen=True)
class DecoderRuntimeStatus:
    running: bool
    chunks_received: int
    last_sequence_id: int
    queue_size: int
    dropped_by_bus: int
    backend_name: str = ""
    backend_mode: str = ""
    backend_loaded: bool = False
    inference_count: int = 0
    last_inference_ms: float = 0.0
    last_label: str = ""
    last_confidence: float = 0.0
    failures: int = 0
    last_error: str = ""
