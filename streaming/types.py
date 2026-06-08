from dataclasses import dataclass, field
from queue import Queue
from typing import Optional, Tuple

import numpy as np

from streaming.enums import BoardIdsEnum, StreamNumeric, StreamState


@dataclass
class StreamConfig:
    serial_port: str
    board_id: int = BoardIdsEnum.CYTON_DAISY
    chunk_size: int = StreamNumeric.CHUNK_SIZE
    chunk_queue_maxsize: int = StreamNumeric.CHUNK_QUEUE_MAXSIZE
    raw_queue_maxsize: int = StreamNumeric.RAW_QUEUE_MAXSIZE
    acquisition_poll_ms: int = StreamNumeric.ACQUISITION_POLL_MS


@dataclass
class DataChunk:
    sequence_id: int
    host_timestamp: float
    device_timestamp_start: float
    device_timestamp_end: float
    sample_rate_hz: int
    eeg_channel_indices: Tuple[int, ...]
    eeg_data: np.ndarray


@dataclass
class RawFrame:
    sequence_id: int
    host_timestamp: float
    frame_data: np.ndarray


@dataclass
class StreamStatus:
    state: StreamState = StreamState.DISCONNECTED
    produced_chunks: int = 0
    dropped_chunks: int = 0
    produced_raw_frames: int = 0
    dropped_raw_frames: int = 0
    last_error: str = ""
    started_monotonic: float = 0.0
    queue_size: int = 0
    raw_queue_size: int = 0


@dataclass
class StreamQueues:
    chunk_queue: Queue[DataChunk]
    raw_queue: Queue[RawFrame]
    eeg_channel_indices: Tuple[int, ...] = field(default_factory=tuple)
