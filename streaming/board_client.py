import threading
import time
from queue import Queue
from typing import Optional, Tuple

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams

from streaming.enums import StreamState
from streaming.queue_bus import QueuePublisher
from streaming.types import DataChunk, RawFrame, StreamConfig, StreamQueues, StreamStatus


class BrainFlowStreamService:
    def __init__(self, config: StreamConfig):
        self._config = config
        self._board: Optional[BoardShim] = None
        self._status = StreamStatus()
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._chunk_seq = 0
        self._raw_seq = 0
        self._sample_rate_hz = 0
        self._eeg_channels: Tuple[int, ...] = ()
        self._timestamp_channel = -1
        self._pending_eeg = np.empty((0, 0))
        self._pending_timestamps = np.empty(0)

        self._chunk_publisher = QueuePublisher[DataChunk](config.chunk_queue_maxsize)
        self._raw_publisher = QueuePublisher[RawFrame](config.raw_queue_maxsize)

    def connect(self) -> None:
        """Prepare BrainFlow session for configured board and serial port."""
        if self._status.state != StreamState.DISCONNECTED:
            return

        params = BrainFlowInputParams()
        params.serial_port = self._config.serial_port
        self._board = BoardShim(self._config.board_id, params)
        self._board.prepare_session()

        self._sample_rate_hz = BoardShim.get_sampling_rate(self._config.board_id)
        self._eeg_channels = tuple(BoardShim.get_eeg_channels(self._config.board_id))
        self._timestamp_channel = BoardShim.get_timestamp_channel(self._config.board_id)
        self._pending_eeg = np.empty((len(self._eeg_channels), 0))
        self._pending_timestamps = np.empty(0)
        self._set_state(StreamState.CONNECTED)

    def start(self) -> None:
        """Start board streaming and acquisition thread."""
        if self._board is None or self._status.state != StreamState.CONNECTED:
            return

        self._board.start_stream()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_acquisition, daemon=True)
        self._thread.start()
        with self._status_lock:
            self._status.started_monotonic = time.monotonic()
        self._set_state(StreamState.STREAMING)

    def stop(self) -> None:
        """Stop acquisition thread and board stream."""
        if self._status.state != StreamState.STREAMING:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._board is not None:
            self._board.stop_stream()
        self._set_state(StreamState.CONNECTED)

    def disconnect(self) -> None:
        """Release BrainFlow session and reset stream state."""
        if self._status.state == StreamState.STREAMING:
            self.stop()
        if self._board is not None:
            self._board.release_session()
            self._board = None
        self._set_state(StreamState.DISCONNECTED)

    def get_queue(self) -> Queue[DataChunk]:
        """Return queue containing EEG-only data chunks for consumers."""
        return self._chunk_publisher.get_queue()

    def get_raw_queue(self) -> Queue[RawFrame]:
        """Return queue containing full raw frames for recording side-path."""
        return self._raw_publisher.get_queue()

    def get_status(self) -> StreamStatus:
        """Return snapshot of current stream and queue health counters."""
        with self._status_lock:
            status = StreamStatus(**self._status.__dict__)
            status.queue_size = self._chunk_publisher.get_queue().qsize()
            status.raw_queue_size = self._raw_publisher.get_queue().qsize()
            return status

    def get_sample_rate_hz(self) -> int:
        """Return prepared board sampling rate for the active session."""
        return self._sample_rate_hz

    def get_eeg_channels(self) -> Tuple[int, ...]:
        """Return EEG channel index tuple for configured board."""
        return self._eeg_channels

    def get_queues(self) -> StreamQueues:
        """Return both chunk and raw queues plus EEG channel index metadata."""
        return StreamQueues(
            chunk_queue=self.get_queue(),
            raw_queue=self.get_raw_queue(),
            eeg_channel_indices=self._eeg_channels,
        )

    def _run_acquisition(self) -> None:
        try:
            poll_s = self._config.acquisition_poll_ms / 1000.0
            while not self._stop_event.is_set():
                self._pull_once()
                time.sleep(poll_s)
        except Exception as exc:
            with self._status_lock:
                self._status.last_error = str(exc)
            self._set_state(StreamState.ERROR)

    def _pull_once(self) -> None:
        if self._board is None:
            return

        frame = self._board.get_board_data()
        if frame.size == 0 or frame.shape[1] == 0:
            return

        host_ts = time.time()
        raw_frame = RawFrame(sequence_id=self._raw_seq, host_timestamp=host_ts, frame_data=frame.copy())
        dropped_raw = self._raw_publisher.publish_keep_latest(raw_frame)
        self._raw_seq += 1

        eeg = frame[list(self._eeg_channels), :]
        ts = frame[self._timestamp_channel, :]
        self._pending_eeg = np.concatenate((self._pending_eeg, eeg), axis=1)
        self._pending_timestamps = np.concatenate((self._pending_timestamps, ts))

        while self._pending_eeg.shape[1] >= self._config.chunk_size:
            eeg_chunk = self._pending_eeg[:, : self._config.chunk_size].copy()
            ts_chunk = self._pending_timestamps[: self._config.chunk_size].copy()
            self._pending_eeg = self._pending_eeg[:, self._config.chunk_size :]
            self._pending_timestamps = self._pending_timestamps[self._config.chunk_size :]
            chunk = DataChunk(
                sequence_id=self._chunk_seq,
                host_timestamp=host_ts,
                device_timestamp_start=float(ts_chunk[0]),
                device_timestamp_end=float(ts_chunk[-1]),
                sample_rate_hz=self._sample_rate_hz,
                eeg_channel_indices=self._eeg_channels,
                eeg_data=eeg_chunk,
            )
            dropped_chunk = self._chunk_publisher.publish_keep_latest(chunk)
            with self._status_lock:
                self._status.produced_chunks += 1
                if dropped_chunk:
                    self._status.dropped_chunks += 1
            self._chunk_seq += 1

        with self._status_lock:
            self._status.produced_raw_frames += 1
            if dropped_raw:
                self._status.dropped_raw_frames += 1

    def _set_state(self, state: StreamState) -> None:
        with self._status_lock:
            self._status.state = state
