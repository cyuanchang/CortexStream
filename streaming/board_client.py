import threading
import time
from queue import Queue
from typing import Optional, Tuple

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams

from streaming.enums import StreamState
from streaming.pubsub_bus import DropPolicy, PubSubBus
from streaming.types import DataChunk, RawFrame, StreamConfig, StreamQueues, StreamStatus


class BrainFlowStreamService:
    _LEGACY_CHUNK_SUBSCRIBER = "legacy_chunk"
    _LEGACY_RAW_SUBSCRIBER = "legacy_raw"
    _PREFERRED_CHUNK_STATUS_SUBSCRIBERS = ("gui", _LEGACY_CHUNK_SUBSCRIBER)
    _PREFERRED_RAW_STATUS_SUBSCRIBERS = ("recorder", _LEGACY_RAW_SUBSCRIBER)

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

        self._chunk_bus = PubSubBus[DataChunk]()
        self._raw_bus = PubSubBus[RawFrame]()
        self.subscribe_chunks(self._LEGACY_CHUNK_SUBSCRIBER, config.chunk_queue_maxsize)
        self.subscribe_raw_frames(self._LEGACY_RAW_SUBSCRIBER, config.raw_queue_maxsize)

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
        if self._status.state not in (StreamState.STREAMING, StreamState.ERROR):
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._safe_stop_board_stream()
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
        """Return legacy EEG chunk queue for compatibility."""
        queue = self._chunk_bus.get_queue(self._LEGACY_CHUNK_SUBSCRIBER)
        if queue is None:
            return self.subscribe_chunks(
                self._LEGACY_CHUNK_SUBSCRIBER,
                self._config.chunk_queue_maxsize,
            )
        return queue

    def get_raw_queue(self) -> Queue[RawFrame]:
        """Return legacy raw frame queue for compatibility."""
        queue = self._raw_bus.get_queue(self._LEGACY_RAW_SUBSCRIBER)
        if queue is None:
            return self.subscribe_raw_frames(
                self._LEGACY_RAW_SUBSCRIBER,
                self._config.raw_queue_maxsize,
            )
        return queue

    def subscribe_chunks(
        self,
        subscriber_name: str,
        maxsize: int | None = None,
        drop_policy: DropPolicy = "keep_latest",
    ) -> Queue[DataChunk]:
        size = self._config.chunk_queue_maxsize if maxsize is None else maxsize
        return self._chunk_bus.subscribe(subscriber_name, size, drop_policy)

    def unsubscribe_chunks(self, subscriber_name: str) -> None:
        self._chunk_bus.unsubscribe(subscriber_name)

    def subscribe_raw_frames(
        self,
        subscriber_name: str,
        maxsize: int | None = None,
        drop_policy: DropPolicy = "keep_latest",
    ) -> Queue[RawFrame]:
        size = self._config.raw_queue_maxsize if maxsize is None else maxsize
        return self._raw_bus.subscribe(subscriber_name, size, drop_policy)

    def unsubscribe_raw_frames(self, subscriber_name: str) -> None:
        self._raw_bus.unsubscribe(subscriber_name)

    def get_chunk_subscriber_queue_size(self, subscriber_name: str) -> int:
        return self._chunk_bus.get_queue_size(subscriber_name)

    def get_chunk_subscriber_dropped(self, subscriber_name: str) -> int:
        return self._chunk_bus.get_dropped_count(subscriber_name)

    def get_raw_subscriber_queue_size(self, subscriber_name: str) -> int:
        return self._raw_bus.get_queue_size(subscriber_name)

    def get_raw_subscriber_dropped(self, subscriber_name: str) -> int:
        return self._raw_bus.get_dropped_count(subscriber_name)

    def get_status(self) -> StreamStatus:
        """Return snapshot of current stream and queue health counters."""
        with self._status_lock:
            status = StreamStatus(**self._status.__dict__)
            status.queue_size = self._preferred_queue_size(
                self._chunk_bus,
                self._PREFERRED_CHUNK_STATUS_SUBSCRIBERS,
            )
            status.raw_queue_size = self._preferred_queue_size(
                self._raw_bus,
                self._PREFERRED_RAW_STATUS_SUBSCRIBERS,
            )
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
            self._handle_acquisition_error(exc)

    def _pull_once(self) -> None:
        if self._board is None:
            return

        frame = self._board.get_board_data()
        if frame.size == 0 or frame.shape[1] == 0:
            return

        host_ts = time.time()
        raw_frame = RawFrame(sequence_id=self._raw_seq, host_timestamp=host_ts, frame_data=frame.copy())
        dropped_raw = self._raw_bus.publish(raw_frame)
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
            dropped_chunk = self._chunk_bus.publish(chunk)
            with self._status_lock:
                self._status.produced_chunks += 1
                self._status.dropped_chunks += dropped_chunk
            self._chunk_seq += 1

        with self._status_lock:
            self._status.produced_raw_frames += 1
            self._status.dropped_raw_frames += dropped_raw

    def _set_state(self, state: StreamState) -> None:
        with self._status_lock:
            self._status.state = state

    def _safe_stop_board_stream(self) -> None:
        if self._board is None:
            return
        try:
            self._board.stop_stream()
        except Exception:
            # Best-effort cleanup path; caller updates error/status.
            pass

    def _handle_acquisition_error(self, exc: Exception) -> None:
        """Persist error details and force stream cleanup before entering ERROR state."""
        self._stop_event.set()
        self._safe_stop_board_stream()
        if self._thread is not None and self._thread.is_alive():
            self._thread = None
        with self._status_lock:
            self._status.last_error = str(exc)
        self._set_state(StreamState.ERROR)

    @staticmethod
    def _preferred_queue_size(bus: PubSubBus, names: Tuple[str, ...]) -> int:
        for name in names:
            queue = bus.get_queue(name)
            if queue is not None:
                return queue.qsize()
        return 0
