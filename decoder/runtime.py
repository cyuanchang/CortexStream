from __future__ import annotations

import threading
import time
from dataclasses import replace
from queue import Empty, Queue

import numpy as np

from Realtime_processing.decoder_preprocessing import DecoderPreprocessConfig, run_decoder_preprocessing
from decoder.contracts import DecoderBackend
from decoder.types import DecoderChunkMeta, DecoderInputSpec, DecoderOutput, DecoderRuntimeStatus
from streaming.board_client import BrainFlowStreamService
from streaming.types import DataChunk


class DecoderRuntime:
    """Model-agnostic decoder runtime that consumes stream chunks and calls a backend."""

    def __init__(
        self,
        backend: DecoderBackend,
        subscriber_name: str = "decoder",
        queue_maxsize: int = 512,
        stride_samples: int = 16,
        strict_start: bool = False,
        preprocess_config: DecoderPreprocessConfig | None = None,
    ) -> None:
        self._backend = backend
        self._subscriber_name = subscriber_name
        self._queue_maxsize = queue_maxsize
        self._stride_samples = max(int(stride_samples), 1)
        self._strict_start = strict_start
        self._preprocess_config = DecoderPreprocessConfig() if preprocess_config is None else preprocess_config
        self._queue: Queue[DataChunk] | None = None
        self._service: BrainFlowStreamService | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._chunks_received = 0
        self._inference_count = 0
        self._failures = 0
        self._last_inference_ms = 0.0
        self._last_label = ""
        self._last_confidence = 0.0
        self._last_error = ""
        self._backend_loaded = False
        self._last_meta: DecoderChunkMeta | None = None
        self._last_output: DecoderOutput | None = None
        self._input_spec: DecoderInputSpec | None = None
        self._sample_accumulator = 0
        self._window_samples = 0
        self._buffer = np.empty((0, 0), dtype=np.float64)

    @property
    def subscriber_name(self) -> str:
        return self._subscriber_name

    def start(self, service: BrainFlowStreamService) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._service = service
        sample_rate_hz = max(int(service.get_sample_rate_hz()), 1)
        channel_count = len(service.get_eeg_channels())
        self._input_spec = self._backend.required_input_spec(sample_rate_hz, channel_count)
        self._window_samples = max(int(self._input_spec.samples), 1)
        self._buffer = np.zeros((channel_count, self._window_samples), dtype=np.float64)
        self._chunks_received = 0
        self._inference_count = 0
        self._failures = 0
        self._sample_accumulator = 0
        self._last_inference_ms = 0.0
        self._last_label = ""
        self._last_confidence = 0.0
        self._last_error = ""
        self._backend_loaded = False
        self._last_meta = None
        self._last_output = None
        try:
            self._backend.load(self._input_spec)
            self._backend_loaded = True
            if self._backend.backend_mode == "mock" and self._strict_start:
                raise RuntimeError(
                    "strict_start requires real model mode, but backend resolved to mock mode"
                )
        except Exception as exc:
            # On load failure, record error counters; re-raise only when strict_start is enabled.
            self._backend_loaded = False
            self._failures += 1
            self._last_error = str(exc) or self._backend.last_load_error
            if self._strict_start:
                raise RuntimeError(self._last_error)
        self._queue = service.subscribe_chunks(
            self._subscriber_name,
            maxsize=self._queue_maxsize,
            drop_policy="keep_latest",
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._backend.close()
        except Exception:
            pass
        self._backend_loaded = False
        if self._service is not None:
            self._service.unsubscribe_chunks(self._subscriber_name)
        self._queue = None
        self._service = None

    def get_latest_window(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()

    def get_last_chunk_meta(self) -> DecoderChunkMeta | None:
        with self._lock:
            return self._last_meta

    def get_latest_output(self) -> DecoderOutput | None:
        with self._lock:
            return self._last_output

    def get_status(self) -> DecoderRuntimeStatus:
        running = self._thread is not None and self._thread.is_alive()
        queue_size = 0
        dropped = 0
        if self._service is not None:
            queue_size = self._service.get_chunk_subscriber_queue_size(self._subscriber_name)
            dropped = self._service.get_chunk_subscriber_dropped(self._subscriber_name)
        with self._lock:
            last_seq = -1 if self._last_meta is None else self._last_meta.sequence_id
            received = self._chunks_received
            inference_count = self._inference_count
            last_inference_ms = self._last_inference_ms
            last_label = self._last_label
            last_confidence = self._last_confidence
            failures = self._failures
            last_error = self._last_error
        return DecoderRuntimeStatus(
            running=running,
            chunks_received=received,
            last_sequence_id=last_seq,
            queue_size=queue_size,
            dropped_by_bus=dropped,
            backend_name=self._backend.backend_name,
            backend_mode=self._backend.backend_mode,
            backend_loaded=self._backend_loaded,
            inference_count=inference_count,
            last_inference_ms=last_inference_ms,
            last_label=last_label,
            last_confidence=last_confidence,
            failures=failures,
            last_error=last_error,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            queue_ref = self._queue
            if queue_ref is None:
                break
            try:
                chunk = queue_ref.get(timeout=0.2)
            except Empty:
                continue
            self._consume_chunk(chunk)

    def _consume_chunk(self, chunk: DataChunk) -> None:
        chunk_data = np.asarray(chunk.eeg_data, dtype=np.float64)
        if chunk_data.ndim != 2:
            return
        if self._buffer.size == 0:
            return
        if self._input_spec is None:
            return
        if chunk_data.shape[0] != self._buffer.shape[0] or chunk_data.shape[0] != self._input_spec.channels:
            self._mark_failure("chunk channel count mismatch")
            return
        chunk_len = chunk_data.shape[1]
        if chunk_len <= 0:
            return
        with self._lock:
            if chunk_len >= self._window_samples:
                self._buffer[:, :] = chunk_data[:, -self._window_samples :]
            else:
                self._buffer[:, :-chunk_len] = self._buffer[:, chunk_len:]
                self._buffer[:, -chunk_len:] = chunk_data
            self._chunks_received += 1
            self._sample_accumulator += chunk_len
            self._last_meta = DecoderChunkMeta(
                sequence_id=chunk.sequence_id,
                host_timestamp=chunk.host_timestamp,
                device_timestamp_start=chunk.device_timestamp_start,
                device_timestamp_end=chunk.device_timestamp_end,
            )
            should_infer = self._backend_loaded and self._sample_accumulator >= self._stride_samples
            if should_infer:
                self._sample_accumulator %= self._stride_samples
                window = self._buffer.copy()
                meta = self._last_meta
            else:
                window = None
                meta = None
        if window is not None and meta is not None:
            self._run_inference(window, meta)

    def _run_inference(self, window: np.ndarray, meta: DecoderChunkMeta) -> None:
        start = time.perf_counter()
        try:
            if self._input_spec is None:
                raise RuntimeError("decoder input spec is not initialized")
            prepared = run_decoder_preprocessing(
                window=window,
                sample_rate_hz=self._input_spec.sample_rate_hz,
                config=self._preprocess_config,
            )
            output = self._backend.infer(prepared, meta)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            output = replace(output, inference_ms=elapsed_ms)
            with self._lock:
                self._inference_count += 1
                self._last_inference_ms = elapsed_ms
                self._last_label = output.label
                self._last_confidence = output.confidence
                self._last_output = output
        except Exception as exc:
            self._mark_failure(str(exc))

    def _mark_failure(self, message: str) -> None:
        with self._lock:
            self._failures += 1
            self._last_error = message
