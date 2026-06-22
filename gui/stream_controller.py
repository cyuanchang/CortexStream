from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Tuple

from decoder.config import (
    DecoderTaskType,
    DeploymentMode,
    build_decoder_runtime,
    resolve_decoder_stack_config,
)
from decoder.preflight import DecoderPreflightReport, run_decoder_preflight
from decoder.runtime import DecoderRuntime
from streaming.board_client import BrainFlowStreamService
from streaming.enums import BoardIdsEnum
from streaming.recorder import RawFrameRecorder
from streaming.types import DataChunk, RawFrame, StreamConfig, StreamStatus


class StreamController:
    """Own stream service + recorder lifecycle for the GUI."""

    def __init__(
        self,
        serial_port: str,
        recordings_dir: str,
        decoder_task_type: DecoderTaskType = "ssvep",
        decoder_model_path: str = "",
        decoder_manifest_path: str = "",
        decoder_deployment_mode: DeploymentMode = "dev",
    ):
        self._serial_port = serial_port
        self._recordings_dir = recordings_dir
        self._service: BrainFlowStreamService | None = None
        self._recorder: RawFrameRecorder | None = None
        self._decoder_config = resolve_decoder_stack_config(
            task_type=decoder_task_type,
            deployment_mode=decoder_deployment_mode,
            model_path=decoder_model_path,
            manifest_path=decoder_manifest_path,
        )
        self._decoder_runtime = build_decoder_runtime(self._decoder_config)
        self._decoder_deployment_mode = decoder_deployment_mode
        self._decoder_last_preflight: DecoderPreflightReport | None = None
        self._stream_started_monotonic: float | None = None
        self._gui_chunk_queue: Queue[DataChunk] | None = None
        self._recorder_raw_queue: Queue[RawFrame] | None = None
        self._active_session_dir: Path | None = None
        self._active_session_id: str = ""
        self._marker_log_lock = threading.Lock()

    @property
    def stream_started_monotonic(self) -> float | None:
        return self._stream_started_monotonic

    def connect(self) -> BrainFlowStreamService:
        if self._service is not None:
            return self._service
        config = StreamConfig(serial_port=self._serial_port, board_id=int(BoardIdsEnum.CYTON_DAISY))
        self._service = BrainFlowStreamService(config)
        self._service.connect()
        self._decoder_last_preflight = run_decoder_preflight(
            config=self._decoder_config,
            stream_sample_rate_hz=self._service.get_sample_rate_hz(),
            stream_channel_count=len(self._service.get_eeg_channels()),
        )
        self._gui_chunk_queue = self._service.subscribe_chunks("gui")
        return self._service

    def start(self) -> Path:
        if self._service is None:
            raise RuntimeError("Service is not connected.")
        preflight = run_decoder_preflight(
            config=self._decoder_config,
            stream_sample_rate_hz=self._service.get_sample_rate_hz(),
            stream_channel_count=len(self._service.get_eeg_channels()),
        )
        self._decoder_last_preflight = preflight
        if not preflight.ok:
            raise RuntimeError("decoder preflight failed: " + "; ".join(preflight.errors))
        if self._decoder_deployment_mode == "live" and preflight.backend_mode != "model":
            raise RuntimeError("live mode requires a real model (backend_mode=model)")
        recordings_dir = Path(self._recordings_dir)
        self._recorder_raw_queue = self._service.subscribe_raw_frames("recorder")
        self._recorder = RawFrameRecorder(
            raw_queue=self._recorder_raw_queue,
            output_root=str(recordings_dir),
            board_id=int(BoardIdsEnum.CYTON_DAISY),
            eeg_channels=self._service.get_eeg_channels(),
            sample_rate_hz=self._service.get_sample_rate_hz(),
        )
        session_dir = self._recorder.start()
        self._service.start()
        self._decoder_runtime.start(self._service)
        self._stream_started_monotonic = time.monotonic()
        self._active_session_dir = session_dir
        self._active_session_id = session_dir.name
        return session_dir

    def stop(self) -> int:
        if self._service is not None:
            self._service.stop()
        self._decoder_runtime.stop()
        frames_written = 0
        if self._recorder is not None:
            self._recorder.stop()
            frames_written = self._recorder.frames_written()
            self._recorder = None
        if self._service is not None:
            self._service.unsubscribe_raw_frames("recorder")
        self._recorder_raw_queue = None
        self._stream_started_monotonic = None
        self._active_session_dir = None
        self._active_session_id = ""
        return frames_written

    def disconnect(self) -> None:
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None
        self._decoder_runtime.stop()
        if self._service is not None:
            self._service.unsubscribe_raw_frames("recorder")
            self._service.unsubscribe_chunks("gui")
            self._service.disconnect()
            self._service = None
        self._gui_chunk_queue = None
        self._recorder_raw_queue = None
        self._stream_started_monotonic = None
        self._active_session_dir = None
        self._active_session_id = ""

    def get_service(self) -> BrainFlowStreamService | None:
        return self._service

    def get_status(self) -> StreamStatus | None:
        if self._service is None:
            return None
        return self._service.get_status()

    def get_chunk_queue(self) -> Queue[DataChunk] | None:
        return self._gui_chunk_queue

    def get_raw_queue(self) -> Queue[RawFrame] | None:
        return self._recorder_raw_queue

    def get_sample_rate_hz(self) -> int:
        if self._service is None:
            return 0
        return self._service.get_sample_rate_hz()

    def get_decoder_runtime(self) -> DecoderRuntime:
        return self._decoder_runtime

    def get_decoder_preflight(self) -> DecoderPreflightReport | None:
        return self._decoder_last_preflight

    def is_recording_active(self) -> bool:
        return self._active_session_dir is not None and self._stream_started_monotonic is not None

    def get_active_session_id(self) -> str:
        return self._active_session_id

    def get_active_session_dir(self) -> Path | None:
        return self._active_session_dir

    def append_marker_event(self, event: dict) -> bool:
        session_dir = self._active_session_dir
        if session_dir is None:
            return False
        payload = dict(event)
        payload.setdefault("session_id", self._active_session_id)
        payload.setdefault("stream_started_monotonic", self._stream_started_monotonic or 0.0)
        marker_path = session_dir / "ssvep_markers.jsonl"
        line = json.dumps(payload, separators=(",", ":"))
        with self._marker_log_lock:
            with marker_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return True

    def get_eeg_channels(self) -> Tuple[int, ...]:
        if self._service is None:
            return ()
        return self._service.get_eeg_channels()
