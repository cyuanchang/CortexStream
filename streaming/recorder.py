import json
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Optional, Tuple

from brainflow.data_filter import DataFilter

from streaming.types import RawFrame


class RawFrameRecorder:
    def __init__(
        self,
        raw_queue: Queue[RawFrame],
        output_root: str,
        board_id: int,
        eeg_channels: Tuple[int, ...],
        sample_rate_hz: int,
    ):
        self._raw_queue = raw_queue
        self._output_root = Path(output_root)
        self._board_id = board_id
        self._eeg_channels = eeg_channels
        self._sample_rate_hz = sample_rate_hz
        self._session_dir: Optional[Path] = None
        self._csv_path: Optional[Path] = None
        self._meta_path: Optional[Path] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames_written = 0

    def start(self) -> Path:
        """Start recorder worker and create output session directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = self._output_root / f"run_{timestamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._session_dir / "raw_frames.csv"
        self._meta_path = self._session_dir / "metadata.json"
        self._write_metadata()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._session_dir

    def stop(self) -> None:
        """Stop recorder worker after flushing queued raw frames."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
            self._thread = None

    def frames_written(self) -> int:
        """Return count of raw frames persisted to disk."""
        return self._frames_written

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._raw_queue.empty():
            try:
                frame = self._raw_queue.get(timeout=0.2)
            except Empty:
                continue
            self._write_frame(frame)
            self._frames_written += 1

    def _write_frame(self, frame: RawFrame) -> None:
        if self._csv_path is None:
            return
        DataFilter.write_file(frame.frame_data, str(self._csv_path), "a")

    def _write_metadata(self) -> None:
        if self._meta_path is None:
            return
        payload = {
            "board_id": self._board_id,
            "sample_rate_hz": self._sample_rate_hz,
            "eeg_channels": list(self._eeg_channels),
            "format": "raw BrainFlow frames appended via DataFilter.write_file",
        }
        self._meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
