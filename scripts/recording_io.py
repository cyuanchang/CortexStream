from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from brainflow.data_filter import DataFilter


@dataclass(frozen=True)
class RecordingData:
    recording_dir: Path
    metadata: dict
    raw_matrix: np.ndarray
    eeg_matrix: np.ndarray
    eeg_channels: tuple[int, ...]
    sample_rate_hz: int

    @property
    def sample_count(self) -> int:
        return int(self.raw_matrix.shape[1])

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate_hz <= 0:
            return 0.0
        return float(self.sample_count / float(self.sample_rate_hz))


def load_recording(recording_dir: str | Path) -> RecordingData:
    base = Path(recording_dir)
    metadata_path = base / "metadata.json"
    csv_path = base / "raw_frames.csv"
    if not metadata_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"recording dir missing metadata/raw file: {base}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    eeg_channels = tuple(int(v) for v in metadata.get("eeg_channels", []))
    if not eeg_channels:
        raise RuntimeError("metadata.eeg_channels is empty")
    sample_rate_hz = int(metadata.get("sample_rate_hz", 0))
    if sample_rate_hz <= 0:
        raise RuntimeError("metadata.sample_rate_hz must be > 0")

    raw = np.asarray(DataFilter.read_file(str(csv_path)), dtype=np.float64)
    if raw.ndim != 2:
        raise RuntimeError(f"unexpected raw data shape: {raw.shape}")
    eeg = raw[list(eeg_channels), :]
    return RecordingData(
        recording_dir=base,
        metadata=metadata,
        raw_matrix=raw,
        eeg_matrix=eeg,
        eeg_channels=eeg_channels,
        sample_rate_hz=sample_rate_hz,
    )
