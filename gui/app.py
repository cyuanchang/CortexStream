import argparse
import sys
import time
from pathlib import Path
from queue import Empty
from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from streaming.board_client import BrainFlowStreamService
from streaming.enums import BoardIdsEnum, StreamNumeric
from streaming.recorder import RawFrameRecorder
from streaming.types import DataChunk, StreamConfig


class StreamWindow(QtWidgets.QWidget):
    def __init__(self, serial_port: str, recordings_dir: str):
        super().__init__()
        self.setWindowTitle("BCI Phase1 Stream Monitor")
        self._serial_port = serial_port
        self._recordings_dir = recordings_dir
        self._service: BrainFlowStreamService | None = None
        self._recorder: RawFrameRecorder | None = None
        self._sample_rate = 125
        self._channel_count = 16
        self._window_samples = self._sample_rate * int(StreamNumeric.GUI_WINDOW_SECONDS)
        self._plot_data = np.zeros((self._channel_count, self._window_samples))
        self._stream_started_monotonic: float | None = None
        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._drain_queue)
        self._timer.start(int(StreamNumeric.GUI_REFRESH_MS))

    def _build_ui(self) -> None:
        controls = QtWidgets.QHBoxLayout()
        self._port_label = QtWidgets.QLabel(f"serial: {self._serial_port}")
        self._status_label = QtWidgets.QLabel("state: disconnected")
        self._autoscale_checkbox = QtWidgets.QCheckBox("Autoscale")
        self._autoscale_checkbox.setChecked(True)
        self._connect_button = QtWidgets.QPushButton("Connect")
        self._start_button = QtWidgets.QPushButton("Start")
        self._stop_button = QtWidgets.QPushButton("Stop")
        self._disconnect_button = QtWidgets.QPushButton("Disconnect")
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._disconnect_button.setEnabled(False)

        self._connect_button.clicked.connect(self._connect)
        self._start_button.clicked.connect(self._start_stream)
        self._stop_button.clicked.connect(self._stop_stream)
        self._disconnect_button.clicked.connect(self._disconnect)

        controls.addWidget(self._port_label)
        controls.addWidget(self._status_label)
        controls.addWidget(self._autoscale_checkbox)
        controls.addWidget(self._connect_button)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addWidget(self._disconnect_button)

        self._health_label = QtWidgets.QLabel("chunks: 0 | dropped: 0 | queue: 0 | uptime_s: 0.0")
        self._recorder_label = QtWidgets.QLabel("recorder: idle")

        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plots: List[pg.PlotItem] = []
        self._curves: List[pg.PlotDataItem] = []
        for idx in range(self._channel_count):
            plot = self._plot_widget.addPlot(row=idx, col=0)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.setLabel("left", f"ch{idx + 1}")
            if idx < self._channel_count - 1:
                plot.hideAxis("bottom")
            else:
                plot.setLabel("bottom", "time (s)")
            curve = plot.plot(pen=pg.mkPen(width=1))
            self._plots.append(plot)
            self._curves.append(curve)

        root = QtWidgets.QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self._health_label)
        root.addWidget(self._recorder_label)
        root.addWidget(self._plot_widget)
        self.setLayout(root)

    def _connect(self) -> None:
        config = StreamConfig(serial_port=self._serial_port, board_id=int(BoardIdsEnum.CYTON_DAISY))
        self._service = BrainFlowStreamService(config)
        self._service.connect()
        self._sample_rate = self._service.get_sample_rate_hz()
        self._window_samples = self._sample_rate * int(StreamNumeric.GUI_WINDOW_SECONDS)
        self._plot_data = np.zeros((self._channel_count, self._window_samples))
        self._status_label.setText("state: connected")
        self._connect_button.setEnabled(False)
        self._start_button.setEnabled(True)
        self._disconnect_button.setEnabled(True)

    def _start_stream(self) -> None:
        if self._service is None:
            return
        recordings_dir = Path(self._recordings_dir)
        self._recorder = RawFrameRecorder(
            raw_queue=self._service.get_raw_queue(),
            output_root=str(recordings_dir),
            board_id=int(BoardIdsEnum.CYTON_DAISY),
            eeg_channels=self._service.get_eeg_channels(),
            sample_rate_hz=self._service.get_sample_rate_hz(),
        )
        session_dir = self._recorder.start()
        self._service.start()
        self._stream_started_monotonic = time.monotonic()
        self._status_label.setText("state: streaming")
        self._recorder_label.setText(f"recorder: {session_dir}")
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)

    def _stop_stream(self) -> None:
        if self._service is None:
            return
        self._service.stop()
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder_label.setText(f"recorder frames: {self._recorder.frames_written()}")
            self._recorder = None
        self._stream_started_monotonic = None
        self._status_label.setText("state: connected")
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)

    def _disconnect(self) -> None:
        if self._service is None:
            return
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None
        self._service.disconnect()
        self._status_label.setText("state: disconnected")
        self._connect_button.setEnabled(True)
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._disconnect_button.setEnabled(False)
        self._stream_started_monotonic = None
        self._service = None

    def _drain_queue(self) -> None:
        if self._service is None:
            return
        chunk_queue = self._service.get_queue()
        newest_chunk: DataChunk | None = None
        for _ in range(64):
            try:
                newest_chunk = chunk_queue.get_nowait()
            except Empty:
                break
        if newest_chunk is None:
            self._refresh_status()
            return

        chunk = newest_chunk.eeg_data
        chunk_len = chunk.shape[1]
        self._plot_data[:, :-chunk_len] = self._plot_data[:, chunk_len:]
        self._plot_data[:, -chunk_len:] = chunk
        dt = 1.0 / max(self._sample_rate, 1)
        elapsed_now = 0.0
        if self._stream_started_monotonic is not None:
            elapsed_now = max(time.monotonic() - self._stream_started_monotonic, 0.0)
        x_end = elapsed_now
        x_start = max(x_end - self._window_samples * dt, 0.0)
        x = np.linspace(x_start, x_end, self._window_samples, endpoint=False)
        for idx, curve in enumerate(self._curves):
            y = self._plot_data[idx]
            curve.setData(x=x, y=y)
            if self._autoscale_checkbox.isChecked():
                max_abs = float(np.max(np.abs(y)))
                y_range = max(max_abs * 1.2, 1e-6)
                self._plots[idx].setYRange(-y_range, y_range, padding=0.0)
            self._plots[idx].setXRange(x_start, x_end, padding=0.0)
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._service is None:
            return
        status = self._service.get_status()
        uptime = 0.0
        if status.started_monotonic > 0:
            uptime = time.monotonic() - status.started_monotonic
        self._health_label.setText(
            "chunks: "
            f"{status.produced_chunks} | dropped: {status.dropped_chunks} | "
            f"queue: {status.queue_size} | uptime_s: {uptime:.1f}"
        )


def run_gui(serial_port: str, recordings_dir: str) -> None:
    """Launch Phase 1 streaming GUI for 16-channel EEG monitoring."""
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    window = StreamWindow(serial_port=serial_port, recordings_dir=recordings_dir)
    window.resize(1200, 700)
    window.show()
    app.exec()


def main() -> None:
    """Run GUI entrypoint with serial port and recording path args."""
    parser = argparse.ArgumentParser(description="Phase 1 BCI stream GUI")
    parser.add_argument("--serial-port", required=True, type=str)
    parser.add_argument("--recordings-dir", default="recordings", type=str)
    args = parser.parse_args()
    run_gui(serial_port=args.serial_port, recordings_dir=args.recordings_dir)


if __name__ == "__main__":
    main()
