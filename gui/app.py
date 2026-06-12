import argparse
import sys
import time
from pathlib import Path
from queue import Empty
from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from Realtime_processing.headmap import IDWHeadmapModel, build_idw_headmap_model
from Realtime_processing.montage import ChannelPosition, get_default_16ch_positions
from Realtime_processing.gui_preprocessing import (
    compute_gui_fft_amplitude,
    run_gui_filter_pipeline,
    smooth_fft_amplitude_openbci,
)
from streaming.board_client import BrainFlowStreamService
from streaming.enums import BoardIdsEnum, StreamFloat, StreamNumeric
from streaming.recorder import RawFrameRecorder
from streaming.types import StreamConfig


class StreamWindow(QtWidgets.QWidget):
    def __init__(self, serial_port: str, recordings_dir: str):
        super().__init__()
        self.setWindowTitle("BCI Phase2.1 Preprocessing Monitor")
        self._serial_port = serial_port
        self._recordings_dir = recordings_dir
        self._service: BrainFlowStreamService | None = None
        self._recorder: RawFrameRecorder | None = None
        self._sample_rate = 125
        self._channel_count = 16
        self._window_samples = self._sample_rate * int(StreamNumeric.GUI_WINDOW_SECONDS)
        self._processing_samples = self._sample_rate * int(StreamNumeric.GUI_PROCESSING_BUFFER_SECONDS)
        self._raw_plot_data = np.zeros((self._channel_count, self._processing_samples))
        self._plot_data = np.zeros((self._channel_count, self._window_samples))
        self._fft_n = int(StreamNumeric.FFT_N)
        self._smoothed_fft_amplitude: np.ndarray | None = None
        self._channel_positions: List[ChannelPosition] = get_default_16ch_positions()
        self._headmap_model: IDWHeadmapModel | None = None
        self._smoothed_headmap: np.ndarray | None = None
        self._stream_started_monotonic: float | None = None
        self._samples_received = 0
        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._drain_queue)
        self._timer.start(int(StreamNumeric.GUI_REFRESH_MS))
        self._spectral_timer = QtCore.QTimer(self)
        self._spectral_timer.timeout.connect(self._update_fft_plot)
        self._spectral_timer.start(int(StreamNumeric.SPECTRAL_REFRESH_MS))

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
        self._band_label = QtWidgets.QLabel("headplot: frozen (FFT upgrade)")

        body = QtWidgets.QHBoxLayout()
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
        body.addWidget(self._plot_widget, stretch=3)

        right_panel = QtWidgets.QVBoxLayout()
        self._fft_widget = pg.PlotWidget(title="FFT (post-preprocess)")
        self._fft_widget.showGrid(x=True, y=True, alpha=0.2)
        self._fft_widget.setLogMode(x=False, y=True)
        self._fft_widget.setLabel("left", "Amplitude (uV)")
        self._fft_widget.setLabel("bottom", "frequency (Hz)")
        self._fft_widget.setXRange(0.1, 60.0, padding=0.0)
        self._fft_widget.setYRange(0.1, float(StreamFloat.FFT_DISPLAY_Y_MAX_UV), padding=0.0)
        self._fft_curves: List[pg.PlotDataItem] = []
        for idx in range(self._channel_count):
            color = pg.intColor(idx, hues=self._channel_count)
            self._fft_curves.append(self._fft_widget.plot(pen=pg.mkPen(color=color, width=1)))
        right_panel.addWidget(self._fft_widget, stretch=2)

        self._head_widget = pg.PlotWidget(title="Head Power (post-preprocess)")
        self._head_widget.setAspectLocked(True)
        self._head_widget.setMenuEnabled(False)
        self._head_widget.setMouseEnabled(x=False, y=False)
        self._head_widget.hideAxis("left")
        self._head_widget.hideAxis("bottom")
        self._head_widget.setXRange(-1.15, 1.15, padding=0.0)
        self._head_widget.setYRange(-1.15, 1.15, padding=0.0)
        self._init_headplot()
        right_panel.addWidget(self._head_widget, stretch=2)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        body.addWidget(right_container, stretch=2)

        root = QtWidgets.QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self._health_label)
        root.addWidget(self._recorder_label)
        root.addWidget(self._band_label)
        root.addLayout(body)
        self.setLayout(root)

    def _connect(self) -> None:
        config = StreamConfig(serial_port=self._serial_port, board_id=int(BoardIdsEnum.CYTON_DAISY))
        self._service = BrainFlowStreamService(config)
        self._service.connect()
        self._sample_rate = self._service.get_sample_rate_hz()
        self._processing_samples = self._sample_rate * int(StreamNumeric.GUI_PROCESSING_BUFFER_SECONDS)
        self._window_samples = self._sample_rate * int(StreamNumeric.GUI_WINDOW_SECONDS)
        self._raw_plot_data = np.zeros((self._channel_count, self._processing_samples))
        self._plot_data = np.zeros((self._channel_count, self._window_samples))
        self._smoothed_fft_amplitude = None
        self._smoothed_headmap = None
        self._samples_received = 0
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
        self._smoothed_fft_amplitude = None
        self._smoothed_headmap = None
        self._samples_received = 0
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
        self._smoothed_fft_amplitude = None
        self._smoothed_headmap = None
        self._samples_received = 0
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
        self._smoothed_fft_amplitude = None
        self._smoothed_headmap = None
        self._samples_received = 0
        self._service = None

    def _drain_queue(self) -> None:
        if self._service is None:
            return
        chunk_queue = self._service.get_queue()
        drained = 0
        for _ in range(64):
            try:
                chunk = chunk_queue.get_nowait()
            except Empty:
                break
            self._append_raw_chunk(chunk.eeg_data)
            drained += 1
        if drained == 0:
            self._refresh_status()
            return

        filtered_processing = run_gui_filter_pipeline(
            self._raw_plot_data,
            self._sample_rate,
            float(StreamFloat.BANDPASS_LOW_HZ),
            float(StreamFloat.BANDPASS_HIGH_HZ),
            float(StreamFloat.NOTCH_HZ),
        )
        self._plot_data = filtered_processing[:, -self._window_samples :]
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

    def _append_raw_chunk(self, chunk: np.ndarray) -> None:
        chunk_len = chunk.shape[1]
        self._raw_plot_data[:, :-chunk_len] = self._raw_plot_data[:, chunk_len:]
        self._raw_plot_data[:, -chunk_len:] = chunk
        self._samples_received += chunk_len

    def _update_fft_plot(self) -> None:
        if self._service is None or self._stream_started_monotonic is None:
            return
        if self._samples_received < self._processing_samples:
            return
        if self._raw_plot_data.shape[1] < self._fft_n:
            return

        filtered = run_gui_filter_pipeline(
            self._raw_plot_data,
            self._sample_rate,
            float(StreamFloat.BANDPASS_LOW_HZ),
            float(StreamFloat.BANDPASS_HIGH_HZ),
            float(StreamFloat.NOTCH_HZ),
        )
        freqs, amplitude = compute_gui_fft_amplitude(filtered, self._sample_rate, self._fft_n)
        self._smoothed_fft_amplitude = smooth_fft_amplitude_openbci(
            amplitude,
            self._smoothed_fft_amplitude,
            float(StreamFloat.FFT_SMOOTHING_ALPHA),
            float(StreamFloat.FFT_MIN_AMPLITUDE_UV),
        )

        max_freq = 60.0
        freq_mask = freqs <= max_freq
        for idx, curve in enumerate(self._fft_curves):
            curve.setData(freqs[freq_mask], self._smoothed_fft_amplitude[idx, freq_mask])

    def _init_headplot(self) -> None:
        self._headmap_model = build_idw_headmap_model(
            self._channel_positions,
            int(StreamNumeric.HEADMAP_GRID_SIZE),
            float(StreamFloat.HEADMAP_IDW_POWER),
            float(StreamFloat.HEADMAP_IDW_EPS),
        )
        self._head_colormap = pg.colormap.get("viridis")
        self._head_lut = self._head_colormap.getLookupTable(0.0, 1.0, 256, alpha=True)
        self._head_image = pg.ImageItem()
        self._head_image.setRect(QtCore.QRectF(-1.0, -1.0, 2.0, 2.0))
        self._head_widget.addItem(self._head_image)

        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        self._head_widget.plot(np.cos(theta), np.sin(theta), pen=pg.mkPen(width=2))
        self._head_widget.plot([0.0, -0.08], [1.06, 0.95], pen=pg.mkPen(width=2))
        self._head_widget.plot([0.0, 0.08], [1.06, 0.95], pen=pg.mkPen(width=2))
        self._head_widget.plot([-1.0, -1.0], [0.15, -0.15], pen=pg.mkPen(width=2))
        self._head_widget.plot([1.0, 1.0], [0.15, -0.15], pen=pg.mkPen(width=2))
        self._update_headplot(np.zeros(len(self._channel_positions), dtype=np.float64))

    def _update_headplot(self, band_power: np.ndarray) -> None:
        if self._headmap_model is None:
            return
        count = min(len(self._channel_positions), len(band_power), self._channel_count)
        values = np.asarray(band_power[:count], dtype=np.float64)
        if values.size == 0 or count == 0:
            return
        head_raw = self._headmap_model.interpolate(values)
        alpha = float(StreamFloat.HEADMAP_SMOOTHING_ALPHA)
        if self._smoothed_headmap is None:
            self._smoothed_headmap = head_raw
        else:
            self._smoothed_headmap = alpha * self._smoothed_headmap + (1.0 - alpha) * head_raw

        masked_values = self._smoothed_headmap[self._headmap_model.mask]
        if masked_values.size == 0:
            return
        vmin = float(np.min(masked_values))
        vmax = float(np.max(masked_values))
        if vmax - vmin < 1e-9:
            norm = np.full(self._smoothed_headmap.shape, 0.5, dtype=np.float64)
        else:
            norm = (self._smoothed_headmap - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0.0, 1.0)

        indices = (norm * 255.0).astype(np.uint8)
        rgba = self._head_lut[indices]
        rgba[..., 3] = np.where(self._headmap_model.mask, 255, 0).astype(np.uint8)
        self._head_image.setImage(np.flipud(rgba), autoLevels=False)

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
        if self._stream_started_monotonic is not None and self._samples_received < self._processing_samples:
            remaining = self._processing_samples - self._samples_received
            warmup_s = remaining / max(self._sample_rate, 1)
            self._status_label.setText(f"state: streaming (warming_up {warmup_s:.1f}s)")
        elif self._stream_started_monotonic is not None:
            self._status_label.setText("state: streaming")


def run_gui(serial_port: str, recordings_dir: str) -> None:
    """Launch real-time preprocessing GUI for 16-channel EEG monitoring."""
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    window = StreamWindow(serial_port=serial_port, recordings_dir=recordings_dir)
    window.resize(1500, 850)
    window.show()
    app.exec()


def main() -> None:
    """Run GUI entrypoint with serial port and recording path args."""
    parser = argparse.ArgumentParser(description="Phase 2.1 BCI preprocessing GUI")
    parser.add_argument("--serial-port", required=True, type=str)
    parser.add_argument("--recordings-dir", default="recordings", type=str)
    args = parser.parse_args()
    run_gui(serial_port=args.serial_port, recordings_dir=args.recordings_dir)


if __name__ == "__main__":
    main()
