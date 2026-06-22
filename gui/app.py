import argparse
import sys
import time
from queue import Empty
from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from Realtime_processing.headmap import OpenBCIHeadmapModel, build_openbci_headmap_model
from Realtime_processing.headplot_render import render_head_image, scalp_rgba_for_display
from Realtime_processing.pipeline_runtime import (
    DisplaySnapshot,
    build_display_snapshot,
    build_spectral_snapshot,
)
from Realtime_processing.montage import (
    ChannelPosition,
    get_default_16ch_positions,
    get_reference_position,
)
from decoder.config import DecoderTaskType, DeploymentMode
from gui.band_power_window import BandPowerWindow
from gui.ssvep_training_panel import SSVEPTrainingPanel
from gui.stream_controller import StreamController
from streaming.enums import StreamFloat, StreamNumeric


class StreamWindow(QtWidgets.QWidget):
    _HEAD_Z_IMAGE = 0
    _HEAD_Z_OUTLINE = 10
    _HEAD_XY_RANGE = (-1.15, 1.15)
    _HEAD_Z_SCATTER = 20
    _HEAD_Z_LABEL = 30

    def __init__(
        self,
        serial_port: str,
        recordings_dir: str,
        decoder_task_type: DecoderTaskType = "ssvep",
        decoder_model_path: str = "",
        decoder_manifest_path: str = "",
        decoder_deployment_mode: DeploymentMode = "dev",
    ):
        super().__init__()
        self.setWindowTitle("EEG Realtime Procesing GUI")
        self._serial_port = serial_port
        self._recordings_dir = recordings_dir
        self._decoder_task_type = decoder_task_type
        self._stream_controller = StreamController(
            serial_port=serial_port,
            recordings_dir=recordings_dir,
            decoder_task_type=decoder_task_type,
            decoder_model_path=decoder_model_path,
            decoder_manifest_path=decoder_manifest_path,
            decoder_deployment_mode=decoder_deployment_mode,
        )
        self._ssvep_panel: SSVEPTrainingPanel | None = None
        self._sample_rate = 125
        self._channel_count = 16
        self._window_samples = self._sample_rate * int(StreamNumeric.GUI_WINDOW_SECONDS)
        self._processing_samples = self._sample_rate * int(StreamNumeric.GUI_PROCESSING_BUFFER_SECONDS)
        self._raw_plot_data = np.zeros((self._channel_count, self._processing_samples))
        self._plot_data = np.zeros((self._channel_count, self._window_samples))
        self._fft_n = int(StreamNumeric.FFT_N)
        self._smoothed_fft_amplitude: np.ndarray | None = None
        self._channel_positions: List[ChannelPosition] = get_default_16ch_positions(
            float(StreamFloat.ELECTRODE_REL_DIAM),
            float(StreamFloat.MONTAGE_LAYOUT_SCALE),
        )
        self._headmap_model: OpenBCIHeadmapModel | None = None
        self._smoothed_pixel_voltage: np.ndarray | None = None
        self._electrode_scatter: pg.ScatterPlotItem | None = None
        self._electrode_labels: List[pg.TextItem] = []
        self._head_outline_items: List[pg.PlotDataItem] = []
        self._head_image_rect_initialized = False
        self._normalized_band_powers: np.ndarray | None = None
        self._latest_display_snapshot: DisplaySnapshot | None = None
        self._band_power_window: BandPowerWindow | None = None
        self._band_edges = self._default_band_edges()
        self._samples_received = 0
        self._build_ui()
        self._init_optional_ssvep_panel()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._drain_queue)
        self._timer.start(int(StreamNumeric.GUI_REFRESH_MS))
        self._spectral_timer = QtCore.QTimer(self)
        self._spectral_timer.timeout.connect(self._update_spectral_views)
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
        self._band_power_button = QtWidgets.QPushButton("Show band power")
        self._band_power_button.clicked.connect(self._toggle_band_power_window)

        controls.addWidget(self._port_label)
        controls.addWidget(self._status_label)
        controls.addWidget(self._autoscale_checkbox)
        controls.addWidget(self._connect_button)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addWidget(self._disconnect_button)
        controls.addWidget(self._band_power_button)

        self._health_label = QtWidgets.QLabel("chunks: 0 | dropped: 0 | queue: 0 | uptime_s: 0.0")
        self._recorder_label = QtWidgets.QLabel("recorder: idle")
        self._decoder_label = QtWidgets.QLabel("decoder: idle")
        self._ssvep_label = QtWidgets.QLabel("ssvep panel: inactive")
        self._band_label = QtWidgets.QLabel("spectral: head std + band power @ 40ms")

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
        fft_view = self._fft_widget.getViewBox()
        fft_view.enableAutoRange(x=False, y=True)
        self._fft_curves: List[pg.PlotDataItem] = []
        for idx in range(self._channel_count):
            color = pg.intColor(idx, hues=self._channel_count)
            self._fft_curves.append(self._fft_widget.plot(pen=pg.mkPen(color=color, width=1)))
        right_panel.addWidget(self._fft_widget, stretch=2)

        self._head_widget = pg.PlotWidget(title="Head Plot (OpenBCI std + polarity)")
        self._head_widget.setAspectLocked(True)
        self._head_widget.setMenuEnabled(False)
        self._head_widget.setMouseEnabled(x=False, y=False)
        self._head_widget.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        self._head_widget.hideAxis("left")
        self._head_widget.hideAxis("bottom")
        self._head_widget.hideAxis("top")
        self._head_widget.hideAxis("right")
        # Keep headplot in a fixed head-centric coordinate frame.
        self._head_widget.hideButtons()
        self._set_headplot_view_range()
        self._init_headplot()
        right_panel.addWidget(self._head_widget, stretch=2)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        body.addWidget(right_container, stretch=2)

        root = QtWidgets.QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self._health_label)
        root.addWidget(self._recorder_label)
        root.addWidget(self._decoder_label)
        root.addWidget(self._ssvep_label)
        root.addWidget(self._band_label)
        root.addLayout(body)
        self.setLayout(root)

    def _init_optional_ssvep_panel(self) -> None:
        if self._decoder_task_type != "ssvep":
            self._ssvep_label.setText("ssvep panel: inactive (task=mi)")
            return
        self._ssvep_panel = SSVEPTrainingPanel(
            marker_writer=self._stream_controller.append_marker_event,
            recording_active_getter=self._stream_controller.is_recording_active,
            session_id_getter=self._stream_controller.get_active_session_id,
            stream_start_getter=lambda: self._stream_controller.stream_started_monotonic,
        )
        self._ssvep_panel.show()
        self._ssvep_label.setText("ssvep panel: active (task=ssvep)")

    @staticmethod
    def _default_band_edges() -> List[tuple[float, float]]:
        return [
            (float(StreamFloat.BAND_DELTA_LOW_HZ), float(StreamFloat.BAND_DELTA_HIGH_HZ)),
            (float(StreamFloat.BAND_THETA_LOW_HZ), float(StreamFloat.BAND_THETA_HIGH_HZ)),
            (float(StreamFloat.BAND_ALPHA_LOW_HZ), float(StreamFloat.BAND_ALPHA_HIGH_HZ)),
            (float(StreamFloat.BAND_BETA_LOW_HZ), float(StreamFloat.BAND_BETA_HIGH_HZ)),
            (float(StreamFloat.BAND_GAMMA_LOW_HZ), float(StreamFloat.BAND_GAMMA_HIGH_HZ)),
        ]

    def _toggle_band_power_window(self) -> None:
        if self._band_power_window is None:
            self._band_power_window = BandPowerWindow(channel_count=self._channel_count)
            self._band_power_window.closed_by_user.connect(self._on_band_power_window_closed)
        self._band_power_window.show()
        self._band_power_window.raise_()
        self._band_power_button.setText("Band power open")
        if self._normalized_band_powers is not None:
            self._band_power_window.update_powers(self._normalized_band_powers)

    def _on_band_power_window_closed(self) -> None:
        self._band_power_button.setText("Show band power")

    def _connect(self) -> None:
        service = self._stream_controller.connect()
        self._sample_rate = service.get_sample_rate_hz()
        self._processing_samples = self._sample_rate * int(StreamNumeric.GUI_PROCESSING_BUFFER_SECONDS)
        self._window_samples = self._sample_rate * int(StreamNumeric.GUI_WINDOW_SECONDS)
        self._raw_plot_data = np.zeros((self._channel_count, self._processing_samples))
        self._plot_data = np.zeros((self._channel_count, self._window_samples))
        self._reset_runtime_state()
        self._status_label.setText("state: connected")
        self._connect_button.setEnabled(False)
        self._start_button.setEnabled(True)
        self._disconnect_button.setEnabled(True)

    def _start_stream(self) -> None:
        service = self._stream_controller.get_service()
        if service is None:
            return
        try:
            session_dir = self._stream_controller.start()
        except Exception as exc:
            self._status_label.setText(f"state: start_failed ({exc})")
            return
        self._reset_runtime_state()
        self._status_label.setText("state: streaming")
        self._recorder_label.setText(f"recorder: {session_dir}")
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)

    def _stop_stream(self) -> None:
        service = self._stream_controller.get_service()
        if service is None:
            return
        frames_written = self._stream_controller.stop()
        self._recorder_label.setText(f"recorder frames: {frames_written}")
        self._reset_runtime_state()
        self._status_label.setText("state: connected")
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)

    def _disconnect(self) -> None:
        service = self._stream_controller.get_service()
        if service is None:
            return
        self._stream_controller.disconnect()
        self._status_label.setText("state: disconnected")
        self._connect_button.setEnabled(True)
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._disconnect_button.setEnabled(False)
        self._recorder_label.setText("recorder: idle")
        self._decoder_label.setText("decoder: idle")
        self._reset_runtime_state()
 
    def _reset_runtime_state(self) -> None:
        self._smoothed_fft_amplitude = None
        self._smoothed_pixel_voltage = None
        self._normalized_band_powers = None
        self._latest_display_snapshot = None
        self._samples_received = 0

    def _drain_queue(self) -> None:
        chunk_queue = self._stream_controller.get_chunk_queue()
        if chunk_queue is None:
            return
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

        display_snapshot = build_display_snapshot(
            self._raw_plot_data,
            self._sample_rate,
            self._window_samples,
            float(StreamFloat.BANDPASS_LOW_HZ),
            float(StreamFloat.BANDPASS_HIGH_HZ),
            float(StreamFloat.NOTCH_HZ),
        )
        self._latest_display_snapshot = display_snapshot
        self._plot_data = display_snapshot.window_data
        dt = 1.0 / max(self._sample_rate, 1)
        elapsed_now = 0.0
        started = self._stream_controller.stream_started_monotonic
        if started is not None:
            elapsed_now = max(time.monotonic() - started, 0.0)
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

    def _update_spectral_views(self) -> None:
        if self._stream_controller.get_service() is None or self._stream_controller.stream_started_monotonic is None:
            return
        if self._samples_received < self._processing_samples:
            return
        if self._latest_display_snapshot is None:
            return
        if self._latest_display_snapshot.filtered_processing.shape[1] < self._fft_n:
            return

        spectral_snapshot = build_spectral_snapshot(
            self._latest_display_snapshot.filtered_processing,
            self._sample_rate,
            self._fft_n,
            self._smoothed_fft_amplitude,
            float(StreamFloat.FFT_SMOOTHING_ALPHA),
            float(StreamFloat.FFT_MIN_AMPLITUDE_UV),
            int(StreamNumeric.HEAD_INTENSITY_WINDOW_SECONDS),
            self._band_edges,
            float(StreamFloat.BAND_POWER_EPS),
        )
        self._smoothed_fft_amplitude = spectral_snapshot.smoothed_fft_amplitude

        max_freq = 60.0
        freq_mask = spectral_snapshot.freqs <= max_freq
        for idx, curve in enumerate(self._fft_curves):
            curve.setData(
                spectral_snapshot.freqs[freq_mask],
                spectral_snapshot.smoothed_fft_amplitude[idx, freq_mask],
            )

        self._update_headplot(
            spectral_snapshot.head_intensity,
            spectral_snapshot.head_polarity,
        )
        self._band_label.setText(
            f"head std max: {float(np.max(spectral_snapshot.head_intensity)):.2f} uV | "
            f"ref ch: {spectral_snapshot.head_ref_idx + 1} | "
            f"spectral @ {int(StreamNumeric.SPECTRAL_REFRESH_MS)}ms"
        )

        self._normalized_band_powers = spectral_snapshot.normalized_band_powers
        if self._band_power_window is not None and self._band_power_window.isVisible():
            self._band_power_window.update_powers(self._normalized_band_powers)

    def _init_headplot(self) -> None:
        print("Building OpenBCI headplot diffusion weights...")
        self._headmap_model = build_openbci_headmap_model(
            self._channel_positions,
            int(StreamNumeric.HEADMAP_GRID_SIZE),
            float(StreamFloat.ELECTRODE_REL_DIAM),
            int(StreamNumeric.HEADMAP_DECIMATION),
        )
        print("Headplot weights ready.")

        self._head_image = pg.ImageItem()
        self._head_image.setZValue(self._HEAD_Z_IMAGE)
        self._head_widget.addItem(self._head_image)

        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        outline_pen = pg.mkPen(width=2)
        self._head_outline_items = [
            self._head_widget.plot(np.cos(theta), np.sin(theta), pen=outline_pen),
            self._head_widget.plot([0.0, -0.08], [1.06, 0.95], pen=outline_pen),
            self._head_widget.plot([0.0, 0.08], [1.06, 0.95], pen=outline_pen),
            self._head_widget.plot([-1.0, -1.0], [0.15, -0.15], pen=outline_pen),
            self._head_widget.plot([1.0, 1.0], [0.15, -0.15], pen=outline_pen),
        ]
        for item in self._head_outline_items:
            item.setZValue(self._HEAD_Z_OUTLINE)

        ring_pen = pg.mkPen(color=(40, 40, 40), width=1.5)
        elec_diam = float(StreamFloat.ELECTRODE_REL_DIAM) * 2.0
        spots = []
        for pos in self._channel_positions:
            spots.append(
                {
                    "pos": (pos.x, pos.y),
                    "size": elec_diam,
                    "pen": ring_pen,
                    "brush": pg.mkBrush(0, 0, 0, 0),
                }
            )
        self._electrode_scatter = pg.ScatterPlotItem(
            size=elec_diam,
            symbol="o",
            pxMode=False,
        )
        self._electrode_scatter.addPoints(spots)
        self._electrode_scatter.setZValue(self._HEAD_Z_SCATTER)
        self._head_widget.addItem(self._electrode_scatter)

        label_font = QtGui.QFont()
        label_font.setPointSize(10)
        label_font.setBold(True)
        label_color = pg.mkColor(30, 50, 100)
        for pos in self._channel_positions:
            label = pg.TextItem(
                pos.display_label or pos.name,
                anchor=(0.5, 0.5),
                color=label_color,
            )
            label.setFont(label_font)
            label.setPos(pos.x, pos.y)
            label.setZValue(self._HEAD_Z_LABEL)
            self._head_widget.addItem(label)
            self._electrode_labels.append(label)

        ref_x, ref_y = get_reference_position()
        ref_label = pg.TextItem("R", anchor=(0.5, 0.5), color=label_color)
        ref_label.setFont(label_font)
        ref_label.setPos(ref_x, ref_y)
        ref_label.setZValue(self._HEAD_Z_LABEL)
        self._head_widget.addItem(ref_label)
        self._electrode_labels.append(ref_label)

        self._update_headplot(
            np.zeros(len(self._channel_positions), dtype=np.float64),
            np.ones(len(self._channel_positions), dtype=np.float64),
        )

    def _update_headplot(self, intensity: np.ndarray, polarity: np.ndarray) -> None:
        if self._headmap_model is None:
            return
        count = min(len(self._channel_positions), len(intensity), len(polarity), self._channel_count)
        if count == 0:
            return

        intense_min = float(StreamFloat.HEAD_INTENSITY_MIN_UV)
        intense_max = float(StreamFloat.HEAD_INTENSITY_MAX_UV)
        intensity_uv = np.clip(np.asarray(intensity[:count], dtype=np.float64), intense_min, intense_max)
        polarity_vec = np.asarray(polarity[:count], dtype=np.float64)
        signed = intensity_uv * polarity_vec

        head_raw = self._headmap_model.interpolate_voltage(signed)
        alpha = float(StreamFloat.HEADMAP_SMOOTHING_ALPHA)
        if self._smoothed_pixel_voltage is None:
            self._smoothed_pixel_voltage = head_raw.copy()
        else:
            valid = head_raw >= 0.0
            self._smoothed_pixel_voltage[valid] = (
                alpha * self._smoothed_pixel_voltage[valid] + (1.0 - alpha) * head_raw[valid]
            )
            invalid = ~valid
            self._smoothed_pixel_voltage[invalid] = head_raw[invalid]

        rgba = render_head_image(
            self._smoothed_pixel_voltage,
            self._headmap_model.mask,
            intense_min,
            intense_max,
            contour_levels=int(StreamNumeric.HEAD_CONTOUR_LEVELS),
        )
        display = scalp_rgba_for_display(rgba)
        self._head_image.setImage(display, autoLevels=False)
        # Important: setRect must happen after first setImage so width/height are known.
        if not self._head_image_rect_initialized:
            self._head_image.setRect(QtCore.QRectF(-1.0, -1.0, 2.0, 2.0))
            self._head_image_rect_initialized = True
        self._set_headplot_view_range()
        self._apply_headplot_z_order()

    def _apply_headplot_z_order(self) -> None:
        self._head_image.setZValue(self._HEAD_Z_IMAGE)
        for item in self._head_outline_items:
            item.setZValue(self._HEAD_Z_OUTLINE)
        if self._electrode_scatter is not None:
            self._electrode_scatter.setZValue(self._HEAD_Z_SCATTER)
        for label in self._electrode_labels:
            label.setZValue(self._HEAD_Z_LABEL)

    def _set_headplot_view_range(self) -> None:
        """Pin headplot view to normalized head coordinates; avoid accidental auto-range."""
        view_box = self._head_widget.getViewBox()
        view_box.enableAutoRange(x=False, y=False)
        lo, hi = self._HEAD_XY_RANGE
        self._head_widget.setXRange(lo, hi, padding=0.0)
        self._head_widget.setYRange(lo, hi, padding=0.0)

    def _refresh_status(self) -> None:
        status = self._stream_controller.get_status()
        if status is None:
            return
        uptime = 0.0
        if status.started_monotonic > 0:
            uptime = time.monotonic() - status.started_monotonic
        self._health_label.setText(
            "chunks: "
            f"{status.produced_chunks} | dropped: {status.dropped_chunks} | "
            f"queue: {status.queue_size} | uptime_s: {uptime:.1f}"
        )
        if self._stream_controller.stream_started_monotonic is not None and self._samples_received < self._processing_samples:
            remaining = self._processing_samples - self._samples_received
            warmup_s = remaining / max(self._sample_rate, 1)
            self._status_label.setText(f"state: streaming (warming_up {warmup_s:.1f}s)")
        elif self._stream_controller.stream_started_monotonic is not None:
            self._status_label.setText("state: streaming")
        self._refresh_decoder_status()

    def _refresh_decoder_status(self) -> None:
        decoder_status = self._stream_controller.get_decoder_runtime().get_status()
        if not decoder_status.running:
            preflight = self._stream_controller.get_decoder_preflight()
            if preflight is None:
                self._decoder_label.setText("decoder: idle")
                return
            if preflight.ok:
                warn = f" | warn={'; '.join(preflight.warnings)}" if preflight.warnings else ""
                self._decoder_label.setText(
                    f"decoder: preflight_ok | mode={preflight.backend_mode}{warn}"
                )
            else:
                self._decoder_label.setText(
                    "decoder: preflight_failed | err=" + "; ".join(preflight.errors)
                )
            return
        self._decoder_label.setText(
            "decoder: "
            f"{decoder_status.backend_name} | "
            f"mode={decoder_status.backend_mode} | "
            f"loaded={int(decoder_status.backend_loaded)} | "
            f"label={decoder_status.last_label or '-'} | "
            f"conf={decoder_status.last_confidence:.3f} | "
            f"chunks={decoder_status.chunks_received} | "
            f"infer={decoder_status.inference_count} | "
            f"queue={decoder_status.queue_size} | "
            f"dropped={decoder_status.dropped_by_bus} | "
            f"last_seq={decoder_status.last_sequence_id} | "
            f"fail={decoder_status.failures}"
        )
        if decoder_status.last_error:
            self._decoder_label.setText(f"{self._decoder_label.text()} | err={decoder_status.last_error}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._ssvep_panel is not None:
            self._ssvep_panel.close()
            self._ssvep_panel = None
        super().closeEvent(event)


def run_gui(
    serial_port: str,
    recordings_dir: str,
    decoder_task_type: DecoderTaskType = "ssvep",
    decoder_model_path: str = "",
    decoder_manifest_path: str = "",
    decoder_deployment_mode: DeploymentMode = "dev",
) -> None:
    """Launch real-time preprocessing GUI for 16-channel EEG monitoring."""
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    window = StreamWindow(
        serial_port=serial_port,
        recordings_dir=recordings_dir,
        decoder_task_type=decoder_task_type,
        decoder_model_path=decoder_model_path,
        decoder_manifest_path=decoder_manifest_path,
        decoder_deployment_mode=decoder_deployment_mode,
    )
    window.resize(1500, 850)
    window.show()
    app.exec()


def main() -> None:
    """Run GUI entrypoint with serial port and recording path args."""
    parser = argparse.ArgumentParser(description="EEG Realtime Procesing GUI")
    parser.add_argument("--serial-port", required=True, type=str)
    parser.add_argument("--recordings-dir", default="recordings", type=str)
    parser.add_argument("--decoder-task-type", default="ssvep", choices=("ssvep", "mi"))
    parser.add_argument("--decoder-model-path", default="", type=str)
    parser.add_argument("--decoder-manifest", default="", type=str)
    parser.add_argument("--decoder-deployment-mode", default="dev", choices=("dev", "live"))
    args = parser.parse_args()
    run_gui(
        serial_port=args.serial_port,
        recordings_dir=args.recordings_dir,
        decoder_task_type=args.decoder_task_type,
        decoder_model_path=args.decoder_model_path,
        decoder_manifest_path=args.decoder_manifest,
        decoder_deployment_mode=args.decoder_deployment_mode,
    )


if __name__ == "__main__":
    main()
