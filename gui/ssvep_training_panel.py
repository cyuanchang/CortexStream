from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets


class SSVEPStimulusSeconds(float, Enum):
    TRIAL_DURATION = 4.0
    INTER_TRIAL_INTERVAL = 2.0


class SSVEPVisual(IntEnum):
    CIRCLE_DIAMETER = 220


class SSVEPFrameLock(float, Enum):
    MAX_FREQ_ERROR_HZ = 0.01


@dataclass(frozen=True)
class SSVEPFrequencySpec:
    requested_hz: float
    realized_hz: float
    half_cycle_frames: int


@dataclass(frozen=True)
class SSVEPPlan:
    frequencies_hz: tuple[float, ...]
    total_trials: int
    randomized_trials: tuple[SSVEPFrequencySpec, ...]
    seed: int
    refresh_hz: float


class BlinkingCircleWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._is_on = False
        self._on_color = QtGui.QColor("#ffffff")
        self._off_color = QtGui.QColor("#111111")
        self.setMinimumSize(500, 400)

    def set_on(self, is_on: bool) -> None:
        self._is_on = is_on
        self.update()

    def set_palette(self, on_hex: str, off_hex: str) -> None:
        self._on_color = QtGui.QColor(on_hex)
        self._off_color = QtGui.QColor(off_hex)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        _ = event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#000000"))
        brush = self._on_color if self._is_on else self._off_color
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(brush))
        center = self.rect().center()
        radius = int(SSVEPVisual.CIRCLE_DIAMETER // 2)
        painter.drawEllipse(center, radius, radius)


class SSVEPTrainingPanel(QtWidgets.QWidget):
    def __init__(
        self,
        marker_writer: Callable[[dict], bool],
        recording_active_getter: Callable[[], bool],
        session_id_getter: Callable[[], str],
        stream_start_getter: Callable[[], float | None],
    ):
        super().__init__()
        self.setWindowTitle("SSVEP Participant View")
        self._marker_writer = marker_writer
        self._recording_active_getter = recording_active_getter
        self._session_id_getter = session_id_getter
        self._stream_start_getter = stream_start_getter

        self._plan: SSVEPPlan | None = None
        self._trial_index = 0
        self._phase: str = "idle"
        self._circle_on = False
        self._refresh_hz = 0.0
        self._frame_interval_ms = 0
        self._trial_frame_count = 0
        self._trial_toggle_count = 0
        self._active_spec: SSVEPFrequencySpec | None = None

        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._frame_timer.timeout.connect(self._on_frame_tick)
        self._phase_timer = QtCore.QTimer(self)
        self._phase_timer.setSingleShot(True)
        self._phase_timer.timeout.connect(self._advance_phase)

        self._stack = QtWidgets.QStackedWidget()
        self._entry_page = self._build_entry_page()
        self._main_page = self._build_main_page()
        self._stack.addWidget(self._entry_page)
        self._stack.addWidget(self._main_page)

        root = QtWidgets.QVBoxLayout()
        root.addWidget(self._stack)
        self.setLayout(root)
        self.resize(700, 560)
        QtCore.QTimer.singleShot(0, self._update_refresh_info)

    def _build_entry_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        title = QtWidgets.QLabel("SSVEP Paradigm Setup")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        self._freq_entry = QtWidgets.QLineEdit("8,10,12,15")
        self._trial_entry = QtWidgets.QLineEdit("40")
        self._seed_entry = QtWidgets.QLineEdit("")
        form = QtWidgets.QFormLayout()
        form.addRow("Frequencies (Hz, comma-separated):", self._freq_entry)
        form.addRow("Total trials:", self._trial_entry)
        form.addRow("Seed (optional):", self._seed_entry)
        self._refresh_info = QtWidgets.QLabel("Display refresh: detecting...")
        self._entry_error = QtWidgets.QLabel("")
        self._entry_error.setStyleSheet("color: #c0392b;")
        proceed_btn = QtWidgets.QPushButton("Proceed to Stimulus")
        proceed_btn.clicked.connect(self._proceed_to_main)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self._refresh_info)
        layout.addWidget(self._entry_error)
        layout.addWidget(proceed_btn)
        layout.addStretch(1)
        return page

    def _build_main_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self._trial_label = QtWidgets.QLabel("Trial: - / -")
        self._freq_label = QtWidgets.QLabel("Frequency: - Hz")
        self._status_label = QtWidgets.QLabel("Status: idle")
        self._circle_widget = BlinkingCircleWidget()
        controls = QtWidgets.QHBoxLayout()
        self._start_btn = QtWidgets.QPushButton("Start Paradigm")
        self._start_btn.clicked.connect(self._start_paradigm)
        self._stop_btn = QtWidgets.QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_paradigm)
        self._debug_btn = QtWidgets.QPushButton("Debug Log: OFF")
        self._debug_btn.setCheckable(True)
        self._debug_btn.toggled.connect(self._on_debug_toggled)
        back_btn = QtWidgets.QPushButton("Back to Setup")
        back_btn.clicked.connect(self._back_to_entry)
        controls.addWidget(self._start_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(self._debug_btn)
        controls.addWidget(back_btn)
        layout.addWidget(self._trial_label)
        layout.addWidget(self._freq_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._circle_widget, stretch=1)
        layout.addLayout(controls)
        return page

    def _proceed_to_main(self) -> None:
        self._update_refresh_info()
        try:
            plan = self._build_plan()
        except Exception as exc:
            self._entry_error.setText(str(exc))
            return
        self._entry_error.setText("")
        self._plan = plan
        self._refresh_hz = plan.refresh_hz
        self._frame_interval_ms = max(int(round(1000.0 / max(self._refresh_hz, 1e-6))), 1)
        self._trial_index = 0
        self._phase = "idle"
        self._active_spec = None
        self._update_main_labels()
        self._stack.setCurrentWidget(self._main_page)

    def _back_to_entry(self) -> None:
        if self._phase in ("trial", "iti"):
            return
        self._stack.setCurrentWidget(self._entry_page)

    def _build_plan(self) -> SSVEPPlan:
        refresh_hz = self._detect_refresh_hz()
        if refresh_hz <= 0.0:
            raise ValueError("Unable to detect display refresh rate.")
        raw_freqs = [part.strip() for part in self._freq_entry.text().split(",")]
        freqs: list[float] = []
        for part in raw_freqs:
            if not part:
                continue
            value = float(part)
            if value <= 0.0:
                raise ValueError("Frequencies must be positive.")
            freqs.append(round(value, 4))
        if not freqs:
            raise ValueError("Provide at least one valid frequency.")
        freqs = sorted(set(freqs))

        total_trials = int(self._trial_entry.text().strip())
        if total_trials <= 0:
            raise ValueError("Total trials must be > 0.")
        if total_trials % len(freqs) != 0:
            raise ValueError("Total trials must be divisible by the number of frequencies.")
        specs = [self._build_frequency_spec(freq_hz=v, refresh_hz=refresh_hz) for v in freqs]

        seed_text = self._seed_entry.text().strip()
        seed = int(seed_text) if seed_text else int(time.time() * 1000) % 1_000_000_000
        repeats = total_trials // len(freqs)
        trial_list = specs * repeats
        random.Random(seed).shuffle(trial_list)
        return SSVEPPlan(
            frequencies_hz=tuple(freqs),
            total_trials=total_trials,
            randomized_trials=tuple(trial_list),
            seed=seed,
            refresh_hz=refresh_hz,
        )

    def _detect_refresh_hz(self) -> float:
        window_handle = self.windowHandle()
        screen = window_handle.screen() if window_handle is not None else self.screen()
        if screen is None:
            return 0.0
        refresh_hz = float(screen.refreshRate())
        return max(refresh_hz, 0.0)

    def _update_refresh_info(self) -> None:
        refresh_hz = self._detect_refresh_hz()
        if refresh_hz <= 0.0:
            self._refresh_info.setText("Display refresh: unavailable")
            return
        self._refresh_info.setText(f"Display refresh: {refresh_hz:.3f} Hz")

    def _build_frequency_spec(self, freq_hz: float, refresh_hz: float) -> SSVEPFrequencySpec:
        half_cycle = refresh_hz / (2.0 * freq_hz)
        half_cycle_frames = max(int(round(half_cycle)), 1)
        realized_hz = refresh_hz / (2.0 * float(half_cycle_frames))
        error_hz = abs(realized_hz - freq_hz)
        if error_hz > float(SSVEPFrameLock.MAX_FREQ_ERROR_HZ.value):
            raise ValueError(
                (
                    f"Frequency {freq_hz:.4f} Hz is not frame-representable at {refresh_hz:.3f} Hz "
                    f"(nearest {realized_hz:.4f} Hz)."
                )
            )
        return SSVEPFrequencySpec(
            requested_hz=freq_hz,
            realized_hz=realized_hz,
            half_cycle_frames=half_cycle_frames,
        )

    def _start_paradigm(self) -> None:
        if self._plan is None:
            self._status_label.setText("Status: setup required")
            return
        if not self._recording_active_getter():
            self._status_label.setText("Status: start stream recording first")
            return
        self._trial_index = 0
        self._phase = "trial"
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._emit_event(
            "session_start",
            {
                "frequencies_hz": list(self._plan.frequencies_hz),
                "total_trials": self._plan.total_trials,
                "seed": self._plan.seed,
                "trial_duration_s": float(SSVEPStimulusSeconds.TRIAL_DURATION.value),
                "iti_duration_s": float(SSVEPStimulusSeconds.INTER_TRIAL_INTERVAL.value),
                "refresh_hz": self._plan.refresh_hz,
            },
        )
        self._debug_print(
            "session_start",
            {
                "refresh_hz": self._plan.refresh_hz,
                "trial_duration_s": float(SSVEPStimulusSeconds.TRIAL_DURATION.value),
                "iti_duration_s": float(SSVEPStimulusSeconds.INTER_TRIAL_INTERVAL.value),
                "total_trials": self._plan.total_trials,
            },
        )
        self._start_trial()

    def _start_trial(self) -> None:
        assert self._plan is not None
        if self._trial_index >= len(self._plan.randomized_trials):
            self._finish_session()
            return
        self._active_spec = self._plan.randomized_trials[self._trial_index]
        self._phase = "trial"
        self._circle_on = False
        self._circle_widget.set_on(False)
        self._trial_frame_count = 0
        self._trial_toggle_count = 0
        self._update_main_labels()
        self._frame_timer.start(self._frame_interval_ms)
        self._phase_timer.start(int(round(1000.0 * float(SSVEPStimulusSeconds.TRIAL_DURATION.value))))
        self._debug_print(
            "trial_start",
            {
                "trial_index": self._trial_index,
                "requested_hz": self._active_spec.requested_hz,
                "realized_hz": self._active_spec.realized_hz,
                "half_cycle_frames": self._active_spec.half_cycle_frames,
            },
        )
        self._emit_event(
            "trial_start",
            {
                "trial_index": self._trial_index,
                "frequency_hz": self._active_spec.requested_hz,
                "requested_frequency_hz": self._active_spec.requested_hz,
                "realized_frequency_hz": self._active_spec.realized_hz,
                "half_cycle_frames": self._active_spec.half_cycle_frames,
                "refresh_hz": self._refresh_hz,
            },
        )

    def _start_iti(self) -> None:
        if self._active_spec is None:
            return
        self._phase = "iti"
        self._frame_timer.stop()
        self._circle_widget.set_on(False)
        self._update_main_labels()
        expected_toggles = self._trial_frame_count // self._active_spec.half_cycle_frames
        self._debug_print(
            "trial_end",
            {
                "trial_index": self._trial_index,
                "requested_hz": self._active_spec.requested_hz,
                "realized_hz": self._active_spec.realized_hz,
                "half_cycle_frames": self._active_spec.half_cycle_frames,
                "frames_elapsed": self._trial_frame_count,
                "toggles_actual": self._trial_toggle_count,
                "toggles_expected": expected_toggles,
                "toggle_delta": self._trial_toggle_count - expected_toggles,
            },
        )
        self._emit_event(
            "trial_end",
            {
                "trial_index": self._trial_index,
                "frequency_hz": self._active_spec.requested_hz,
                "requested_frequency_hz": self._active_spec.requested_hz,
                "realized_frequency_hz": self._active_spec.realized_hz,
                "half_cycle_frames": self._active_spec.half_cycle_frames,
                "refresh_hz": self._refresh_hz,
                "trial_frames_elapsed": self._trial_frame_count,
            },
        )
        self._emit_event(
            "iti_start",
            {
                "trial_index": self._trial_index,
                "frequency_hz": self._active_spec.requested_hz,
                "requested_frequency_hz": self._active_spec.requested_hz,
                "realized_frequency_hz": self._active_spec.realized_hz,
                "half_cycle_frames": self._active_spec.half_cycle_frames,
                "refresh_hz": self._refresh_hz,
            },
        )
        self._phase_timer.start(int(round(1000.0 * float(SSVEPStimulusSeconds.INTER_TRIAL_INTERVAL.value))))

    def _advance_phase(self) -> None:
        if self._phase == "trial":
            self._start_iti()
            return
        if self._phase == "iti":
            if self._active_spec is None:
                return
            self._emit_event(
                "iti_end",
                {
                    "trial_index": self._trial_index,
                    "frequency_hz": self._active_spec.requested_hz,
                    "requested_frequency_hz": self._active_spec.requested_hz,
                    "realized_frequency_hz": self._active_spec.realized_hz,
                    "half_cycle_frames": self._active_spec.half_cycle_frames,
                    "refresh_hz": self._refresh_hz,
                },
            )
            self._trial_index += 1
            self._start_trial()

    def _finish_session(self) -> None:
        self._frame_timer.stop()
        self._phase = "done"
        self._active_spec = None
        self._circle_widget.set_on(False)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._update_main_labels()
        self._emit_event("session_end", {"trials_completed": self._trial_index})

    def _stop_paradigm(self) -> None:
        if self._phase not in ("trial", "iti"):
            return
        self._frame_timer.stop()
        self._phase_timer.stop()
        self._phase = "idle"
        self._active_spec = None
        self._circle_widget.set_on(False)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._update_main_labels()
        self._emit_event("session_abort", {"trials_completed": self._trial_index})

    def _on_frame_tick(self) -> None:
        if self._phase != "trial" or self._active_spec is None:
            return
        self._trial_frame_count += 1
        if self._trial_frame_count % self._active_spec.half_cycle_frames == 0:
            self._circle_on = not self._circle_on
            self._trial_toggle_count += 1
            self._circle_widget.set_on(self._circle_on)

    def _on_debug_toggled(self, enabled: bool) -> None:
        self._debug_btn.setText("Debug Log: ON" if enabled else "Debug Log: OFF")
        self._debug_print("debug_toggle", {"enabled": enabled})

    def _debug_print(self, event_name: str, payload: dict) -> None:
        if not self._debug_btn.isChecked():
            return
        details = ", ".join(f"{k}={v}" for k, v in payload.items())
        print(f"[SSVEP_DEBUG] {event_name}: {details}")

    def _emit_event(self, event_name: str, payload: dict) -> None:
        now_wall = time.time()
        now_mono = time.monotonic()
        event = {
            "event": event_name,
            "wall_time_unix": now_wall,
            "wall_time_iso": datetime.fromtimestamp(now_wall, tz=timezone.utc).isoformat(),
            "monotonic_s": now_mono,
            "session_id": self._session_id_getter(),
            "stream_started_monotonic": self._stream_start_getter(),
            "frame_lock_enabled": True,
            "refresh_hz": self._refresh_hz,
            **payload,
        }
        written = self._marker_writer(event)
        if not written:
            self._status_label.setText("Status: marker write skipped (no active recording)")

    def _update_main_labels(self) -> None:
        if self._plan is None:
            self._trial_label.setText("Trial: - / -")
            self._freq_label.setText("Frequency: - Hz")
            self._status_label.setText("Status: setup required")
            return
        total = len(self._plan.randomized_trials)
        shown_idx = min(self._trial_index + 1, total)
        self._trial_label.setText(f"Trial: {shown_idx} / {total}")
        if self._trial_index < total and self._active_spec is not None:
            self._freq_label.setText(
                (
                    "Frequency: "
                    f"{self._active_spec.requested_hz:.2f} Hz "
                    f"(realized {self._active_spec.realized_hz:.3f} Hz)"
                )
            )
        elif self._trial_index < total:
            pending_spec = self._plan.randomized_trials[self._trial_index]
            self._freq_label.setText(
                (
                    "Frequency: "
                    f"{pending_spec.requested_hz:.2f} Hz "
                    f"(realized {pending_spec.realized_hz:.3f} Hz)"
                )
            )
        else:
            self._freq_label.setText("Frequency: complete")
        if self._phase == "trial":
            self._status_label.setText(
                f"Status: trial blinking | frame-lock {self._refresh_hz:.2f} Hz"
            )
        elif self._phase == "iti":
            self._status_label.setText(
                f"Status: inter-trial interval | frame-lock {self._refresh_hz:.2f} Hz"
            )
        elif self._phase == "done":
            self._status_label.setText("Status: complete")
        else:
            self._status_label.setText(
                f"Status: ready | frame-lock {self._refresh_hz:.2f} Hz"
            )
