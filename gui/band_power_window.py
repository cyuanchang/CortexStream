from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

_BAND_LABELS = ("D", "T", "A", "B", "G")
_BAND_X = np.arange(len(_BAND_LABELS), dtype=np.float64)


class BandPowerWindow(QtWidgets.QWidget):
    """Secondary window showing per-channel normalized EEG band power histograms."""

    closed_by_user = QtCore.Signal()

    def __init__(self, channel_count: int = 16):
        super().__init__()
        self._channel_count = channel_count
        self.setWindowTitle("Band Power (per channel, normalized)")
        self.resize(1100, 900)

        layout = QtWidgets.QVBoxLayout()
        self._plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self._plot_widget)
        self.setLayout(layout)

        self._bar_items: List[pg.BarGraphItem] = []
        for idx in range(self._channel_count):
            row = idx // 4
            col = idx % 4
            plot = self._plot_widget.addPlot(row=row, col=col)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.setTitle(f"ch{idx + 1}")
            plot.setLabel("bottom", "band")
            plot.setLabel("left", "norm")
            plot.setYRange(0.0, 1.0, padding=0.0)
            plot.setXRange(-0.5, len(_BAND_LABELS) - 0.5, padding=0.0)
            plot.getAxis("bottom").setTicks([[(i, label) for i, label in enumerate(_BAND_LABELS)]])
            bar = pg.BarGraphItem(
                x=_BAND_X,
                height=np.zeros(len(_BAND_LABELS)),
                width=0.65,
                brushes=[pg.intColor(i, hues=len(_BAND_LABELS)) for i in range(len(_BAND_LABELS))],
            )
            plot.addItem(bar)
            self._bar_items.append(bar)

    def update_powers(self, normalized_powers: np.ndarray) -> None:
        """Update 16 channel histograms from [n_channels, 5] normalized powers."""
        matrix = np.asarray(normalized_powers, dtype=np.float64)
        count = min(self._channel_count, matrix.shape[0], len(self._bar_items))
        for idx in range(count):
            heights = matrix[idx, : len(_BAND_LABELS)]
            self._bar_items[idx].setOpts(x=_BAND_X, height=heights)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[name-defined]
        self.closed_by_user.emit()
        super().closeEvent(event)
