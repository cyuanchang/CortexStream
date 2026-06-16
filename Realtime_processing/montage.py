from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ChannelPosition:
    name: str
    x: float
    y: float
    display_label: str = ""


def _openbci_default_rel_xy(elec_rel_diam: float) -> List[Tuple[float, float]]:
    """OpenBCI createDefaultElectrodeLocations relative coords (Y: +posterior)."""
    elec: List[Tuple[float, float]] = [(0.0, 0.0)] * 16

    elec[0] = (-0.125, -0.5 + elec_rel_diam * (0.5 + 0.2))
    elec[1] = (-elec[0][0], elec[0][1])

    elec[2] = (-0.2, 0.0)
    elec[3] = (-elec[2][0], elec[2][1])

    elec[4] = (-0.3425, 0.27)
    elec[5] = (-elec[4][0], elec[4][1])

    elec[6] = (-0.125, +0.5 - elec_rel_diam * (0.5 + 0.2))
    elec[7] = (-elec[6][0], elec[6][1])

    elec[8] = (elec[4][0], -elec[4][1])
    elec[9] = (-elec[8][0], elec[8][1])

    elec[10] = (-0.18, -0.15)
    elec[11] = (-elec[10][0], elec[10][1])

    elec[12] = (-0.5 + elec_rel_diam * (0.5 + 0.15), 0.0)
    elec[13] = (-elec[12][0], elec[12][1])

    elec[14] = (elec[10][0], -elec[10][1])
    elec[15] = (-elec[14][0], elec[14][1])

    return elec


def get_default_16ch_positions(
    elec_rel_diam: float = 0.12,
    layout_scale: float = 2.0,
) -> List[ChannelPosition]:
    """Return OpenBCI default 16-channel layout; Y flipped for plot coords (+Y = anterior).

    OpenBCI rel coords are fractions of head diameter (scalp edge at +/-0.5). The GUI head
    circle uses radius 1.0, so layout_scale=2.0 maps those coords onto the full scalp.
    """
    rel = _openbci_default_rel_xy(elec_rel_diam)
    positions: List[ChannelPosition] = []
    for idx, (x, y_obci) in enumerate(rel):
        label = str(idx + 1)
        positions.append(
            ChannelPosition(
                name=f"ch{idx + 1}",
                x=float(x) * layout_scale,
                y=float(-y_obci) * layout_scale,
                display_label=label,
            )
        )
    return positions


def get_reference_position() -> Tuple[float, float]:
    """Reference electrode at head center (OpenBCI 'R' marker)."""
    return (0.0, 0.0)
