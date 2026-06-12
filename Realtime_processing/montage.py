from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ChannelPosition:
    name: str
    x: float
    y: float


def get_default_16ch_positions() -> List[ChannelPosition]:
    """Return provisional 10-20 style positions mapped by channel index."""
    return [
        ChannelPosition("ch1", -0.35, 0.88),
        ChannelPosition("ch2", 0.35, 0.88),
        ChannelPosition("ch3", -0.78, 0.52),
        ChannelPosition("ch4", -0.35, 0.45),
        ChannelPosition("ch5", 0.00, 0.50),
        ChannelPosition("ch6", 0.35, 0.45),
        ChannelPosition("ch7", 0.78, 0.52),
        ChannelPosition("ch8", -0.96, 0.00),
        ChannelPosition("ch9", -0.42, 0.00),
        ChannelPosition("ch10", 0.00, 0.00),
        ChannelPosition("ch11", 0.42, 0.00),
        ChannelPosition("ch12", 0.96, 0.00),
        ChannelPosition("ch13", -0.35, -0.50),
        ChannelPosition("ch14", 0.00, -0.55),
        ChannelPosition("ch15", 0.35, -0.50),
        ChannelPosition("ch16", 0.00, -0.88),
    ]
