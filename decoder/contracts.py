from __future__ import annotations

from typing import Protocol

import numpy as np

from decoder.types import DecoderChunkMeta, DecoderInputSpec, DecoderOutput


class DecoderBackend(Protocol):
    """Model-agnostic backend contract used by DecoderRuntime."""

    @property
    def backend_name(self) -> str:
        ...

    @property
    def backend_mode(self) -> str:
        ...

    @property
    def last_load_error(self) -> str:
        ...

    def required_input_spec(
        self,
        stream_sample_rate_hz: int,
        stream_channel_count: int,
    ) -> DecoderInputSpec:
        ...

    def load(self, input_spec: DecoderInputSpec) -> None:
        ...

    def infer(self, window: np.ndarray, meta: DecoderChunkMeta) -> DecoderOutput:
        ...

    def close(self) -> None:
        ...
