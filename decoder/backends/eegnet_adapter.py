from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from decoder.types import DecoderChunkMeta, DecoderInputSpec, DecoderOutput


@dataclass(frozen=True)
class EEGNetBackendConfig:
    model_path: str = ""
    class_labels: tuple[str, ...] = ("class_0", "class_1")
    samples: int = 256
    expected_channels: int | None = None
    input_layout: Literal["channels_first", "channels_last"] = "channels_last"
    fallback_to_mock: bool = True


class EEGNetBackend:
    """EEGNet adapter behind a generic decoder backend contract."""

    def __init__(self, config: EEGNetBackendConfig):
        self._config = config
        self._input_spec: DecoderInputSpec | None = None
        self._model = None
        self._loaded = False
        self._mode: Literal["model", "mock"] | None = None
        self._last_load_error = ""

    @property
    def backend_name(self) -> str:
        return "eegnet"

    @property
    def backend_mode(self) -> str:
        return self._mode or "unloaded"

    @property
    def last_load_error(self) -> str:
        return self._last_load_error

    def required_input_spec(
        self,
        stream_sample_rate_hz: int,
        stream_channel_count: int,
    ) -> DecoderInputSpec:
        channels = stream_channel_count if self._config.expected_channels is None else self._config.expected_channels
        return DecoderInputSpec(
            channels=int(channels),
            samples=max(int(self._config.samples), 1),
            sample_rate_hz=max(int(stream_sample_rate_hz), 1),
            layout=self._config.input_layout,
        )

    def load(self, input_spec: DecoderInputSpec) -> None:
        self._input_spec = input_spec
        self._model = None
        self._loaded = False
        self._mode = None
        self._last_load_error = ""
        model_path = self._config.model_path.strip()
        if model_path:
            path = Path(model_path)
            if not path.exists():
                if not self._config.fallback_to_mock:
                    self._last_load_error = f"EEGNet model file does not exist: {path}"
                    raise RuntimeError(self._last_load_error)
                self._last_load_error = f"EEGNet model file does not exist: {path}"
            else:
                try:
                    from tensorflow.keras.models import load_model

                    self._model = load_model(str(path))
                    self._validate_loaded_model()
                    self._loaded = True
                    self._mode = "model"
                    return
                except Exception as exc:
                    self._last_load_error = f"failed to load EEGNet model: {exc}"
                    if not self._config.fallback_to_mock:
                        raise RuntimeError(self._last_load_error) from exc
        if self._config.fallback_to_mock:
            self._loaded = True
            self._mode = "mock"
            return
        self._last_load_error = "EEGNet model is not loaded and mock fallback is disabled."
        raise RuntimeError(self._last_load_error)

    def infer(self, window: np.ndarray, meta: DecoderChunkMeta) -> DecoderOutput:
        if not self._loaded or self._input_spec is None:
            raise RuntimeError("EEGNet backend is not loaded.")
        expected_shape = (self._input_spec.channels, self._input_spec.samples)
        matrix = np.asarray(window, dtype=np.float64)
        if matrix.shape != expected_shape:
            raise RuntimeError(f"unexpected decoder window shape {matrix.shape}, expected {expected_shape}")
        if self._mode == "model" and self._model is not None:
            scores = self._predict_model_scores(matrix)
        else:
            scores = self._predict_mock_scores(matrix)
        label_idx = int(np.argmax(scores))
        label = self._config.class_labels[label_idx] if label_idx < len(self._config.class_labels) else f"class_{label_idx}"
        confidence = float(scores[label_idx])
        return DecoderOutput(
            label=label,
            confidence=confidence,
            scores=tuple(float(v) for v in scores),
            inference_ms=0.0,
            chunk_sequence_end=meta.sequence_id,
            device_timestamp_end=meta.device_timestamp_end,
            host_timestamp=meta.host_timestamp,
        )

    def close(self) -> None:
        self._model = None
        self._loaded = False
        self._mode = None

    def _predict_model_scores(self, matrix: np.ndarray) -> np.ndarray:
        x = self._prepare_input_tensor(matrix)
        preds = self._model.predict(x, verbose=0)
        vector = np.asarray(preds, dtype=np.float64).reshape(-1)
        return self._normalize_scores(vector)

    def _validate_loaded_model(self) -> None:
        if self._model is None or self._input_spec is None:
            raise RuntimeError("cannot validate model without input spec")
        sample = np.zeros((self._input_spec.channels, self._input_spec.samples), dtype=np.float64)
        x = self._prepare_input_tensor(sample)
        preds = self._model.predict(x, verbose=0)
        vector = np.asarray(preds, dtype=np.float64).reshape(-1)
        if vector.size <= 0:
            raise RuntimeError("model output is empty")
        expected_classes = len(self._config.class_labels)
        if expected_classes > 0 and vector.size != expected_classes:
            raise RuntimeError(
                f"model class count mismatch: model={vector.size}, labels={expected_classes}"
            )

    def _prepare_input_tensor(self, matrix: np.ndarray) -> np.ndarray:
        if self._input_spec is None:
            raise RuntimeError("input spec missing")
        # Shape to (batch, chans, samples, 1) for channels_first, else transpose to channels_last.
        if self._input_spec.layout == "channels_first":
            tensor = matrix[np.newaxis, :, :, np.newaxis]
        else:
            tensor = matrix.T[np.newaxis, :, :, np.newaxis]
        return np.asarray(tensor, dtype=np.float32)

    def _predict_mock_scores(self, matrix: np.ndarray) -> np.ndarray:
        n_classes = max(len(self._config.class_labels), 2)
        # Mock mode: map global variance through a sigmoid into class-0 score; remaining mass is shared.
        variance = float(np.var(matrix))
        anchor = 1.0 / (1.0 + np.exp(-0.02 * (variance - 25.0)))
        scores = np.full(n_classes, (1.0 - anchor) / float(max(n_classes - 1, 1)), dtype=np.float64)
        scores[0] = anchor
        return self._normalize_scores(scores)

    @staticmethod
    def _normalize_scores(values: np.ndarray) -> np.ndarray:
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
        if scores.size == 0:
            return np.array([1.0], dtype=np.float64)
        scores = np.clip(scores, 0.0, None)
        total = float(np.sum(scores))
        if total <= 1e-12:
            return np.full(scores.size, 1.0 / scores.size, dtype=np.float64)
        return scores / total
