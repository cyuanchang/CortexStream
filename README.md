# CortexStream

Real-time EEG acquisition, visualization, and neural decoding for OpenBCI Cyton+Daisy (16 channels · 125 Hz).

CortexStream provides a closed-loop Python stack: BrainFlow acquisition fans out through a pub-sub bus to independent GUI, recorder, and decoder consumers so monitoring and inference stay decoupled.

## Architecture

- **Streaming** (`streaming/`) — BrainFlow board client, `PubSubBus` fanout, raw-frame recorder
- **Realtime processing** (`Realtime_processing/`) — bandpass / notch / CAR, FFT, scalp headmap, band power, decoder preprocessing
- **GUI** (`gui/`) — PySide6 + pyqtgraph time series, spectrum, head plot, band-power window, stream lifecycle
- **Decoder** (`decoder/`) — rolling-window runtime, `manifest.json` contract, EEGNet backend (TensorFlow/Keras)
- **Offline pipeline** (`Offline_training/`, `scripts/`, `Validation/`) — epoching, training, PSD validation, offline replay gate

## Run the GUI

```bash
pip install brainflow numpy scipy pyqtgraph PySide6 tensorflow
python -m gui.app --serial-port COM3 --recordings-dir recordings
```

Live decode with a trained manifest:

```bash
python -m gui.app \
  --serial-port COM3 \
  --recordings-dir recordings \
  --decoder-manifest models/ssvep/baseline_8x2/manifest.json \
  --decoder-task-type ssvep \
  --decoder-deployment-mode live
```

## Key defaults

Defined in `streaming/enums.py` and processing modules:

- Board: OpenBCI Cyton+Daisy (`board_id = 2`)
- Chunk size: 16 samples
- GUI refresh: ~30 ms; spectral clock: 40 ms
- GUI filter path: bandpass 4–40 Hz, notch 60 Hz (CAR off for display)
- Decoder preprocess (manifest-driven): bandpass, notch, CAR as specified per model

## Portfolio site

Static site in `docs/` (GitHub Pages: branch `/docs`).

```text
https://cyuanchang.github.io/BCI/
```

## Repository layout

```text
streaming/              acquisition + pub-sub + recorder
Realtime_processing/    filters, FFT, headmap, decoder preprocess
gui/                    monitoring UI + stream controller
decoder/                online inference runtime + EEGNet backend
Offline_training/       SSVEP EEGNet training entrypoints
scripts/                epoching, export, offline replay utilities
Validation/             PSD / separability analysis scripts
docs/                   CortexStream portfolio site
```
