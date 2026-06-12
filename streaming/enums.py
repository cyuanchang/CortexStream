from enum import Enum, IntEnum


class BoardIdsEnum(IntEnum):
    CYTON_DAISY = 2


class StreamNumeric(IntEnum):
    CHUNK_SIZE = 16
    CHUNK_QUEUE_MAXSIZE = 256
    RAW_QUEUE_MAXSIZE = 256
    ACQUISITION_POLL_MS = 20
    GUI_REFRESH_MS = 30
    GUI_WINDOW_SECONDS = 6
    GUI_PROCESSING_BUFFER_SECONDS = 22
    GUI_FILTER_GUARD_SECONDS = 2
    SPECTRAL_REFRESH_MS = 40
    FFT_N = 256
    HEADMAP_GRID_SIZE = 96


class StreamFloat(float, Enum):
    BANDPASS_LOW_HZ = 4.0
    BANDPASS_HIGH_HZ = 40.0
    NOTCH_HZ = 60.0
    BANDPOWER_LOW_HZ = 8.0
    BANDPOWER_HIGH_HZ = 13.0
    FFT_SMOOTHING_ALPHA = 0.9
    FFT_MIN_AMPLITUDE_UV = 0.01
    FFT_DISPLAY_Y_MAX_UV = 100.0
    HEADMAP_IDW_POWER = 2.0
    HEADMAP_IDW_EPS = 1e-6
    HEADMAP_SMOOTHING_ALPHA = 0.9


class StreamState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
