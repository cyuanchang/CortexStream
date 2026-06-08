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


class StreamState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
