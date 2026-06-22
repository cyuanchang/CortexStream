from decoder.backends.eegnet_adapter import EEGNetBackend, EEGNetBackendConfig
from decoder.backends.eegmodels_reference import build_reference_eegnet, ensure_eegmodels_import_path

__all__ = [
    "EEGNetBackend",
    "EEGNetBackendConfig",
    "build_reference_eegnet",
    "ensure_eegmodels_import_path",
]
