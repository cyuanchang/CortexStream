from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vendor_eegmodels_path() -> Path:
    return _repo_root() / "third_party" / "arl-eegmodels"


def ensure_eegmodels_import_path() -> Path:
    """Insert ``third_party/arl-eegmodels`` on ``sys.path`` and return that path."""
    vendor_path = _vendor_eegmodels_path()
    if not vendor_path.exists():
        raise FileNotFoundError(f"EEGModels vendor path not found: {vendor_path}")
    vendor_str = str(vendor_path)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    return vendor_path


def build_reference_eegnet(nb_classes: int, chans: int, samples: int, **kwargs: Any) -> Any:
    """Build an EEGNet model from the vendored arl-eegmodels package."""
    ensure_eegmodels_import_path()
    from EEGModels import EEGNet  # type: ignore

    return EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples, **kwargs)
