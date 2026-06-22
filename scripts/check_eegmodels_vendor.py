from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoder.backends.eegmodels_reference import build_reference_eegnet, ensure_eegmodels_import_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check vendored arl-eegmodels import")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--chans", type=int, default=16)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()

    vendor_path = ensure_eegmodels_import_path()
    model = build_reference_eegnet(nb_classes=args.classes, chans=args.chans, samples=args.samples)
    print(f"vendor_path={vendor_path}")
    print(f"model={model.__class__.__name__}")
    print(f"input_shape={model.input_shape}")
    print(f"output_shape={model.output_shape}")


if __name__ == "__main__":
    main()
