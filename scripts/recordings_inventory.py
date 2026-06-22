from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.recording_io import load_recording


def _iter_recording_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            dirs.append(child)
    return sorted(dirs)


def build_inventory(recordings_roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in recordings_roots:
        for recording_dir in _iter_recording_dirs(root):
            meta_path = recording_dir / "metadata.json"
            csv_path = recording_dir / "raw_frames.csv"
            row: dict[str, object] = {
                "recording_dir": str(recording_dir),
                "has_metadata": meta_path.exists(),
                "has_raw_csv": csv_path.exists(),
                "status": "ok",
                "error": "",
                "sample_rate_hz": 0,
                "eeg_channels": "",
                "sample_count": 0,
                "duration_seconds": 0.0,
            }
            try:
                data = load_recording(recording_dir)
                row["sample_rate_hz"] = data.sample_rate_hz
                row["eeg_channels"] = ",".join(str(v) for v in data.eeg_channels)
                row["sample_count"] = data.sample_count
                row["duration_seconds"] = round(data.duration_seconds, 3)
            except Exception as exc:
                row["status"] = "invalid"
                row["error"] = str(exc)
                if meta_path.exists():
                    try:
                        payload = json.loads(meta_path.read_text(encoding="utf-8"))
                        row["sample_rate_hz"] = int(payload.get("sample_rate_hz", 0))
                        eeg_channels = payload.get("eeg_channels", [])
                        row["eeg_channels"] = ",".join(str(v) for v in eeg_channels)
                    except Exception:
                        pass
            rows.append(row)
    return rows


def write_inventory(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recording_dir",
        "has_metadata",
        "has_raw_csv",
        "status",
        "error",
        "sample_rate_hz",
        "eeg_channels",
        "sample_count",
        "duration_seconds",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and QA recordings folders")
    parser.add_argument("--recordings-root", default="recordings", type=str)
    parser.add_argument("--extra-root", default="recordinigs", type=str)
    parser.add_argument("--output-csv", default="recordings_index.csv", type=str)
    args = parser.parse_args()

    roots = [Path(args.recordings_root), Path(args.extra_root)]
    rows = build_inventory(roots)
    output_csv = Path(args.output_csv)
    write_inventory(rows, output_csv)
    total = len(rows)
    invalid = sum(1 for row in rows if row["status"] != "ok")
    print(f"inventory_rows={total} invalid_rows={invalid} output={output_csv}")


if __name__ == "__main__":
    main()
