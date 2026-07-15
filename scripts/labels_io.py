from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialLabel:
    start_s: float
    end_s: float
    label: str
    task: str
    notes: str = ""


def load_trials(path: str | Path) -> list[TrialLabel]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"trials file not found: {source}")
    trials: list[TrialLabel] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        start_s = float(payload.get("start_s", 0.0))
        end_s = float(payload.get("end_s", 0.0))
        if end_s <= start_s:
            raise ValueError(f"invalid trial interval at line {line_number}: end_s must be > start_s")
        label = str(payload.get("label", "")).strip()
        if not label:
            raise ValueError(f"missing label at line {line_number}")
        task = str(payload.get("task", "")).strip()
        if not task:
            raise ValueError(f"missing task at line {line_number}")
        trials.append(
            TrialLabel(
                start_s=start_s,
                end_s=end_s,
                label=label,
                task=task,
                notes=str(payload.get("notes", "")),
            )
        )
    return trials
