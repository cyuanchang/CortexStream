from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_epoch_dir(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    x_path = path / "X.npy"
    y_path = path / "y.npy"
    meta_path = path / "metadata.json"
    if not x_path.exists() or not y_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"epoch directory missing required files: {path}")
    x = np.load(x_path)
    y = np.load(y_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if x.shape[0] != y.shape[0]:
        raise RuntimeError(f"X/Y count mismatch in {path}: {x.shape[0]} != {y.shape[0]}")
    return x, y, meta


def _make_split_indices(n: int, seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(min(n_train, n), 0)
    n_val = max(min(n_val, n - n_train), 0)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def export_dataset(
    epoch_dirs: list[Path],
    output_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    source_rows: list[dict] = []
    offset = 0
    class_labels: list[str] | None = None
    for epoch_dir in epoch_dirs:
        x, y, meta = _load_epoch_dir(epoch_dir)
        xs.append(x)
        ys.append(y)
        current_labels = list(meta.get("class_labels", []))
        if class_labels is None:
            class_labels = current_labels
        elif class_labels != current_labels:
            raise RuntimeError(f"class label mismatch across epoch dirs: {epoch_dir}")
        for local_idx in range(x.shape[0]):
            source_rows.append(
                {
                    "global_index": offset + local_idx,
                    "epoch_dir": str(epoch_dir),
                    "label_index": int(y[local_idx]),
                }
            )
        offset += x.shape[0]

    if not xs:
        raise RuntimeError("no epoch directories provided")
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    splits = _make_split_indices(
        n=x_all.shape[0],
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", x_all.astype(np.float32))
    np.save(output_dir / "y.npy", y_all.astype(np.int64))
    manifest = {
        "count": int(x_all.shape[0]),
        "shape": list(x_all.shape),
        "class_labels": class_labels or [],
        "splits": {k: len(v) for k, v in splits.items()},
        "seed": seed,
        "sources": source_rows,
        "split_indices": splits,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "dataset_exported "
        f"count={x_all.shape[0]} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export merged training dataset from epoch directories")
    parser.add_argument("--epoch-dir", action="append", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    args = parser.parse_args()
    if args.train_ratio < 0.0 or args.val_ratio < 0.0 or args.train_ratio + args.val_ratio > 1.0:
        raise SystemExit("train/val ratios must be >=0 and sum <=1")
    export_dataset(
        epoch_dirs=[Path(p) for p in args.epoch_dir],
        output_dir=Path(args.output_dir),
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
    )


if __name__ == "__main__":
    main()
