"""Calibrate Architecture C's package-prototype margin on held-out crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from shared.perception import ClipGrounder
from shared.perception.warehouse_clip import _l2_normalize


def load_crops(dataset: Path) -> tuple[list[np.ndarray], list[int]]:
    crops: list[np.ndarray] = []
    labels: list[int] = []
    for label_path in sorted((dataset / "labels" / "val").glob("*.txt")):
        image_path = dataset / "images" / "val" / f"{label_path.stem}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, cx, cy, bw, bh = map(float, line.split())
            x1 = max(0, int((cx - bw / 2) * width))
            y1 = max(0, int((cy - bh / 2) * height))
            x2 = min(width, int((cx + bw / 2) * width) + 1)
            y2 = min(height, int((cy + bh / 2) * height) + 1)
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2])
                labels.append(int(class_id))
    return crops, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/yolo_warehouse_dataset_v3")
    )
    parser.add_argument(
        "--prototypes",
        type=Path,
        default=Path("outputs/train/warehouse_clip_prototypes.npz"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/train/warehouse_clip_margin_calibration.json"),
    )
    args = parser.parse_args()

    payload = np.load(args.prototypes, allow_pickle=False)
    names = [str(value) for value in payload["class_names"]]
    prototypes = _l2_normalize(np.asarray(payload["prototypes"], dtype=np.float64))
    package_index = names.index("package")
    crops, labels = load_crops(args.dataset)
    embedder = ClipGrounder(device=args.device)._ensure_embedder()
    features = []
    for start in range(0, len(crops), args.batch_size):
        features.append(np.asarray(embedder.embed_images(crops[start:start + args.batch_size])))
    scores = _l2_normalize(np.concatenate(features)) @ prototypes.T
    margins = scores[:, package_index] - np.max(
        np.delete(scores, package_index, axis=1), axis=1
    )
    positive = np.asarray(labels) == package_index

    candidates = np.unique(np.concatenate((margins, np.asarray([-1.0, 1.0]))))
    rows = []
    for threshold in candidates:
        accepted = margins >= threshold
        recall = float(np.mean(accepted[positive]))
        false_accept = float(np.mean(accepted[~positive]))
        rows.append((float(threshold), recall, false_accept))
    eligible = [row for row in rows if row[2] <= 0.05]
    selected = max(eligible, key=lambda row: (row[1], row[0]))
    report = {
        "schema": "warehouse_clip_margin_calibration/v1",
        "dataset": str(args.dataset),
        "prototypes": str(args.prototypes),
        "positive_class": "package",
        "positive_crops": int(np.sum(positive)),
        "negative_crops": int(np.sum(~positive)),
        "selected_threshold": selected[0],
        "positive_recall": selected[1],
        "negative_false_accept_rate": selected[2],
        "positive_margin": {
            "minimum": float(np.min(margins[positive])),
            "median": float(np.median(margins[positive])),
            "maximum": float(np.max(margins[positive])),
        },
        "negative_margin": {
            "minimum": float(np.min(margins[~positive])),
            "median": float(np.median(margins[~positive])),
            "maximum": float(np.max(margins[~positive])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
