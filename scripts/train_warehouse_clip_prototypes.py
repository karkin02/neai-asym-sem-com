"""Build compact warehouse class prototypes from frozen CLIP embeddings."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from shared.perception import ClipGrounder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("outputs/yolo_warehouse_dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/train/warehouse_clip_prototypes.npz"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    yaml_text = (args.dataset / "warehouse.yaml").read_text(encoding="utf-8")
    names = {}
    for line in yaml_text.splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and ":" in stripped:
            index, name = stripped.split(":", 1)
            names[int(index)] = name.strip()

    crops: list[np.ndarray] = []
    labels: list[int] = []
    for label_path in sorted((args.dataset / "labels" / "train").glob("*.txt")):
        image = np.asarray(Image.open(args.dataset / "images" / "train" / f"{label_path.stem}.png"))
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, cx, cy, bw, bh = map(float, line.split())
            x1 = max(0, int((cx - bw / 2) * width))
            y1 = max(0, int((cy - bh / 2) * height))
            x2 = min(width, int((cx + bw / 2) * width) + 1)
            y2 = min(height, int((cy + bh / 2) * height) + 1)
            crops.append(image[y1:y2, x1:x2])
            labels.append(int(class_id))

    embedder = ClipGrounder(device=args.device)._ensure_embedder()
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(crops), args.batch_size):
        features = np.asarray(embedder.embed_images(crops[start : start + args.batch_size]))
        for class_id, feature in zip(labels[start : start + args.batch_size], features):
            norm = np.linalg.norm(feature)
            grouped[class_id].append(feature / max(float(norm), 1e-12))

    class_ids = sorted(names)
    prototypes = np.stack(
        [np.mean(np.stack(grouped[class_id]), axis=0) for class_id in class_ids]
    )
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        class_names=np.asarray([names[class_id] for class_id in class_ids]),
        prototypes=prototypes.astype(np.float32),
    )
    print(f"wrote {args.output} from {len(crops)} crops")


if __name__ == "__main__":
    main()
