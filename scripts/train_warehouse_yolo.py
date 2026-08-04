"""Train a compact warehouse detector using an existing YOLOv8n baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/yolo_warehouse_dataset/warehouse.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("outputs/train"))
    parser.add_argument("--name", default="warehouse_yolov8n")
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        seed=1010,
        deterministic=True,
        workers=0,
    )


if __name__ == "__main__":
    main()
