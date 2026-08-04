"""Generate a YOLO detection dataset from MuJoCo segmentation renders."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from architecture_a.so101_env import SO101MuJoCoEnvironment


CLASSES = (
    "package",
    "left_tray",
    "right_tray",
    "conveyor",
    "outbound_bin",
    "obstacle",
    "barcode",
    "damage_mark",
)

GEOM_GROUPS = {
    "package": ("sample_geom", "package_label"),
    "left_tray": ("left_tray_geom",),
    "right_tray": ("right_tray_geom",),
    "conveyor": (
        "conveyor_belt",
        "conveyor_roller_left",
        "conveyor_roller_right",
        *(f"conveyor_leg_{i}" for i in range(1, 5)),
        *(f"belt_slat_geom_{i}" for i in range(1, 8)),
    ),
    "outbound_bin": (
        "outbound_bin_floor",
        "outbound_bin_front",
        "outbound_bin_back",
        "outbound_bin_end",
    ),
    "obstacle": ("obstacle_post", "obstacle_stripe"),
    "barcode": tuple(f"barcode_{i}" for i in range(1, 5)),
    "damage_mark": ("damage_mark_1", "damage_mark_2", "damage_mark_front", "damage_mark_side"),
}

SCENARIOS = (
    "warehouse_normal",
    "barcode_missing",
    "package_damaged",
    "unexpected_obstacle",
)


def _geom_ids(env: SO101MuJoCoEnvironment) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for label, names in GEOM_GROUPS.items():
        result[label] = {int(env._model.geom(name).id) for name in names}  # noqa: SLF001
    return result


def _boxes(
    segmentation: np.ndarray,
    groups: dict[str, set[int]],
    enabled_classes: set[str],
) -> list[tuple[int, float, float, float, float]]:
    height, width = segmentation.shape[:2]
    object_ids = segmentation[:, :, 0]
    object_types = segmentation[:, :, 1]
    is_geom = object_types == int(mujoco.mjtObj.mjOBJ_GEOM)
    labels = []
    for class_id, class_name in enumerate(CLASSES):
        if class_name not in enabled_classes:
            continue
        mask = is_geom & np.isin(object_ids, list(groups[class_name]))
        ys, xs = np.where(mask)
        if len(xs) < 4:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        box_w, box_h = x2 - x1, y2 - y1
        labels.append(
            (
                class_id,
                ((x1 + x2) / 2) / width,
                ((y1 + y2) / 2) / height,
                box_w / width,
                box_h / height,
            )
        )
    return labels


def _set_inspection_pose(env: SO101MuJoCoEnvironment, scenario: str) -> None:
    """Move the arm so its original wrist camera sees the condition mark."""
    if scenario == "package_damaged":
        joints = env.DAMAGE_INSPECTION_POSE
    else:
        joints = env.BARCODE_INSPECTION_POSE
    for value, address in zip(joints, env._joint_qpos_addresses):  # noqa: SLF001
        env._data.qpos[address] = value  # noqa: SLF001
    env._data.qvel[:] = 0.0  # noqa: SLF001
    mujoco.mj_forward(env._model, env._data)  # noqa: SLF001


def _write_example(
    *,
    output: Path,
    split: str,
    stem: str,
    frame: np.ndarray,
    segmentation: np.ndarray,
    groups: dict[str, set[int]],
    enabled: set[str],
) -> None:
    labels = _boxes(segmentation, groups, enabled)
    Image.fromarray(frame).save(output / "images" / split / f"{stem}.png")
    text = "\n".join(
        f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        for class_id, cx, cy, bw, bh in labels
    )
    (output / "labels" / split / f"{stem}.txt").write_text(
        text + ("\n" if text else ""), encoding="utf-8"
    )


def generate(
    output: Path,
    train_count: int,
    val_count: int,
    width: int,
    height: int,
    warehouse_layout: str = "v1",
) -> Path:
    output = output.resolve()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, count, seed_base in (("train", train_count, 10_000), ("val", val_count, 90_000)):
        environments = {
            scenario: SO101MuJoCoEnvironment(
                realtime=False,
                observation_images=False,
                scenario=scenario,
                warehouse_layout=warehouse_layout,
            )
            for scenario in SCENARIOS
        }
        renderers = {}
        try:
            for scenario, env in environments.items():
                rgb = mujoco.Renderer(env._model, height=height, width=width)  # noqa: SLF001
                seg = mujoco.Renderer(env._model, height=height, width=width)  # noqa: SLF001
                seg.enable_segmentation_rendering()
                renderers[scenario] = (rgb, seg)
            for index in range(count):
                scenario = SCENARIOS[index % len(SCENARIOS)]
                env = environments[scenario]
                seed = seed_base + index
                env.reset(seed=seed, instruction="warehouse perception dataset")
                rgb_renderer, seg_renderer = renderers[scenario]
                rgb_renderer.update_scene(env._data, camera="overhead")  # noqa: SLF001
                frame = rgb_renderer.render()
                seg_renderer.update_scene(env._data, camera="overhead")  # noqa: SLF001
                segmentation = seg_renderer.render()
                enabled = {"package", "left_tray", "right_tray", "conveyor", "outbound_bin"}
                if scenario == "warehouse_normal":
                    enabled.add("barcode")
                elif scenario == "package_damaged":
                    enabled.add("damage_mark")
                elif scenario == "unexpected_obstacle":
                    enabled.add("obstacle")
                stem = f"{scenario}-{seed}"
                groups = _geom_ids(env)
                _write_example(
                    output=output,
                    split=split,
                    stem=stem,
                    frame=frame,
                    segmentation=segmentation,
                    groups=groups,
                    enabled=enabled,
                )

                # Tiny condition marks are not learnable from the overhead
                # view alone. Add close wrist frames for positive barcode and
                # damage examples plus missing-barcode negatives.
                if scenario in {"warehouse_normal", "barcode_missing", "package_damaged"}:
                    if scenario == "package_damaged":
                        _set_inspection_pose(env, scenario)
                    else:
                        _set_inspection_pose(env, scenario)
                    inspection_camera = "wrist"
                    rgb_renderer.update_scene(env._data, camera=inspection_camera)  # noqa: SLF001
                    close_frame = rgb_renderer.render()
                    seg_renderer.update_scene(env._data, camera=inspection_camera)  # noqa: SLF001
                    close_segmentation = seg_renderer.render()
                    close_enabled = {"package"}
                    if scenario == "warehouse_normal":
                        close_enabled.add("barcode")
                    elif scenario == "package_damaged":
                        close_enabled.add("damage_mark")
                    _write_example(
                        output=output,
                        split=split,
                        stem=f"{stem}-inspection",
                        frame=close_frame,
                        segmentation=close_segmentation,
                        groups=groups,
                        enabled=close_enabled,
                    )
        finally:
            for pair in renderers.values():
                for renderer in pair:
                    renderer.close()
            for env in environments.values():
                env.close()

    yaml_path = output / "warehouse.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    yaml_path.write_text(
        f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_warehouse_dataset"))
    parser.add_argument("--train-count", type=int, default=160)
    parser.add_argument("--val-count", type=int, default=40)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--warehouse-layout",
        choices=("v1", "v2", "v3"),
        default="v1",
        help="Versioned warehouse geometry represented in the renders.",
    )
    args = parser.parse_args()
    yaml_path = generate(
        args.output,
        args.train_count,
        args.val_count,
        args.width,
        args.height,
        args.warehouse_layout,
    )
    print(yaml_path)


if __name__ == "__main__":
    main()
