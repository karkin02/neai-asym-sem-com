"""Standalone unit tests for the shared B/C perception pipeline.

These use injected fakes (no YOLO/CLIP weights, no network), so they run in any
environment with numpy.
"""

from __future__ import annotations

import unittest

import numpy as np

from shared.perception import (
    ClipGrounder,
    Detection,
    YoloDetector,
    build_scene_graph,
    crop_bbox,
    parse_yolo_result,
)


# --------------------------------------------------------------------------
# Fakes mimicking the Ultralytics result / CLIP embedder contracts
# --------------------------------------------------------------------------
class _FakeBox:
    def __init__(self, conf, cls, xyxy):
        self.conf = [conf]
        self.cls = [cls]
        self.xyxy = [type("T", (), {"tolist": lambda self, v=xyxy: list(v)})()]


class _FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeYolo:
    def __init__(self, result):
        self._result = result

    def __call__(self, frame, verbose=False):
        return [self._result]


class _FakeEmbedder:
    """Returns fixed vectors keyed by a marker channel in each crop/text."""

    def __init__(self, image_vectors, text_vector):
        self._image_vectors = np.asarray(image_vectors, dtype=float)
        self._text_vector = np.asarray(text_vector, dtype=float)

    def embed_images(self, crops):
        return self._image_vectors[: len(crops)]

    def embed_text(self, text):
        return self._text_vector


class DetectorTest(unittest.TestCase):
    def test_threshold_filters_low_confidence(self):
        result = _FakeResult(
            boxes=[
                _FakeBox(0.9, 0, (10, 10, 50, 50)),
                _FakeBox(0.2, 1, (60, 60, 90, 90)),
            ],
            names={0: "cup", 1: "bottle"},
        )
        detector = YoloDetector(conf_threshold=0.5, model=_FakeYolo(result))
        dets = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].label, "cup")
        self.assertEqual(dets[0].center, (30.0, 30.0))

    def test_parse_yolo_result_is_pure(self):
        result = _FakeResult([_FakeBox(0.7, 0, (0, 0, 20, 20))], {0: "tray"})
        dets = parse_yolo_result(result, conf_threshold=0.5)
        self.assertEqual(dets[0].as_dict()["label"], "tray")

    def test_crop_bbox_clamps_and_never_empty(self):
        frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        crop = crop_bbox(frame, (-10, -10, 30, 40))
        self.assertEqual(crop.shape, (40, 30, 3))
        degenerate = crop_bbox(frame, (50, 50, 50, 50))
        self.assertEqual(degenerate.shape, (1, 1, 3))


class SceneGraphTest(unittest.TestCase):
    def test_left_of_and_near_relations(self):
        dets = [
            Detection("sample", 0.9, (10, 40, 30, 60)),   # centre (20, 50)
            Detection("tray", 0.9, (200, 40, 240, 60)),   # centre (220, 50)
        ]
        graph = build_scene_graph(dets, image_width=320, image_height=240, task="pick")
        self.assertEqual(graph["task"], "pick")
        self.assertEqual(len(graph["objects"]), 2)
        rel = graph["relations"][0]
        self.assertEqual(
            (rel["subject"], rel["relation"], rel["object"]),
            ("sample_0", "left_of", "tray_1"),
        )
        self.assertIn("sample_0 left_of tray_1", graph["summary"])

    def test_zone_in_and_near(self):
        dets = [Detection("sample", 0.9, (95, 95, 105, 105))]  # centre (100, 100)
        zones = {"tray": {"bbox": [90, 90, 110, 110]}}
        graph = build_scene_graph(dets, 320, 240, zones=zones)
        zone_rels = [r for r in graph["relations"] if r["object"] == "tray"]
        self.assertEqual(zone_rels[0]["relation"], "in")


class ClipGroundingTest(unittest.TestCase):
    def test_scores_pick_best_matching_crop(self):
        # text vector aligns with the second crop
        grounder = ClipGrounder(
            embedder=_FakeEmbedder(
                image_vectors=[[1.0, 0.0], [0.0, 1.0]],
                text_vector=[0.0, 1.0],
            )
        )
        crops = [np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8)]
        result = grounder.score("the target", crops)
        self.assertEqual(result.best_index, 1)
        self.assertGreater(result.confidence, 0.5)
        self.assertAlmostEqual(sum(result.probabilities), 1.0, places=5)

    def test_empty_crops_zero_confidence(self):
        grounder = ClipGrounder(embedder=_FakeEmbedder([[1.0]], [1.0]))
        result = grounder.score("anything", [])
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.best_index, -1)


if __name__ == "__main__":
    unittest.main()
