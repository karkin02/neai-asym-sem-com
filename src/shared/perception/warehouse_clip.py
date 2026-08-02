"""Warehouse-calibrated grounding on top of frozen CLIP image embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .clip_grounding import ClipGrounder, GroundingResult, _l2_normalize, _softmax


TARGET_ALIASES = {
    "sample": "package",
    "package": "package",
    "box": "package",
    "inspection tray": "left_tray",
    "rejection tray": "right_tray",
    "conveyor": "conveyor",
    "outbound bin": "outbound_bin",
    "obstacle": "obstacle",
    "barcode": "barcode",
    "damage": "damage_mark",
}


def target_class_from_text(text: str) -> str | None:
    lowered = text.lower()
    for phrase, class_name in TARGET_ALIASES.items():
        if phrase in lowered:
            return class_name
    return None


def normalize_object_class(label: str) -> str:
    """Normalize detector/task aliases to prototype class names."""
    lowered = label.strip().lower()
    return "package" if lowered in {"sample", "box", "package"} else lowered


class WarehouseClipGrounder:
    """Use frozen CLIP embeddings with synthetic warehouse class prototypes."""

    def __init__(
        self,
        prototype_path: str | Path,
        *,
        device: str = "cpu",
        clip_grounder: ClipGrounder | None = None,
        temperature: float = 0.01,
    ) -> None:
        payload = np.load(Path(prototype_path), allow_pickle=False)
        self.class_names = [str(value) for value in payload["class_names"]]
        self.prototypes = _l2_normalize(np.asarray(payload["prototypes"], dtype=np.float64))
        self._clip = clip_grounder or ClipGrounder(device=device)
        self.temperature = temperature

    def score(self, referring_expression: str, crops: Sequence[np.ndarray]) -> GroundingResult:
        if not crops:
            return GroundingResult([], [], -1, 0.0, 0.0)
        target_class = target_class_from_text(referring_expression)
        if target_class not in self.class_names:
            return self._clip.score(referring_expression, crops)
        embedder = self._clip._ensure_embedder()  # shared frozen backbone
        features = _l2_normalize(np.asarray(embedder.embed_images(crops), dtype=np.float64))
        target_index = self.class_names.index(target_class)
        all_class_scores = features @ self.prototypes.T
        scores = all_class_scores[:, target_index]
        probabilities = _softmax(scores, self.temperature)
        best_index = int(np.argmax(scores))
        confidence = float(np.clip((scores[best_index] + 1.0) / 2.0, 0.0, 1.0))
        alternatives = np.delete(all_class_scores[best_index], target_index)
        margin = float(scores[best_index] - np.max(alternatives))
        return GroundingResult(
            scores=[float(value) for value in scores],
            probabilities=[float(value) for value in probabilities],
            best_index=best_index,
            confidence=confidence,
            margin=margin,
        )
