"""CLIP grounding of an instruction's referring expression against object crops.

Given a referring expression (e.g. "the red sample") and a list of detected
object image crops, this scores each crop against the text and returns a
per-object similarity plus a single confidence in ``[0, 1]`` — the shared
routing/uncertainty signal used by Architecture C's gate.

``open_clip`` and ``torch`` are imported lazily inside the default embedder, so
this module imports and unit-tests without the model installed. Tests inject a
fake embedder; production builds an ``open_clip`` ViT-B/32 embedder on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class GroundingResult:
    """Result of grounding one expression against N candidate crops.

    Attributes:
        scores: Raw cosine similarities per crop (``[-1, 1]``).
        probabilities: Softmax over ``scores`` (sums to 1); empty if no crops.
        best_index: Index of the highest-scoring crop, or ``-1`` if none.
        confidence: Routing signal in ``[0, 1]`` — the top softmax probability
            (how decisively one crop matches the expression).
    """

    scores: List[float]
    probabilities: List[float]
    best_index: int
    confidence: float
    margin: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": [round(float(s), 4) for s in self.scores],
            "probabilities": [round(float(p), 4) for p in self.probabilities],
            "best_index": self.best_index,
            "confidence": round(float(self.confidence), 4),
            "margin": round(float(self.margin), 4),
        }


class Embedder(Protocol):
    """Duck-typed embedding backend (real open_clip or a test fake)."""

    def embed_images(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...

    def embed_text(self, text: str) -> np.ndarray: ...


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    if x.size == 0:
        return x
    scaled = x / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.sum(exp)


class ClipGrounder:
    """Score object crops against a referring expression with CLIP.

    Args:
        model_name: open_clip architecture (default ``"ViT-B-32"``).
        pretrained: open_clip pretrained tag (default ``"openai"``).
        device: Torch device for the default embedder.
        temperature: Softmax temperature over cosine similarities; lower =
            sharper/more confident.
        embedder: Optional injected :class:`Embedder`. If ``None`` a lazy
            open_clip embedder is built on first use.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",  # matches the 'openai' pretrained tag
        pretrained: str = "openai",
        device: str = "cpu",
        temperature: float = 0.01,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.temperature = temperature
        self._embedder = embedder

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = _OpenClipEmbedder(
                self.model_name, self.pretrained, self.device
            )
        return self._embedder

    def score(self, referring_expression: str, crops: Sequence[np.ndarray]) -> GroundingResult:
        """Ground ``referring_expression`` against ``crops``.

        Args:
            referring_expression: The instruction's target phrase.
            crops: Object image crops (HWC arrays), one per candidate.

        Returns:
            A :class:`GroundingResult`. With no crops, confidence is ``0.0``.
        """
        if len(crops) == 0:
            return GroundingResult(scores=[], probabilities=[], best_index=-1, confidence=0.0, margin=0.0)

        embedder = self._ensure_embedder()
        image_features = np.asarray(embedder.embed_images(crops), dtype=np.float64)
        text_feature = np.asarray(embedder.embed_text(referring_expression), dtype=np.float64)

        image_features = _l2_normalize(image_features)
        text_feature = _l2_normalize(text_feature.reshape(1, -1))[0]

        scores = image_features @ text_feature  # cosine similarity, shape (N,)
        probabilities = _softmax(scores, self.temperature)
        best_index = int(np.argmax(scores))
        ranked = np.sort(scores)[::-1]
        # Keep softmax for ranking only. Routing confidence must not decrease
        # merely because the detector returned more candidate crops.
        confidence = float(np.clip((ranked[0] + 1.0) / 2.0, 0.0, 1.0))
        margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else 1.0
        return GroundingResult(
            scores=[float(s) for s in scores],
            probabilities=[float(p) for p in probabilities],
            best_index=best_index,
            confidence=confidence,
            margin=margin,
        )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class _OpenClipEmbedder:
    """Default embedder backed by open_clip (lazy heavy import)."""

    def __init__(self, model_name: str, pretrained: str, device: str) -> None:
        import open_clip  # lazy
        import torch  # lazy

        self._torch = torch
        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

    def embed_images(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        from PIL import Image  # lazy

        torch = self._torch
        tensors = []
        for crop in crops:
            array = np.asarray(crop)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            tensors.append(self._preprocess(Image.fromarray(array)))
        batch = torch.stack(tensors).to(self._device)
        with torch.inference_mode():
            features = self._model.encode_image(batch)
        return features.detach().cpu().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        torch = self._torch
        tokens = self._tokenizer([text]).to(self._device)
        with torch.inference_mode():
            features = self._model.encode_text(tokens)
        return features.detach().cpu().numpy()[0]
