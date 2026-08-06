"""Cloud planners for Architecture B.

The planner receives a (possibly compressed) scene representation + instruction
that arrived over the simulated channel and returns a structured action target
the scripted controller can execute.

Two interchangeable implementations share the :class:`Planner` interface:

- :class:`GptPlanner` — OpenAI GPT-4o-mini. The API key is read from the
  ``OPENAI_API_KEY`` environment variable (never hardcoded); the client is
  injectable for testing.
- :class:`HeuristicPlanner` — a deterministic, offline fallback so B can run and
  be tested with no API key or network.

The structured target uses semantic destination roles (``conveyor`` /
``inspection_tray`` / ``rejection_tray``). Legacy physical names are accepted
at the parser boundary and normalized before local execution.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Optional, Protocol

from .payload import CompressionLevel, Payload
from shared.destinations import normalize_destination

DESTINATIONS = ("conveyor", "inspection_tray", "rejection_tray")
COMMANDS = ("STOP", "RETRY", "REROUTE", "CONTINUE")
MAX_COMPLETION_TOKENS = 150


@dataclass(frozen=True)
class ActionTarget:
    """Structured target returned by a planner.

    Attributes:
        target_object: Label of the object to act on (or ``None``).
        destination: One of :data:`DESTINATIONS`, or ``None`` if unspecified.
        gripper: ``"open"`` or ``"close"``.
        reasoning: Short natural-language justification.
        source: Which planner produced this (``"gpt-4o-mini"`` / ``"heuristic"``).
    """

    target_object: Optional[str]
    destination: Optional[str]
    gripper: str
    reasoning: str
    source: str
    command: str = "STOP"
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_object": self.target_object,
            "destination": self.destination,
            "gripper": self.gripper,
            "reasoning": self.reasoning,
            "source": self.source,
            "command": self.command,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


class Planner(Protocol):
    def plan(self, payload: Payload) -> ActionTarget: ...


_SCHEMA_INSTRUCTION = (
    "Use only detected visual evidence, including the detector-derived boolean "
    "visual_observations and their camera provenance. A false barcode is a "
    "reliable negative only when barcode_inspection_complete=true. "
    "A complete barcode inspection with barcode_detected=false is affirmative "
    "visual evidence of a missing-barcode defect, not uncertainty; when that "
    "evidence is present, use confidence at least 0.5 for the inspection-tray reroute. "
    "Policy: damage_mark_detected=true => REROUTE to "
    "rejection_tray; barcode_inspection_complete=true with barcode_detected=false => "
    "REROUTE to inspection_tray; barcode_detected=true with no damage => "
    "CONTINUE to conveyor; obstacle, incomplete inspection, "
    "conflict, or insufficient evidence => STOP with destination null. "
    "Respond with STRICT JSON only, no prose, exactly: "
    '{"command":"STOP|RETRY|REROUTE|CONTINUE",'
    '"target_object":"<detected label or null>",'
    '"destination":"conveyor|inspection_tray|rejection_tray|null",'
    '"gripper":"open|close","confidence":<number 0..1>,'
    '"evidence":["<detected fact>"],"reasoning":"<one sentence>"}'
)


def parse_action_target(text: str, source: str) -> ActionTarget:
    """Parse a planner's raw text into an :class:`ActionTarget` (robust)."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in planner response: {text!r}")
    data = json.loads(match.group(0))
    destination = normalize_destination(data.get("destination"))
    if destination not in DESTINATIONS:
        destination = None
    gripper = str(data.get("gripper", "close")).lower()
    if gripper not in ("open", "close"):
        gripper = "close"
    target_object = data.get("target_object")
    command = str(data.get("command", "STOP")).upper()
    if command not in COMMANDS:
        command = "STOP"
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence_value = data.get("evidence", [])
    evidence = tuple(str(item) for item in evidence_value) if isinstance(evidence_value, list) else ()
    return ActionTarget(
        target_object=str(target_object) if target_object else None,
        destination=destination,
        gripper=gripper,
        reasoning=str(data.get("reasoning", "")),
        source=source,
        command=command,
        confidence=confidence,
        evidence=evidence,
    )


class GptPlanner:
    """GPT-4o-mini planner. Key from ``OPENAI_API_KEY``; client injectable."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, client: Any = None) -> None:
        self.model = model
        self._client = client
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _ensure_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Set the environment variable, or run "
                    "Architecture B with --planner heuristic for an offline planner."
                )
            from openai import OpenAI  # lazy

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _build_messages(self, payload: Payload) -> list[dict[str, Any]]:
        if payload.level is CompressionLevel.RAW_IMAGE:
            data_uri = f"data:image/png;base64,{payload.content['image_b64']}"
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Instruction: {payload.instruction}\n{_SCHEMA_INSTRUCTION}"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ]
        scene_json = json.dumps(payload.content.get("scene", {}))
        prompt = (
            f"You are a warehouse robot task planner.\nInstruction: {payload.instruction}\n"
            f"Detected scene (JSON): {scene_json}\n{_SCHEMA_INSTRUCTION}"
        )
        return [{"role": "user", "content": prompt}]

    def plan(self, payload: Payload) -> ActionTarget:
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(payload),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        text = response.choices[0].message.content or ""
        target = parse_action_target(text, source=self.model)
        scene = payload.content.get("scene", {}) if isinstance(payload.content, dict) else {}
        observations = scene.get("visual_observations", {})
        certified_missing_barcode = bool(
            observations.get("barcode_inspection_complete")
            and not observations.get("barcode_detected")
            and observations.get("damage_inspection_complete")
            and not observations.get("damage_mark_detected")
        )
        if (
            certified_missing_barcode
            and target.command == "REROUTE"
            and target.destination == "inspection_tray"
            and target.confidence < 0.5
        ):
            confidence_by_view = observations.get("class_confidence_by_view", {})
            package_confidences = [
                float(confidence_by_view.get(view, {}).get("package", 0.0))
                for view in ("barcode", "damage")
            ]
            certified_confidence = min(package_confidences)
            if certified_confidence >= 0.5:
                target = replace(
                    target,
                    confidence=max(target.confidence, certified_confidence),
                    evidence=target.evidence + ("certified_missing_barcode",),
                )
        return target


class HeuristicPlanner:
    """Deterministic offline planner (no network) for testing / no-key runs.

    Chooses a destination from instruction keywords and the target object from
    the first non-tray detected label. It is intentionally simple — a stand-in
    for the cloud planner, clearly labelled ``source="heuristic"`` in metrics.
    """

    REJECT_WORDS = ("reject", "damaged", "defect", "discard")
    INSPECT_WORDS = ("inspect", "barcode", "missing", "check", "left")
    TARGET_LABELS = ("package", "sample", "parcel", "box")
    INFRASTRUCTURE_WORDS = ("tray", "zone", "bin", "conveyor", "obstacle")

    def plan(self, payload: Payload) -> ActionTarget:
        instruction = payload.instruction.lower()
        scene = payload.content.get("scene", {}) if isinstance(payload.content, dict) else {}
        labels = [str(obj.get("label", "")).lower() for obj in scene.get("objects", [])]
        labels = ["package" if label in {"sample", "box", "parcel"} else label for label in labels]
        observations = scene.get("visual_observations", {})
        if observations.get("obstacle_detected") or "obstacle" in labels:
            command, destination = "STOP", None
        elif (
            observations
            and (
                observations.get("barcode_inspection_complete") is False
                or observations.get("damage_inspection_complete") is False
            )
        ):
            command, destination = "STOP", None
        elif observations.get("damage_mark_detected") or (
            not observations and "damage_mark" in labels
        ):
            command = "REROUTE"
            destination = "rejection_tray"
        elif observations.get("barcode_detected"):
            command = "CONTINUE"
            destination = "conveyor"
        elif observations.get("barcode_inspection_complete") or observations.get("inspection_complete"):
            command = "REROUTE"
            destination = "inspection_tray"
        elif any(w in instruction for w in self.REJECT_WORDS):
            command, destination = "REROUTE", "rejection_tray"
        elif any(w in instruction for w in self.INSPECT_WORDS):
            command, destination = "REROUTE", "inspection_tray"
        else:
            command, destination = "STOP", None

        target_object = None
        target_object = next(
            (label for label in labels if any(word in label for word in self.TARGET_LABELS)),
            None,
        )
        if target_object is None:
            target_object = next(
                (
                    label
                    for label in labels
                    if label and not any(word in label for word in self.INFRASTRUCTURE_WORDS)
                ),
                None,
            )

        return ActionTarget(
            target_object=target_object,
            destination=destination,
            gripper="open",
            reasoning=f"heuristic route to {destination}",
            source="heuristic",
            command=command,
            confidence=1.0,
            evidence=tuple(labels),
        )


def get_planner(name: str, **kwargs: Any) -> Planner:
    """Build a planner by name (``gpt`` or ``heuristic``)."""
    if name == "gpt":
        return GptPlanner(**kwargs)
    if name == "heuristic":
        return HeuristicPlanner()
    raise ValueError(f"Unknown planner '{name}'. Choose 'gpt' or 'heuristic'.")
