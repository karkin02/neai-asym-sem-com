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

The structured target's ``destination`` uses the same vocabulary as the shared
handoff contract (``conveyor`` / ``left_tray`` / ``right_tray``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .payload import CompressionLevel, Payload

DESTINATIONS = ("conveyor", "left_tray", "right_tray")


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_object": self.target_object,
            "destination": self.destination,
            "gripper": self.gripper,
            "reasoning": self.reasoning,
            "source": self.source,
        }


class Planner(Protocol):
    def plan(self, payload: Payload) -> ActionTarget: ...


_SCHEMA_INSTRUCTION = (
    "Respond with STRICT JSON only, no prose, exactly: "
    '{"target_object": "<label or null>", '
    '"destination": "conveyor|left_tray|right_tray", '
    '"gripper": "open|close", "reasoning": "<one sentence>"}'
)


def parse_action_target(text: str, source: str) -> ActionTarget:
    """Parse a planner's raw text into an :class:`ActionTarget` (robust)."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in planner response: {text!r}")
    data = json.loads(match.group(0))
    destination = data.get("destination")
    if destination not in DESTINATIONS:
        destination = None
    gripper = str(data.get("gripper", "close")).lower()
    if gripper not in ("open", "close"):
        gripper = "close"
    target_object = data.get("target_object")
    return ActionTarget(
        target_object=str(target_object) if target_object else None,
        destination=destination,
        gripper=gripper,
        reasoning=str(data.get("reasoning", "")),
        source=source,
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
            model=self.model, messages=self._build_messages(payload)
        )
        text = response.choices[0].message.content or ""
        return parse_action_target(text, source=self.model)


class HeuristicPlanner:
    """Deterministic offline planner (no network) for testing / no-key runs.

    Chooses a destination from instruction keywords and the target object from
    the first non-tray detected label. It is intentionally simple — a stand-in
    for the cloud planner, clearly labelled ``source="heuristic"`` in metrics.
    """

    REJECT_WORDS = ("reject", "damaged", "defect", "discard")
    INSPECT_WORDS = ("inspect", "barcode", "missing", "check", "left")

    def plan(self, payload: Payload) -> ActionTarget:
        instruction = payload.instruction.lower()
        if any(w in instruction for w in self.REJECT_WORDS):
            destination = "right_tray"
        elif any(w in instruction for w in self.INSPECT_WORDS):
            destination = "left_tray"
        else:
            destination = "conveyor"

        target_object = None
        scene = payload.content.get("scene", {}) if isinstance(payload.content, dict) else {}
        for obj in scene.get("objects", []):
            label = obj.get("label", "")
            if "tray" not in label and "zone" not in label:
                target_object = label
                break

        return ActionTarget(
            target_object=target_object,
            destination=destination,
            gripper="close",
            reasoning=f"heuristic route to {destination}",
            source="heuristic",
        )


def get_planner(name: str, **kwargs: Any) -> Planner:
    """Build a planner by name (``gpt`` or ``heuristic``)."""
    if name == "gpt":
        return GptPlanner(**kwargs)
    if name == "heuristic":
        return HeuristicPlanner()
    raise ValueError(f"Unknown planner '{name}'. Choose 'gpt' or 'heuristic'.")
