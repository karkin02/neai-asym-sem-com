from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

REQUEST_SCHEMA = "architecture_a_to_b/v1"
RESPONSE_SCHEMA = "architecture_b_to_a/v1"
DESTINATIONS = frozenset(("conveyor", "left_tray", "right_tray"))


class RecoveryCommand(str, Enum):
    STOP = "STOP"
    RETRY = "RETRY"
    REROUTE = "REROUTE"
    CONTINUE = "CONTINUE"


@dataclass(frozen=True)
class ValidatedRecovery:
    command: RecoveryCommand
    reason: str
    destination: str | None
    confidence: float


def build_escalation_request(
    *, instruction: str, evidence_image: str, robot_state: Sequence[float],
    task_state: Mapping[str, Any], architecture_a_signals: Mapping[str, Any],
    reasons: Sequence[str], episode_id: int, handoff_stage: str,
) -> dict[str, Any]:
    request = {
        "schema": REQUEST_SCHEMA,
        "request_id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "handoff_stage": handoff_stage,
        "episode_id": episode_id,
        "instruction": instruction,
        "reasons": list(reasons),
        "evidence_image": evidence_image,
        "robot_state": [float(value) for value in robot_state],
        "task_state": dict(task_state),
        "architecture_a_signals": dict(architecture_a_signals),
        "architecture_b_pipeline": {
            "yolo": "pending", "clip": "pending",
            "scene_graph": "pending", "llm": "pending",
        },
    }
    validate_escalation_request(request)
    return request


def validate_escalation_request(request: Mapping[str, Any]) -> None:
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"Unsupported request schema: {request.get('schema')!r}")
    for key in ("request_id", "created_at", "handoff_stage", "instruction",
                "evidence_image", "task_state", "architecture_a_signals"):
        if not request.get(key):
            raise ValueError(f"Escalation request requires {key!r}.")
    state = request.get("robot_state")
    if not isinstance(state, list) or len(state) != 6:
        raise ValueError("Escalation robot_state must contain six joint values.")
    if not all(isinstance(value, (int, float)) for value in state):
        raise ValueError("Escalation robot_state values must be numeric.")


def validate_recovery_response(
    response: Mapping[str, Any], request: Mapping[str, Any]
) -> ValidatedRecovery:
    validate_escalation_request(request)
    if response.get("schema") != RESPONSE_SCHEMA:
        raise ValueError(f"Unsupported response schema: {response.get('schema')!r}")
    if response.get("request_id") != request["request_id"]:
        raise ValueError("Recovery response request_id does not match the request.")
    try:
        command = RecoveryCommand(str(response.get("command")))
    except ValueError as error:
        raise ValueError("Recovery command must be STOP, RETRY, REROUTE, or CONTINUE.") from error
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Recovery response requires a non-empty reason.")
    confidence = float(response.get("confidence", -1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Recovery confidence must be between 0 and 1.")
    destination = response.get("destination")
    if command is RecoveryCommand.REROUTE:
        if destination not in DESTINATIONS:
            raise ValueError("REROUTE requires a known destination.")
    elif destination is not None:
        raise ValueError(f"{command.value} must not include a destination.")
    problem = request["task_state"].get("problem")
    if problem == "unexpected_obstacle" and command is not RecoveryCommand.STOP:
        raise ValueError("An unexpected obstacle only permits STOP.")
    if problem == "package_damaged" and destination == "conveyor":
        raise ValueError("A damaged package cannot be rerouted to the conveyor.")
    if problem == "barcode_missing" and command is RecoveryCommand.REROUTE:
        if destination != "left_tray":
            raise ValueError("A missing barcode must be rerouted to left_tray.")
    return ValidatedRecovery(command, reason.strip(), destination, confidence)


class FileHandoffTransport:
    """Atomic filesystem transport usable by independently running A and B."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_request(self, request: Mapping[str, Any]) -> Path:
        validate_escalation_request(request)
        path = self.directory / f"request_{request['request_id']}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def read_response(self, request: Mapping[str, Any]) -> ValidatedRecovery | None:
        path = self.directory / f"response_{request['request_id']}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return validate_recovery_response(payload, request)
