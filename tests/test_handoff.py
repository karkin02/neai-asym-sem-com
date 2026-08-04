from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shared.handoff import (
    FileHandoffTransport,
    RecoveryCommand,
    build_escalation_request,
    validate_recovery_response,
)


def request_for(problem: str | None) -> dict[str, object]:
    return build_escalation_request(
        instruction="Route package.",
        evidence_image="evidence.png",
        robot_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.02),
        task_state={"problem": problem, "target_name": "conveyor"},
        architecture_a_signals={"task_success": False},
        reasons=(problem or "low_vla_action_consistency",),
        episode_id=0,
        handoff_stage="pre_execution_validation",
    )


class HandoffContractTest(unittest.TestCase):
    def test_valid_barcode_reroute_is_approved(self) -> None:
        request = request_for("barcode_missing")
        recovery = validate_recovery_response(
            {
                "schema": "architecture_b_to_a/v1",
                "request_id": request["request_id"],
                "command": "REROUTE",
                "destination": "left_tray",
                "reason": "manual barcode inspection required",
                "confidence": 0.91,
            },
            request,
        )
        self.assertEqual(recovery.command, RecoveryCommand.REROUTE)
        self.assertEqual(recovery.destination, "inspection_tray")

    def test_semantic_rejection_destination_is_approved(self) -> None:
        request = request_for("package_damaged")
        recovery = validate_recovery_response(
            {
                "schema": "architecture_b_to_a/v1",
                "request_id": request["request_id"],
                "command": "REROUTE",
                "destination": "rejection_tray",
                "reason": "damage confirmed",
                "confidence": 0.95,
            },
            request,
        )
        self.assertEqual(recovery.destination, "rejection_tray")

    def test_obstacle_cannot_be_overridden_with_continue(self) -> None:
        request = request_for("unexpected_obstacle")
        with self.assertRaisesRegex(ValueError, "only permits STOP"):
            validate_recovery_response(
                {
                    "schema": "architecture_b_to_a/v1",
                    "request_id": request["request_id"],
                    "command": "CONTINUE",
                    "reason": "path looks clear",
                    "confidence": 0.99,
                },
                request,
            )

    def test_file_transport_rejects_response_for_another_request(self) -> None:
        request = request_for(None)
        with TemporaryDirectory() as directory:
            transport = FileHandoffTransport(Path(directory))
            transport.write_request(request)
            response_path = Path(directory) / f"response_{request['request_id']}.json"
            response_path.write_text(
                json.dumps(
                    {
                        "schema": "architecture_b_to_a/v1",
                        "request_id": "wrong-id",
                        "command": "STOP",
                        "reason": "uncertain",
                        "confidence": 0.8,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                transport.read_response(request)


if __name__ == "__main__":
    unittest.main()
