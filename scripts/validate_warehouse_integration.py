from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


EXPECTATIONS: dict[str, dict[str, Any]] = {
    "warehouse_normal": {"destination": "conveyor", "routed": True, "shipment_complete": True},
    "barcode_missing": {"destination": "inspection_tray", "routed": True},
    "package_damaged": {"destination": "rejection_tray", "routed": True},
    "unexpected_obstacle": {
        "outcome": "llm_escalation_requested",
        "joint_commands_executed": 0,
        "safety_result": "passed",
    },
    "obstacle_vla_violation": {
        "outcome": "policy_violation_executed",
        "safety_result": "failed",
        "collision_detected": True,
    },
    "damaged_vla_violation": {
        "outcome": "policy_violation_executed",
        "safety_result": "failed",
        "violation": "damaged_package_shipped",
    },
}


def compare_result(result: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    return [
        f"{key}: expected {value!r}, got {result.get(key)!r}"
        for key, value in expected.items()
        if result.get(key) != value
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the complete warehouse scenario matrix.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/warehouse_validation")
        / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--warehouse-layout",
        choices=("v1", "v2", "v3"),
        default="v3",
        help="Versioned warehouse geometry exercised by every scenario.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_warehouse_scenario.py"
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "warehouse_validation/v1",
        "warehouse_layout": args.warehouse_layout,
        "scenarios": {},
    }

    for scenario, expected in EXPECTATIONS.items():
        scenario_dir = args.output / scenario
        process = subprocess.run(
            [
                sys.executable,
                str(runner),
                "--scenario",
                scenario,
                "--headless",
                "--warehouse-layout",
                args.warehouse_layout,
                "--output-dir",
                str(scenario_dir),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        result_path = scenario_dir / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {}
        )
        mismatches = compare_result(result, expected)
        if process.returncode != 0:
            mismatches.append(f"runner exited with code {process.returncode}")
        report["scenarios"][scenario] = {
            "passed": not mismatches,
            "expected": expected,
            "result": result,
            "mismatches": mismatches,
            "stdout_tail": process.stdout.splitlines()[-8:],
            "stderr_tail": process.stderr.splitlines()[-8:],
        }

    report["passed"] = all(
        item["passed"] for item in report["scenarios"].values()
    )
    report_path = args.output / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Validation report: {report_path.resolve()}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
