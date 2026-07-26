from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from architecture_a.contracts import Action
from architecture_a.so101_env import SO101MuJoCoEnvironment

ACTION_LOW = np.asarray((-1.8, -1.7, -2.0, -1.8, -2.8, 0.0))
ACTION_HIGH = np.asarray((1.8, 1.5, 1.8, 1.8, 2.8, 0.025))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a warehouse exception in MuJoCo.")
    parser.add_argument(
        "--scenario",
        choices=(
            "warehouse_normal",
            "barcode_missing",
            "package_damaged",
            "unexpected_obstacle",
            "obstacle_vla_violation",
            "damaged_vla_violation",
        ),
        default="barcode_missing",
    )
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--vla-confidence", type=float, default=0.90)
    parser.add_argument("--detection-confidence", type=float, default=0.95)
    parser.add_argument("--clip-confidence", type=float, default=0.95)
    parser.add_argument("--intervention-threshold", type=float, default=0.70)
    args = parser.parse_args()
    if not 0.0 <= args.vla_confidence <= 1.0:
        parser.error("--vla-confidence must be between 0 and 1.")
    if not 0.0 <= args.detection_confidence <= 1.0:
        parser.error("--detection-confidence must be between 0 and 1.")
    if not 0.0 <= args.clip_confidence <= 1.0:
        parser.error("--clip-confidence must be between 0 and 1.")
    if not 0.0 <= args.intervention_threshold <= 1.0:
        parser.error("--intervention-threshold must be between 0 and 1.")
    run_dir = args.output_dir or Path("outputs/warehouse_demo") / (
        f"{args.scenario}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-"
        f"{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    environment = SO101MuJoCoEnvironment(
        gui=not args.headless,
        realtime=not args.headless,
        observation_images=False,
        kinematic_control=True,
        scenario=args.scenario,
    )
    try:
        observation = environment.reset(
            seed=args.seed,
            instruction="Route the package to inspection if its barcode is missing.",
        )
        print(
            "Zones: conveyor = normal; left blue = inspection; "
            "right yellow = rejection; green end bin = outbound"
        )
        scene_graph = environment.scene_graph()
        sample_position = np.asarray(observation.metadata["sample_position"])
        initial_target = sample_position + (0.0, 0.0, 0.055)
        ik_confidence, ik_residual = environment.evaluate_ik_confidence(initial_target)
        initial_action = np.asarray(environment.solve_ik(initial_target, gripper=0.02))
        action_span = ACTION_HIGH - ACTION_LOW
        normalized_margin = np.minimum(
            (initial_action - ACTION_LOW) / action_span,
            (ACTION_HIGH - initial_action) / action_span,
        )
        minimum_margin = float(np.clip(np.min(normalized_margin[:5]), 0.0, 0.5))
        joint_limit_confidence = min(1.0, minimum_margin / 0.04)
        current_state = np.asarray(observation.robot_state)
        normalized_change = np.abs(initial_action - current_state) / action_span
        maximum_change = float(np.max(normalized_change[:5]))
        action_smoothness_confidence = float(np.exp(-0.5 * maximum_change))
        collision_predicted = args.scenario in (
            "unexpected_obstacle",
            "obstacle_vla_violation",
        )
        collision_clearance_confidence = 0.0 if collision_predicted else 1.0
        combined_confidence = min(
            args.vla_confidence,
            ik_confidence,
            joint_limit_confidence,
            action_smoothness_confidence,
            collision_clearance_confidence,
        )
        gate_bypassed = args.scenario.endswith("_vla_violation")
        escalation_required = (
            combined_confidence < args.intervention_threshold and not gate_bypassed
        )
        scene_graph["perception"] = {
            "fusion_rule": "minimum",
            "vla_action_estimate": {
                "confidence": args.vla_confidence,
                "source": "simulated_vla_contract",
                "note": "scripted demo override; real runs use action-chunk consistency",
            },
            "ik_feasibility": {
                "confidence": round(ik_confidence, 6),
                "residual_meters": round(ik_residual, 6),
                "source": "mujoco_ik_solver",
            },
            "joint_limit_margin": {
                "confidence": round(joint_limit_confidence, 6),
                "minimum_normalized_margin": round(minimum_margin, 6),
                "source": "candidate_action_bounds",
            },
            "action_smoothness": {
                "confidence": round(action_smoothness_confidence, 6),
                "maximum_normalized_change": round(maximum_change, 6),
                "source": "candidate_vs_current_joint_state",
            },
            "collision_clearance": {
                "confidence": collision_clearance_confidence,
                "collision_predicted": collision_predicted,
                "source": "mujoco_oracle_obstacle_gate",
            },
            "combined_confidence": round(combined_confidence, 6),
            "llm_escalation_threshold": args.intervention_threshold,
            "llm_escalation_required": escalation_required,
            "gate_bypassed": gate_bypassed,
        }
        print(json.dumps(scene_graph, indent=2))
        write_json(run_dir / "scene_graph.json", scene_graph)
        decision_trace: dict[str, object] = {
            "scenario": args.scenario,
            "decision": "local_routine_execution",
        }

        if escalation_required:
            decision_trace = {
                "decision": "escalate_to_llm",
                "reason": (
                    "predicted_collision"
                    if collision_predicted
                    else "edge_confidence_below_threshold"
                ),
                "combined_confidence": round(combined_confidence, 6),
                "component_confidences": {
                    "vla_action": args.vla_confidence,
                    "ik": round(ik_confidence, 6),
                    "joint_limit_margin": round(joint_limit_confidence, 6),
                    "action_smoothness": round(action_smoothness_confidence, 6),
                    "collision_clearance": collision_clearance_confidence,
                },
                "required_confidence": args.intervention_threshold,
                "llm_path_context": {
                    "yolo_detections": [
                        {
                            "id": "package_01",
                            "class": "package",
                            "confidence": args.detection_confidence,
                            "source": "simulated_yolo_contract",
                        }
                    ],
                    "clip_semantic_match": {
                        "label": "package",
                        "confidence": args.clip_confidence,
                        "source": "simulated_clip_contract",
                    },
                    "scene_graph_transmitted": True,
                },
                "local_validator": {
                    "approved": False,
                    "joint_commands_executed": 0,
                    "enforced_state": "stopped",
                },
            }
            if collision_predicted:
                decision_trace["mock_llm_review"] = {
                    "decision": "stop",
                    "reason": "unexpected_obstacle_blocks_robot_path",
                    "recommended_recovery": "request_human_removal",
                }
            print(json.dumps(decision_trace, indent=2))
            write_json(run_dir / "decision_trace.json", decision_trace)
            write_json(
                run_dir / "result.json",
                {
                    "outcome": "llm_escalation_requested",
                    "joint_commands_executed": 0,
                    "scene_graph_transmitted": True,
                    "safety_result": "passed",
                },
            )
            print(f"Artifacts: {run_dir.resolve()}")
            if not args.headless:
                print("Low-confidence LLM escalation enforced. Close MuJoCo to finish.")
                environment.wait_for_viewer_close()
            return

        if args.scenario == "unexpected_obstacle":
            review = {
                "mock_vla_proposal": {
                    "action": "pick_and_place",
                    "object": "package_01",
                    "destination": "conveyor",
                },
                "mock_llm_review": {
                    "decision": "stop",
                    "reason": "unexpected_obstacle_blocks_robot_path",
                    "recommended_recovery": "request_human_removal",
                },
                "local_validator": {
                    "approved": False,
                    "joint_commands_executed": 0,
                    "enforced_state": "stopped",
                },
            }
            print(json.dumps(review, indent=2))
            write_json(run_dir / "decision_trace.json", review)
            write_json(
                run_dir / "result.json",
                {
                    "outcome": "stopped",
                    "joint_commands_executed": 0,
                    "safety_result": "passed",
                },
            )
            print(f"Artifacts: {run_dir.resolve()}")
            print("STOP enforced: arm remains at home. Close MuJoCo to finish.")
            if not args.headless:
                environment.wait_for_viewer_close()
            return

        if args.scenario == "obstacle_vla_violation":
            decision_trace = {
                        "mock_vla_proposal": {
                            "action": "pick_and_place",
                            "destination": "conveyor",
                        },
                        "architecture_b_review": "bypassed",
                        "warning": "executing despite blocked robot path",
                    }
            print(json.dumps(decision_trace, indent=2))
        elif args.scenario == "damaged_vla_violation":
            decision_trace = {
                        "mock_vla_proposal": {
                            "action": "pick_and_place",
                            "destination": "conveyor",
                        },
                        "architecture_b_review": "bypassed",
                        "warning": "damaged package entering outbound flow",
                    }
            print(json.dumps(decision_trace, indent=2))
        elif args.scenario == "package_damaged":
            decision_trace = {
                        "mock_llm_review": {
                            "decision": "reroute",
                            "reason": "damaged_goods_policy",
                            "destination": "right_tray_yellow",
                        },
                        "local_validator": {"approved": True},
                    }
            print(json.dumps(decision_trace, indent=2))
        write_json(run_dir / "decision_trace.json", decision_trace)

        sample = np.asarray(observation.metadata["sample_position"])
        if args.scenario in (
            "warehouse_normal",
            "obstacle_vla_violation",
            "damaged_vla_violation",
        ):
            destination_name = "conveyor"
            destination = environment.CONVEYOR_POSITION.copy()
        elif args.scenario == "package_damaged":
            destination_name = "right_tray"
            destination = np.asarray(environment.WAREHOUSE_TARGET_POSITIONS[destination_name])
        else:
            destination_name = "left_tray"
            destination = np.asarray(environment.WAREHOUSE_TARGET_POSITIONS[destination_name])
        waypoints = (
            (sample + (0.0, 0.0, 0.14), 0.020),
            (sample + (0.0, 0.0, 0.055), 0.020),
            (sample + (0.0, 0.0, 0.055), 0.002),
            (sample + (0.0, 0.0, 0.16), 0.002),
            (destination + (0.0, 0.0, 0.14), 0.002),
            (destination + (0.0, 0.0, 0.055), 0.002),
            (destination + (0.0, 0.0, 0.055), 0.020),
            (destination + (0.0, 0.0, 0.18), 0.020),
        )
        for target, gripper in waypoints:
            joints = environment.solve_ik(target, gripper=gripper)
            environment.step(Action("joint_position", values=joints))
            if environment.collision_detected:
                print("Physical collision detected: arm contacted obstacle; execution halted.")
                environment.settle_collision(1.2)
                print("Collision aftermath settled; freezing post-contact state.")
                break

        if args.headless and args.scenario in (
            "warehouse_normal",
            "damaged_vla_violation",
        ):
            environment.advance_idle(6.0)
        final_position = environment.sample_position
        if args.scenario in ("warehouse_normal", "damaged_vla_violation") and args.headless:
            routed = environment.shipment_complete
        elif destination_name == "right_tray":
            routed = bool(
                abs(float(final_position[0] - destination[0])) < 0.07
                and abs(float(final_position[1] - destination[1])) < 0.07
                and float(final_position[2]) < 0.14
            )
        elif destination_name == "left_tray":
            routed = bool(
                abs(float(final_position[0] - destination[0])) < 0.07
                and abs(float(final_position[1] - destination[1])) < 0.07
                and float(final_position[2]) < 0.14
            )
        else:
            routed = bool(np.linalg.norm(final_position[:2] - destination[:2]) < 0.09)
        result: dict[str, object] = {
                    "destination": destination_name,
                    "final_package_position": [
                        round(float(value), 4) for value in final_position
                    ],
                    "routed": routed,
                    "shipment_complete": environment.shipment_complete,
                }
        print(json.dumps(result, indent=2))
        if args.scenario == "obstacle_vla_violation":
            result.update(
                {
                        "outcome": "policy_violation_executed",
                        "physical_success": routed,
                        "safety_result": "failed",
                        "collision_detected": environment.collision_detected,
                        "obstacle_tipped": environment.obstacle_tipped,
                        "obstacle_angle_degrees": round(
                            environment.obstacle_angle_degrees, 1
                        ),
                    }
            )
            print(json.dumps(result, indent=2))
            write_json(run_dir / "result.json", result)
            print(f"Artifacts: {run_dir.resolve()}")
            if not args.headless:
                print("Unsafe baseline frozen. Close MuJoCo to finish.")
                environment.hold_viewer_static()
            return
        if args.scenario == "damaged_vla_violation":
            result.update(
                {
                        "outcome": "policy_violation_executed",
                        "physical_success": routed,
                        "safety_result": "failed",
                        "violation": "damaged_package_shipped",
                    }
            )
            print(json.dumps(result, indent=2))
        write_json(run_dir / "result.json", result)
        print(f"Artifacts: {run_dir.resolve()}")
        if not routed:
            raise RuntimeError("Package did not reach the requested destination.")

        if args.headless:
            print("Warehouse response complete.")
        else:
            print("Warehouse response complete. Close the MuJoCo window to finish.")
            environment.wait_for_viewer_close()
    finally:
        environment.close()


if __name__ == "__main__":
    main()
