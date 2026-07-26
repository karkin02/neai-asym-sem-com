from __future__ import annotations

import random
import re

from .contracts import Action, Observation, PolicyDecision, StepResult


class MockPickPlaceEnvironment:
    """Small deterministic task used to verify the architecture end to end."""

    OBJECTS = ("red_sample", "blue_sample", "green_sample")
    TARGETS = ("left_tray", "right_tray")

    def __init__(self) -> None:
        self._rng = random.Random()
        self._object = ""
        self._target = ""
        self._held_object: str | None = None
        self._steps = 0

    def reset(self, *, seed: int, instruction: str) -> Observation:
        self._rng.seed(seed)
        self._object = self._rng.choice(self.OBJECTS)
        self._target = self._rng.choice(self.TARGETS)
        self._held_object = None
        self._steps = 0
        return self._observation()

    def step(self, action: Action) -> StepResult:
        self._steps += 1
        success = False
        failure = None

        if action.name == "pick":
            candidate = str(action.arguments.get("object", ""))
            if candidate == self._object and self._held_object is None:
                self._held_object = candidate
            else:
                failure = "wrong_object"
        elif action.name == "place":
            destination = str(action.arguments.get("target", ""))
            if self._held_object == self._object and destination == self._target:
                success = True
            else:
                failure = "wrong_target_or_empty_gripper"
        else:
            failure = "unsupported_action"

        terminated = success or failure is not None
        truncated = self._steps >= 4 and not terminated
        return StepResult(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            terminated=terminated,
            truncated=truncated,
            info={"success": success, "failure_reason": failure},
        )

    def _observation(self) -> Observation:
        return Observation(
            image=None,
            robot_state=(float(self._held_object is not None),),
            metadata={
                "visible_objects": [self._object],
                "visible_targets": [self._target],
                "held_object": self._held_object,
            },
        )


class MockLocalPolicy:
    """A model-shaped local policy for integration tests and harness work."""

    _PATTERN = re.compile(
        r"pick up the (?P<object>[a-z_]+) and place it in the (?P<target>[a-z_]+)",
        re.IGNORECASE,
    )

    def reset(self) -> None:
        pass

    def decide(self, observation: Observation, instruction: str) -> PolicyDecision:
        match = self._PATTERN.search(instruction.strip().rstrip("."))
        if not match:
            return PolicyDecision(
                action=Action("noop"),
                can_handle_instruction=False,
                grounding_confidence=0.0,
                policy_confidence=0.0,
                reason="instruction_outside_pick_place_skill",
            )

        object_name = match.group("object").lower()
        target_name = match.group("target").lower()
        visible_objects = observation.metadata.get("visible_objects", [])
        visible_targets = observation.metadata.get("visible_targets", [])
        held_object = observation.metadata.get("held_object")
        grounding = float(object_name in visible_objects and target_name in visible_targets)

        if held_object is None:
            action = Action("pick", arguments={"object": object_name})
        else:
            action = Action("place", arguments={"target": target_name})

        return PolicyDecision(
            action=action,
            can_handle_instruction=True,
            grounding_confidence=grounding,
            policy_confidence=0.95 if grounding else 0.35,
            reason=None if grounding else "requested_entities_not_visible",
        )

