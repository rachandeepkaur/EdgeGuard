"""
Simulated consumer for the EdgeGuard Defender (Part 2).

Represents what happens AFTER the Defender decides. SOFTWARE SIMULATION
ONLY:
  - It only reads the shared DefenderOutput.
  - It never receives or modifies CAN frames.
  - It never opens, writes or changes ROAD files.
  - It never sees hidden labels (DefenderOutput contains none).

    ATTACK -> SIMULATED_ALERT or SIMULATED_ISOLATION (team setting)
    ACCEPT -> SIMULATED_FORWARD
"""

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict

from shared.schemas import DefenderOutput

SIMULATED_ALERT = "SIMULATED_ALERT"
SIMULATED_ISOLATION = "SIMULATED_ISOLATION"
SIMULATED_FORWARD = "SIMULATED_FORWARD"

Action = Literal["SIMULATED_ALERT", "SIMULATED_ISOLATION", "SIMULATED_FORWARD"]


class ActionRecord(BaseModel):
    """Structured, read-only record of one simulated action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_id: str
    attack_score: float
    decision: Literal["ATTACK", "ACCEPT"]
    action: Action
    simulated: Literal[True] = True
    message: str


class SimulatedConsumer:
    def __init__(self, attack_action: str = SIMULATED_ALERT):
        if attack_action not in (SIMULATED_ALERT, SIMULATED_ISOLATION):
            raise ValueError(
                f"attack_action must be {SIMULATED_ALERT!r} or "
                f"{SIMULATED_ISOLATION!r}, got {attack_action!r}"
            )
        self.attack_action = attack_action
        self.history: List[ActionRecord] = []

    def handle(self, output: DefenderOutput) -> ActionRecord:
        """Return the simulated action for one Defender result."""
        if not isinstance(output, DefenderOutput):
            raise TypeError(
                f"handle() needs a shared DefenderOutput, got {type(output).__name__}"
            )
        if output.decision == "ATTACK":
            action = self.attack_action
        else:
            action = SIMULATED_FORWARD
        record = ActionRecord(
            window_id=output.window_id,
            attack_score=output.attack_score,
            decision=output.decision,
            action=action,
            message=(
                f"[SIMULATION ONLY] {action} for window {output.window_id} "
                f"(score {output.attack_score:.2f}, threshold {output.threshold:.2f})"
            ),
        )
        self.history.append(record)
        return record

    def summary(self) -> Dict[str, int]:
        """Count simulated actions, for the demo."""
        counts = {SIMULATED_ALERT: 0, SIMULATED_ISOLATION: 0, SIMULATED_FORWARD: 0}
        for record in self.history:
            counts[record.action] += 1
        return counts