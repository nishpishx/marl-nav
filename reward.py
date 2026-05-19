"""
Author: Brandon Bishop

Reward shaping for racecar_gym environment
(https://github.com/axelbr/racecar_gym?tab=readme-ov-file#installation)

Expected inputs:
- state[agent_id] is the per-agent observation dict from racecar_gym.
- action is the current control dict, usually {"motor", "steering"} or
  {"speed", "steering"}.
"""
from math import cos, sin
from typing import Any, Dict, Optional, Tuple

AgentID = str
State = Dict[str, Any]
ActionDict = Dict[str, Any]

class RewardConfig:
    """Weights used by :class:`RacecarReward`."""

    def __init__(
        self,
        progress_scale: float = 50.0,
        step_penalty: float = -0.01,
        wall_collision_penalty: float = -10.0,
        opponent_collision_penalty: float = -5.0,
        wrong_way_penalty: float = -2.0,
        action_smoothness_penalty: float = 0.25,
        forward_velocity_scale: float = 0.05,
    ):
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.wall_collision_penalty = wall_collision_penalty
        self.opponent_collision_penalty = opponent_collision_penalty
        self.wrong_way_penalty = wrong_way_penalty
        self.action_smoothness_penalty = action_smoothness_penalty
        self.forward_velocity_scale = forward_velocity_scale


class RacecarReward:
    """Compute a shaped reward from the current agent state and action."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config if config is not None else RewardConfig()
        self.reset()

    def reset(self) -> None:
        self._last_progress: Optional[float] = None
        self._last_action: Optional[Tuple[float, float]] = None

    def reward(self, agent_id: AgentID, state: State, action: ActionDict) -> float:
        """Return the reward for one agent at a single environment step."""
        agent_state = state[agent_id]

        # Start with a small step cost so doing nothing is slightly negative.
        reward = self.config.step_penalty

        # Progress is lap number plus within-lap progress in [0, 1].
        progress = float(agent_state["lap"]) + float(agent_state["progress"])
        if self._last_progress is not None:
            delta = progress - self._last_progress
            # Treat lap wraparound as forward motion instead of a big jump.
            if delta > 0.5:
                delta -= 1.0
            elif delta < -0.5:
                delta += 1.0
            reward += delta * self.config.progress_scale
        self._last_progress = progress

        if agent_state["wall_collision"]:
            reward += self.config.wall_collision_penalty

        # opponent_collisions is expected to be a sequence of contact entries.
        opponent_collisions = agent_state["opponent_collisions"]
        reward += self.config.opponent_collision_penalty * len(opponent_collisions)

        if agent_state["wrong_way"]:
            reward += self.config.wrong_way_penalty

        # Project the velocity vector onto the car heading to estimate forward speed.
        pose = agent_state["pose"]
        velocity = agent_state["velocity"]
        yaw = float(pose[5])
        forward_speed = float(velocity[0]) * cos(yaw) + float(velocity[1]) * sin(yaw)
        reward += self.config.forward_velocity_scale * max(0.0, forward_speed)

        # Some envs expose throttle as "motor", others as "speed".
        throttle = action["motor"] if "motor" in action else action["speed"]
        steering = action["steering"]
        throttle = float(throttle)
        steering = float(steering)
        if self._last_action is not None:
            last_throttle, last_steering = self._last_action
            # Penalize abrupt control changes relative to the last step.
            change = abs(throttle - last_throttle) + abs(steering - last_steering)
            reward -= self.config.action_smoothness_penalty * change
        self._last_action = (throttle, steering)

        return float(reward)
