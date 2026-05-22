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
from typing import Any, Dict, Optional

import numpy as np

AgentID = str
State = Dict[str, Any]
ActionDict = Dict[str, Any]


class RewardConfig:
    """Small set of tunable reward weights."""

    def __init__(
        self,
        progress_scale: float = 2500.0,
        forward_velocity_scale: float = 1.0,
        step_penalty: float = -0.002,
        target_forward_speed: float = 0.6,
        slow_speed_penalty_scale: float = 0.4,
        stall_speed_threshold: float = 0.15,
        stall_penalty: float = -0.15,
        motor_reward_scale: float = 0.15,
        brake_penalty_scale: float = 2.0,
        wall_collision_penalty: float = -300.0,
        opponent_collision_penalty: float = -180.0,
        wrong_way_penalty: float = -20.0,
        reverse_velocity_penalty_scale: float = 2.0,
    ):
        self.progress_scale = progress_scale
        self.forward_velocity_scale = forward_velocity_scale
        self.step_penalty = step_penalty
        self.target_forward_speed = target_forward_speed
        self.slow_speed_penalty_scale = slow_speed_penalty_scale
        self.stall_speed_threshold = stall_speed_threshold
        self.stall_penalty = stall_penalty
        self.motor_reward_scale = motor_reward_scale
        self.brake_penalty_scale = brake_penalty_scale
        self.wall_collision_penalty = wall_collision_penalty
        self.opponent_collision_penalty = opponent_collision_penalty
        self.wrong_way_penalty = wrong_way_penalty
        self.reverse_velocity_penalty_scale = reverse_velocity_penalty_scale


class RacecarReward:
    """Compute a compact reward from progress, speed, and safety events."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config if config is not None else RewardConfig()
        self.reset()

    def reset(self) -> None:
        self._last_progress: Optional[float] = None

    def reward(self, agent_id: AgentID, state: State, action: ActionDict) -> float:
        """Return the reward for one agent at a single environment step."""
        agent_state = state[agent_id]

        reward = self.config.step_penalty

        progress_delta = self._progress_delta(agent_state)
        reward += self.config.progress_scale * progress_delta

        forward_speed = self.forward_speed(agent_state)
        reward += self.config.forward_velocity_scale * max(0.0, forward_speed)
        reward -= self.config.reverse_velocity_penalty_scale * max(0.0, -forward_speed)

        slow_speed_gap = self.config.target_forward_speed - max(0.0, forward_speed)
        if slow_speed_gap > 0.0:
            reward -= (
                self.config.slow_speed_penalty_scale
                * slow_speed_gap
                / max(self.config.target_forward_speed, 1e-6)
            )

        if forward_speed < self.config.stall_speed_threshold:
            reward += self.config.stall_penalty

        throttle = self._throttle(action)
        if throttle >= 0.0:
            reward += self.config.motor_reward_scale * throttle
        else:
            reward -= self.config.brake_penalty_scale * abs(throttle)

        if agent_state.get("wall_collision", False):
            reward += self.config.wall_collision_penalty

        opponent_collisions = agent_state.get("opponent_collisions", [])
        reward += self.config.opponent_collision_penalty * len(opponent_collisions)

        if agent_state.get("wrong_way", False):
            reward += self.config.wrong_way_penalty

        return float(reward)

    def _progress_delta(self, agent_state: Dict[str, Any]) -> float:
        progress = float(agent_state["lap"]) + float(agent_state["progress"])
        if self._last_progress is None:
            self._last_progress = progress
            return 0.0

        delta = progress - self._last_progress
        self._last_progress = progress

        # Progress should be monotonic, but keep wraparound robust if lap is delayed
        if delta > 0.5:
            delta -= 1.0
        elif delta < -0.5:
            delta += 1.0
        return delta

    def forward_speed(agent_state: Dict[str, Any]) -> float:
        """Project world-frame velocity onto the car heading"""
        pose = np.asarray(agent_state["pose"], dtype=np.float32).reshape(-1)
        velocity = np.asarray(agent_state["velocity"], dtype=np.float32).reshape(-1)
        yaw = float(pose[5])
        return float(velocity[0]) * cos(yaw) + float(velocity[1]) * sin(yaw)

    def _throttle(action: ActionDict) -> float:
        key = "motor" if "motor" in action else "speed"
        value = np.asarray(action[key], dtype=np.float32).reshape(-1)
        return float(value[0])
