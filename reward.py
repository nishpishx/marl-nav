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
        brake_penalty_scale: float = 0.35,
        turn_assist_steering_scale: float = 0.45,
        turn_assist_speed_scale: float = 0.25,
        turn_assist_target_speed: float = 0.7,
        turn_assist_min_side_bias: float = 0.12,
        right_steering_sign: float = -1.0,
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
        self.turn_assist_steering_scale = turn_assist_steering_scale
        self.turn_assist_speed_scale = turn_assist_speed_scale
        self.turn_assist_target_speed = turn_assist_target_speed
        self.turn_assist_min_side_bias = turn_assist_min_side_bias
        self.right_steering_sign = right_steering_sign
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

        reward += self._turn_assist_reward(agent_state, action, forward_speed)

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

    def _turn_assist_reward(
        self,
        agent_state: Dict[str, Any],
        action: ActionDict,
        forward_speed: float,
    ) -> float:
        clearances = self._lidar_clearances(agent_state)
        if clearances is None:
            return 0.0

        front, left, right = clearances
        side_clearance = max(left, right)
        if side_clearance <= 1e-6:
            return 0.0

        front_blocked = np.clip((side_clearance - front) / side_clearance, 0.0, 1.0)
        side_bias = (right - left) / max(right + left, 1e-6)
        if front_blocked <= 0.0 or abs(side_bias) < self.config.turn_assist_min_side_bias:
            return 0.0

        desired_steering_sign = (
            self.config.right_steering_sign
            if side_bias > 0.0
            else -self.config.right_steering_sign
        )
        steering = self._steering(action)
        steering_alignment = desired_steering_sign * steering

        steering_reward = (
            self.config.turn_assist_steering_scale
            * front_blocked
            * abs(side_bias)
            * steering_alignment
        )

        excess_speed = max(0.0, forward_speed - self.config.turn_assist_target_speed)
        speed_penalty = self.config.turn_assist_speed_scale * front_blocked * excess_speed
        return float(steering_reward - speed_penalty)

    @staticmethod
    def _lidar_clearances(agent_state: Dict[str, Any]):
        lidar = agent_state.get("lidar")
        if lidar is None:
            return None

        readings = np.asarray(lidar, dtype=np.float32).reshape(-1)
        readings = readings[np.isfinite(readings)]
        if readings.size < 8:
            return None

        n = readings.size
        right_sector = readings[: max(1, int(0.30 * n))]
        front_sector = readings[int(0.40 * n): max(int(0.60 * n), int(0.40 * n) + 1)]
        left_sector = readings[int(0.70 * n):]

        right = float(np.percentile(right_sector, 75))
        front = float(np.percentile(front_sector, 35))
        left = float(np.percentile(left_sector, 75))
        return front, left, right

    @staticmethod
    def forward_speed(agent_state: Dict[str, Any]) -> float:
        """Project world-frame velocity onto the car heading"""
        pose = np.asarray(agent_state["pose"], dtype=np.float32).reshape(-1)
        velocity = np.asarray(agent_state["velocity"], dtype=np.float32).reshape(-1)
        yaw = float(pose[5])
        return float(velocity[0]) * cos(yaw) + float(velocity[1]) * sin(yaw)

    @staticmethod
    def _throttle(action: ActionDict) -> float:
        key = "motor" if "motor" in action else "speed"
        value = np.asarray(action[key], dtype=np.float32).reshape(-1)
        return float(value[0])

    @staticmethod
    def _steering(action: ActionDict) -> float:
        value = np.asarray(action.get("steering", 0.0), dtype=np.float32).reshape(-1)
        return float(value[0])
