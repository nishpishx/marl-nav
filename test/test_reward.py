import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from reward import RacecarReward, RewardConfig


def make_state(
    *,
    progress=0.0,
    lap=0,
    wall_collision=False,
    opponent_collisions=None,
    wrong_way=False,
    pose=None,
    velocity=None,
):
    return {
        "A": {
            "progress": progress,
            "lap": lap,
            "wall_collision": wall_collision,
            "opponent_collisions": opponent_collisions if opponent_collisions is not None else [],
            "wrong_way": wrong_way,
            "pose": pose if pose is not None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity": velocity if velocity is not None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }


class RacecarRewardTests(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig(
            progress_scale=100.0,
            forward_velocity_scale=1.0,
            step_penalty=0.0,
            target_forward_speed=0.0,
            slow_speed_penalty_scale=0.0,
            stall_speed_threshold=0.0,
            stall_penalty=0.0,
            motor_reward_scale=0.0,
            brake_penalty_scale=1.0,
            wall_collision_penalty=-10.0,
            opponent_collision_penalty=-5.0,
            wrong_way_penalty=-2.0,
            reverse_velocity_penalty_scale=1.0,
        )
        self.reward = RacecarReward(self.config)

    def test_reward_increases_with_forward_progress(self):
        action = {"motor": 0.25, "steering": 0.0}
        first = self.reward.reward("A", make_state(progress=0.10), action)
        second = self.reward.reward("A", make_state(progress=0.20), action)

        self.assertAlmostEqual(first, 0.0)
        self.assertGreater(second, first)
        self.assertAlmostEqual(second, 10.0)

    def test_reward_handles_wraparound_progress(self):
        action = {"motor": 0.25, "steering": 0.0}
        self.reward.reward("A", make_state(progress=0.98, lap=0), action)
        reward = self.reward.reward("A", make_state(progress=0.02, lap=1), action)

        self.assertAlmostEqual(reward, 4.0)

    def test_forward_speed_is_rewarded(self):
        stopped = self.reward.reward("A", make_state(), {"motor": 0.0, "steering": 0.0})

        self.reward.reset()
        moving = self.reward.reward(
            "A",
            make_state(velocity=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            {"motor": 0.0, "steering": 0.0},
        )

        self.assertGreater(moving, stopped)
        self.assertAlmostEqual(moving, 2.0)

    def test_reverse_speed_is_penalized(self):
        reward = self.reward.reward(
            "A",
            make_state(velocity=[-1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            {"motor": 0.0, "steering": 0.0},
        )

        self.assertAlmostEqual(reward, -1.5)

    def test_idle_stall_is_penalized(self):
        reward = RacecarReward(RewardConfig(
            progress_scale=0.0,
            forward_velocity_scale=0.0,
            step_penalty=0.0,
            target_forward_speed=0.0,
            slow_speed_penalty_scale=0.0,
            stall_speed_threshold=0.12,
            stall_penalty=-0.05,
            motor_reward_scale=0.0,
        ))

        idle = reward.reward("A", make_state(), {"motor": 0.0, "steering": 0.0})
        reward.reset()
        crawling = reward.reward(
            "A",
            make_state(velocity=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            {"motor": 0.0, "steering": 0.0},
        )

        self.assertLess(idle, crawling)
        self.assertAlmostEqual(idle, -0.05)

    def test_motor_reward_prefers_throttle_over_braking(self):
        reward = RacecarReward(RewardConfig(
            progress_scale=0.0,
            forward_velocity_scale=0.0,
            step_penalty=0.0,
            target_forward_speed=0.0,
            slow_speed_penalty_scale=0.0,
            stall_speed_threshold=0.0,
            stall_penalty=0.0,
            motor_reward_scale=0.2,
            brake_penalty_scale=1.0,
        ))

        throttle = reward.reward("A", make_state(), {"motor": 0.5, "steering": 0.0})
        reward.reset()
        braking = reward.reward("A", make_state(), {"motor": -0.5, "steering": 0.0})

        self.assertGreater(throttle, braking)
        self.assertAlmostEqual(throttle, 0.1)
        self.assertAlmostEqual(braking, -0.5)

    def test_collisions_and_wrong_way_are_clearly_negative(self):
        self.reward.reward("A", make_state(progress=0.10), {"motor": 0.0, "steering": 0.0})
        reward = self.reward.reward(
            "A",
            make_state(
                progress=0.12,
                wall_collision=True,
                opponent_collisions=["B"],
                wrong_way=True,
            ),
            {"motor": 0.0, "steering": 0.0},
        )

        self.assertLess(reward, -10.0)


if __name__ == "__main__":
    unittest.main()
