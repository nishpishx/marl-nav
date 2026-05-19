import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from reward import RacecarReward, RewardConfig

def make_state(
    *,
    progress=0.0,
    lap=0,
    time=0.0,
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
            "time": time,
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
            progress_scale=50.0,
            step_penalty=0.0,
            wall_collision_penalty=-10.0,
            opponent_collision_penalty=-5.0,
            wrong_way_penalty=-2.0,
            action_smoothness_penalty=1.0,
            forward_velocity_scale=0.0,
        )
        self.reward = RacecarReward(self.config)
        self.reward.reset()

    def test_reward_increases_with_forward_progress(self):
        action = {"motor": 0.25, "steering": 0.0}
        first = self.reward.reward("A", make_state(progress=0.10, time=1.0), action)
        second = self.reward.reward("A", make_state(progress=0.20, time=2.0), action)

        self.assertAlmostEqual(first, 0.0)
        self.assertGreater(second, first)
        self.assertAlmostEqual(second, 5.0)

    def test_reward_handles_wraparound_progress(self):
        action = {"motor": 0.25, "steering": 0.0}
        self.reward.reward("A", make_state(progress=0.98, lap=0, time=1.0), action)
        reward = self.reward.reward("A", make_state(progress=0.02, lap=1, time=2.0), action)

        self.assertAlmostEqual(reward, 2.0)

    def test_reward_penalizes_collision_wrong_way_and_action_changes(self):
        first_action = {"motor": 0.0, "steering": 0.0}
        second_action = {"motor": 1.0, "steering": -1.0}

        self.reward.reward("A", make_state(progress=0.10, time=1.0), first_action)
        reward = self.reward.reward(
            "A",
            make_state(
                progress=0.12,
                time=2.0,
                wall_collision=True,
                opponent_collisions=["B"],
                wrong_way=True,
            ),
            second_action,
        )

        self.assertLess(reward, -15.0)

    def test_reset_clears_history_and_supports_speed_actions(self):
        motor_action = {"motor": 0.1, "steering": 0.2}
        speed_action = {"speed": 0.1, "steering": 0.2}

        self.reward.reward("A", make_state(progress=0.30, time=1.0), motor_action)
        self.reward.reset()
        reward = self.reward.reward("A", make_state(progress=0.30, time=1.0), speed_action)

        self.assertAlmostEqual(reward, 0.0)

if __name__ == "__main__":
    unittest.main()
