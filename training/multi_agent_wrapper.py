import os
import sys
import gymnasium
from gymnasium import spaces
import numpy as np
import racecar_gym.envs.gym_api
from stable_baselines3.common.env_checker import check_env

# reward.py lives in the project root, one level up from training/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reward import RacecarReward


class SingleAgentWrapper(gymnasium.Env):
    # wraps the multi-agent env so SB3 can use it (expects single agent interface)

    def __init__(
        self,
        scenario="env/coop_austria.yml",
        agent_id="A",
        render_mode="rgb_array_birds_eye",
        render_options=None,
        terminate_on_wall_collision=True,
        terminate_on_opponent_collision=True,
        opponent_deterministic=False,
        motor_min_action=0.0,
        motor_max_action=1.0,
    ):
        super().__init__()
        self.agent_id = agent_id
        self._render_options = dict(render_options or {})
        self.terminate_on_wall_collision = terminate_on_wall_collision
        self.terminate_on_opponent_collision = terminate_on_opponent_collision
        self._opponent_model = None
        self._opponent_deterministic = opponent_deterministic
        self.motor_min_action = motor_min_action
        self.motor_max_action = motor_max_action
        if render_mode.startswith("rgb_array_") and "agent" not in self._render_options:
            self._render_options["agent"] = self.agent_id
        self.env = gymnasium.make(
            "MultiAgentRaceEnv-v0",
            scenario=scenario,
            render_mode=render_mode,
            render_options=self._render_options,
            disable_env_checker=True,
        )
        self.agent_ids = list(self.env.action_space.spaces.keys())

        # flatten the dict observation space into one vector
        self._obs_keys_by_agent = {}
        self._act_keys_by_agent = {}
        for aid in self.agent_ids:
            agent_obs = self.env.observation_space[aid]
            self._obs_keys_by_agent[aid] = sorted(agent_obs.spaces.keys())
            agent_act = self.env.action_space[aid]
            self._act_keys_by_agent[aid] = sorted(agent_act.spaces.keys())

        self._obs_keys = self._obs_keys_by_agent[self.agent_id]

        agent_obs = self.env.observation_space[self.agent_id]
        low = np.concatenate([agent_obs[k].low.flatten() for k in self._obs_keys])
        high = np.concatenate([agent_obs[k].high.flatten() for k in self._obs_keys])
        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
        )

        self._act_keys = self._act_keys_by_agent[self.agent_id]
        agent_act = self.env.action_space[self.agent_id]
        act_low, act_high = self._flat_action_bounds(self.agent_id)
        self.action_space = spaces.Box(
            low=act_low.astype(np.float32),
            high=act_high.astype(np.float32),
        )

        self.reward_fn = RacecarReward()

    def _flatten_obs(self, obs_dict, agent_id=None):
        agent_id = agent_id or self.agent_id
        obs_keys = self._obs_keys_by_agent[agent_id]
        return np.concatenate([
            np.array(obs_dict[k], dtype=np.float32).flatten()
            for k in obs_keys
        ])

    def flatten_obs(self, obs_dict, agent_id=None):
        return self._flatten_obs(obs_dict, agent_id=agent_id)

    def _unflatten_action(self, flat_action, agent_id=None):
        # split the flat array back into the dict the env expects
        result = {}
        idx = 0
        agent_id = agent_id or self.agent_id
        agent_act = self.env.action_space[agent_id]
        act_keys = self._act_keys_by_agent[agent_id]
        for k in act_keys:
            size = agent_act[k].shape[0]
            result[k] = flat_action[idx:idx + size].astype(np.float32)
            if k == "motor":
                result[k] = np.clip(
                    result[k],
                    self.motor_min_action,
                    self.motor_max_action,
                ).astype(np.float32)
            idx += size
        return result

    def unflatten_action(self, flat_action, agent_id=None):
        return self._unflatten_action(flat_action, agent_id=agent_id)

    def reset(self, **kwargs):
        options = dict(kwargs.pop("options", {}) or {})
        options.setdefault("mode", "grid")
        obs, info = self.env.reset(options=options, **kwargs)
        self.reward_fn.reset()
        self._all_obs = obs
        return self._flatten_obs(obs[self.agent_id]), info.get(self.agent_id, {})

    def set_opponent_model(self, model, deterministic=None):
        self._opponent_model = model
        if deterministic is not None:
            self._opponent_deterministic = deterministic

    def step(self, action):
        action_dict = self._unflatten_action(action)
        actions = {self.agent_id: action_dict}
        for aid in self.agent_ids:
            if aid == self.agent_id:
                continue
            if self._opponent_model is not None:
                flat = self._flatten_obs(self._all_obs[aid], agent_id=aid)
                opp_action, _ = self._opponent_model.predict(
                    flat,
                    deterministic=self._opponent_deterministic,
                )
                actions[aid] = self._unflatten_action(opp_action, agent_id=aid)
            else:
                actions[aid] = self._sample_action(aid)
        obs, _rewards, terminated, truncated, state = self.env.step(actions)
        self._all_obs = obs

        flat_obs = self._flatten_obs(obs[self.agent_id])
        # use custom reward instead of the env default
        reward_state = self._state_with_obs_sensors(state, obs)
        reward = self.reward_fn.reward(self.agent_id, reward_state, action_dict)
        done = self._flag_for_agent(terminated)
        trunc = self._flag_for_agent(truncated)

        agent_state = state.get(self.agent_id, {})
        collision_done, termination_reason = self._collision_termination(state)
        if collision_done:
            done = True

        info = dict(agent_state)
        if termination_reason is not None:
            info["termination_reason"] = termination_reason

        return flat_obs, reward, done, trunc, info

    def _state_with_obs_sensors(self, state, obs):
        agent_state = dict(state.get(self.agent_id, {}))
        agent_obs = obs.get(self.agent_id, {})
        if "lidar" in agent_obs and "lidar" not in agent_state:
            agent_state["lidar"] = agent_obs["lidar"]

        reward_state = dict(state)
        reward_state[self.agent_id] = agent_state
        return reward_state

    def _flat_action_bounds(self, agent_id):
        agent_act = self.env.action_space[agent_id]
        act_keys = self._act_keys_by_agent[agent_id]
        lows = []
        highs = []
        for key in act_keys:
            low = agent_act[key].low.flatten().astype(np.float32)
            high = agent_act[key].high.flatten().astype(np.float32)
            if key == "motor":
                low = np.maximum(low, self.motor_min_action)
                high = np.minimum(high, self.motor_max_action)
            lows.append(low)
            highs.append(high)
        return np.concatenate(lows), np.concatenate(highs)

    def _sample_action(self, agent_id):
        action = self.env.action_space[agent_id].sample()
        if "motor" in action:
            action["motor"] = np.clip(
                action["motor"],
                self.motor_min_action,
                self.motor_max_action,
            ).astype(np.float32)
        return action

    def _flag_for_agent(self, flag):
        if isinstance(flag, dict):
            return bool(flag.get(self.agent_id, False))
        return bool(flag)

    def _collision_termination(self, state):
        agent_state = state.get(self.agent_id, {})
        if self.terminate_on_wall_collision and agent_state.get("wall_collision", False):
            return True, "wall_collision"
        if (
            self.terminate_on_opponent_collision
            and len(agent_state.get("opponent_collisions", [])) > 0
        ):
            return True, "opponent_collision"

        return False, None

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = SingleAgentWrapper()
    print("obs shape: ", env.observation_space.shape)
    print("action space: ", env.action_space)
    check_env(env, warn=True)
    print("passed")
    env.close()
