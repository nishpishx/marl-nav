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

    def __init__(self, scenario="env/coop_austria.yml", agent_id="A"):
        super().__init__()
        self.env = gymnasium.make(
            "MultiAgentRaceEnv-v0",
            scenario=scenario,
            render_mode="rgb_array_birds_eye",
        )
        self.agent_id = agent_id
        self.agent_ids = list(self.env.action_space.spaces.keys())

        # flatten the dict observation space into one vector
        agent_obs = self.env.observation_space[self.agent_id]
        self._obs_keys = sorted(agent_obs.spaces.keys())

        low = np.concatenate([agent_obs[k].low.flatten() for k in self._obs_keys])
        high = np.concatenate([agent_obs[k].high.flatten() for k in self._obs_keys])
        self.observation_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))

        
        agent_act = self.env.action_space[self.agent_id]
        self._act_keys = sorted(agent_act.spaces.keys())
        act_low = np.concatenate([agent_act[k].low.flatten() for k in self._act_keys])
        act_high = np.concatenate([agent_act[k].high.flatten() for k in self._act_keys])
        self.action_space = spaces.Box(low=act_low.astype(np.float32), high=act_high.astype(np.float32))

        self.reward_fn = RacecarReward()

    def _flatten_obs(self, obs_dict):
        return np.concatenate([np.array(obs_dict[k], dtype=np.float32).flatten() for k in self._obs_keys])

    def _unflatten_action(self, flat_action):
        # split the flat array back into the dict the env expects
        result = {}
        idx = 0
        agent_act = self.env.action_space[self.agent_id]
        for k in self._act_keys:
            size = agent_act[k].shape[0]
            result[k] = flat_action[idx:idx+size].astype(np.float32)
            idx += size
        return result

    def reset(self, **kwargs):
        obs, info = self.env.reset(options={"mode": "grid"})
        self.reward_fn.reset()
        return self._flatten_obs(obs[self.agent_id]), info.get(self.agent_id, {})

    def step(self, action):

        actions = {aid: self.env.action_space[aid].sample() for aid in self.agent_ids}
        action_dict = self._unflatten_action(action)
        actions[self.agent_id] = action_dict

        obs, rewards, terminated, truncated, state = self.env.step(actions)

        flat_obs = self._flatten_obs(obs[self.agent_id])
        # use custom reward instead of the env default
        reward = self.reward_fn.reward(self.agent_id, state, action_dict)
        done = bool(terminated[self.agent_id])
        trunc = bool(truncated)

        return flat_obs, reward, done, trunc, state.get(self.agent_id, {})

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = SingleAgentWrapper()
    print("obs shape: ", env.observation_space.shape)
    print("action space: ", env.action_space)
    check_env(env, warn=True)
    print("passed")
    env.close()