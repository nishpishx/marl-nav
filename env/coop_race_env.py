import gymnasium
import racecar_gym.envs.gym_api

env = gymnasium.make("MultiAgentRaceEnv-v0", scenario="env/coop_austria.yml")
obs, info = env.reset(options={"mode": "grid"})

done = False
while not done:
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()