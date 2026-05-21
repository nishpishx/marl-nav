from pathlib import Path
import time
import gymnasium
import racecar_gym.envs.gym_api
from stable_baselines3 import PPO
from multi_agent_wrapper import SingleAgentWrapper


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "env" / "coop_austria.yml"
MODEL_DIR = ROOT / "checkpoints"
MODEL_PATH = None  # Set to a specific model file to override the latest checkpoint
AGENT_ID = "A"
MAX_STEPS = 2000
FRAME_DELAY = 0.05


def resolve_model_path():
    if MODEL_PATH:
        model_path = Path(MODEL_PATH)
        return model_path if model_path.is_absolute() else ROOT / model_path

    candidates = list(MODEL_DIR.glob("*.zip"))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def make_env():
    env = SingleAgentWrapper(scenario=str(SCENARIO), agent_id=AGENT_ID)
    env.env.close()
    env.env = gymnasium.make(
        "MultiAgentRaceEnv-v0",
        scenario=str(SCENARIO),
        render_mode="human",
    )
    env.agent_ids = list(env.env.action_space.spaces.keys())
    return env


def run():
    env = make_env()
    model = PPO.load(str(resolve_model_path()), env=env)

    try:
        obs, _ = env.reset()
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            env.env.render()
            time.sleep(FRAME_DELAY)

            if terminated or truncated:
                print(f"finished after {step + 1} steps")
                break
        else:
            print(f"reached the {MAX_STEPS}-step cap")
    finally:
        env.close()


if __name__ == "__main__":
    run()
