from pathlib import Path
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces
from multi_agent_wrapper import SingleAgentWrapper


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "env" / "coop_austria.yml"
MODEL_DIR = ROOT / "checkpoints"
MODEL_PATH = None  # Set to a specific model file to override the latest checkpoint
CAMERA_AGENT_ID = "C"
RENDER_MODE = "rgb_array_follow"
MAX_STEPS = 2000
FRAME_DELAY = 0.0125
VIEW_WIDTH = 720
VIEW_HEIGHT = 480


def resolve_model_path():
    if MODEL_PATH:
        model_path = Path(MODEL_PATH)
        return model_path if model_path.is_absolute() else ROOT / model_path

    candidates = list(MODEL_DIR.rglob("*.zip"))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def make_env(action_space):
    motor_min_action = float(action_space.low[0])
    motor_max_action = float(action_space.high[0])
    return SingleAgentWrapper(
        scenario=str(SCENARIO),
        agent_id=CAMERA_AGENT_ID,
        render_mode=RENDER_MODE,
        render_options={"width": VIEW_WIDTH, "height": VIEW_HEIGHT},
        terminate_on_wall_collision=False,
        terminate_on_opponent_collision=False,
        opponent_deterministic=True,
        motor_min_action=motor_min_action,
        motor_max_action=motor_max_action,
    )


def run():
    model_path = resolve_model_path()
    print(f"loading model: {model_path}")
    model = PPO.load(str(model_path))
    env = make_env(model.action_space)
    check_for_correct_spaces(env, model.observation_space, model.action_space)
    env.set_opponent_model(model, deterministic=True)

    try:
        obs, _ = env.reset()
        plt.ion()
        fig, ax = plt.subplots(figsize=(VIEW_WIDTH / 100, VIEW_HEIGHT / 100), dpi=100)
        ax.axis("off")
        frame = env.render()
        image = ax.imshow(frame)
        fig.tight_layout(pad=0)
        plt.show(block=False)

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                image.set_data(frame)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
            time.sleep(FRAME_DELAY)

            if terminated or truncated:
                reason = info.get("termination_reason", "environment_done")
                print(f"finished after {step + 1} steps: {reason}")
                break
        else:
            print(f"reached the {MAX_STEPS}-step cap")
    finally:
        plt.ioff()
        plt.close("all")
        env.close()


if __name__ == "__main__":
    run()
