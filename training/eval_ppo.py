from pathlib import Path
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_agent_wrapper import SingleAgentWrapper


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "env" / "coop_austria.yml"
MODEL_DIR = ROOT / "checkpoints"
MODEL_PATH = None  # Set to a specific model file to override the latest checkpoint
CAMERA_AGENT_ID = "A"
RENDER_MODE = "rgb_array_follow"
MAX_STEPS = 2000
FRAME_DELAY = 0.05
VIEW_WIDTH = 720
VIEW_HEIGHT = 480


def resolve_model_path():
    if MODEL_PATH:
        model_path = Path(MODEL_PATH)
        return model_path if model_path.is_absolute() else ROOT / model_path

    candidates = list(MODEL_DIR.glob("*.zip"))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def make_env():
    # Follow one car while the same policy drives every car in the scene.
    return SingleAgentWrapper(
        scenario=str(SCENARIO),
        agent_id=CAMERA_AGENT_ID,
        render_mode=RENDER_MODE,
        render_options={"width": VIEW_WIDTH, "height": VIEW_HEIGHT},
    )


def run():
    env = make_env()
    model = PPO.load(str(resolve_model_path()), env=env)

    try:
        obs, _ = env.env.reset(options={"mode": "grid"})
        plt.ion()
        # Show the follow-camera frames in a lightweight live preview window.
        fig, ax = plt.subplots(figsize=(VIEW_WIDTH / 100, VIEW_HEIGHT / 100), dpi=100)
        ax.axis("off")
        frame = env.render()
        image = ax.imshow(frame)
        fig.tight_layout(pad=0)
        plt.show(block=False)
        for step in range(MAX_STEPS):
            # Shared-policy control: feed each car's observation through the same actor.
            actions = {}
            for aid in env.agent_ids:
                agent_obs = env.flatten_obs(obs[aid], agent_id=aid)
                action, _ = model.predict(agent_obs, deterministic=True)
                actions[aid] = env.unflatten_action(action, agent_id=aid)

            obs, _, terminated, truncated, _ = env.env.step(actions)
            frame = env.render()
            if frame is not None:
                image.set_data(frame)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
            time.sleep(FRAME_DELAY)

            if all(terminated.values()) or truncated:
                print(f"finished after {step + 1} steps")
                break
        else:
            print(f"reached the {MAX_STEPS}-step cap")
    finally:
        plt.ioff()
        plt.close("all")
        env.close()


if __name__ == "__main__":
    run()
