# run from project root: python training/train_ppo.py
# monitors: tensorboard --logdir logs/
# checkpoints saved to checkpoints/ppo_forward/

import io
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from multi_agent_wrapper import SingleAgentWrapper
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor

RUN_NAME = "ppo_forward"
CHECKPOINT_DIR = Path("checkpoints") / RUN_NAME
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TOTAL_TIMESTEPS = 300_000
CHECKPOINT_FREQ = 50_000

env = SingleAgentWrapper(
    scenario="env/coop_austria.yml",
    agent_id="A",
    motor_min_action=-0.5,
    motor_max_action=1.0,
)
env = Monitor(env, filename=f"{CHECKPOINT_DIR}/")

print("obs shape: ", env.observation_space.shape)
print("action space: ", env.action_space)
print("checkpoint dir: ", CHECKPOINT_DIR)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.18,
    ent_coef=0.003,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log="logs/",
    target_kl=0.03,
    verbose=1,
)


class SelfPlayCallback(BaseCallback):
    def __init__(self, update_freq=50_000, start_after=75_000):
        super().__init__()
        self.update_freq = update_freq
        self.start_after = start_after

    def _on_step(self):
        if (
            self.num_timesteps >= self.start_after
            and self.num_timesteps % self.update_freq == 0
        ):
            # DummyVecEnv -> Monitor -> SingleAgentWrapper
            inner_env = self.training_env.envs[0].env
            buf = io.BytesIO()
            self.model.save(buf)
            buf.seek(0)
            inner_env.set_opponent_model(PPO.load(buf), deterministic=False)
        return True


checkpoint_cb = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_checkpoint",
)

selfplay_cb = SelfPlayCallback()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_cb, selfplay_cb])
model.save(str(CHECKPOINT_DIR / "ppo_final"))

plot_results([str(CHECKPOINT_DIR)], TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, "PPO Training")
plt.savefig(CHECKPOINT_DIR / "training_curve.png")

env.close()
print("done")
