# run from project root: python training/train_ppo.py
# monitors: tensorboard --logdir logs/
# checkpoints saved to checkpoints/ every 50k steps

import os
import copy
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from multi_agent_wrapper import SingleAgentWrapper
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor

os.makedirs("checkpoints", exist_ok=True)
env = SingleAgentWrapper(scenario="env/coop_austria.yml", agent_id="A")
env = Monitor(env, filename="checkpoints/") 

print("obs shape: ", env.observation_space.shape)
print("action space: ", env.action_space)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log="logs/",
    verbose=1,
)

class SelfPlayCallback(BaseCallback):
      def __init__(self, update_freq=10_000):
          super().__init__()
          self.update_freq = update_freq

      def _on_step(self):
          if self.num_timesteps % self.update_freq == 0:
              # DummyVecEnv -> Monitor -> SingleAgentWrapper
              inner_env = self.training_env.envs[0].env
              inner_env.set_opponent_model(copy.deepcopy(self.model))        
          return True

checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path="checkpoints/",
    name_prefix="ppo_checkpoint",
)

selfplay_cb = SelfPlayCallback(update_freq=10_000)
model.learn(total_timesteps=500_000, callback=[checkpoint_cb, selfplay_cb])
model.save("checkpoints/ppo_final")

plot_results(["checkpoints/"], 500_000, results_plotter.X_TIMESTEPS, "PPO Training")
plt.savefig("checkpoints/training_curve.png")

env.close()
print("done")