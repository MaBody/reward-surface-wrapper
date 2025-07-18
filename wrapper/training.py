import os
import numpy as np
from pathlib import Path
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm


class EvalCheckpointCallback(BaseCallback):
    def __init__(
        self,
        env,
        save_freq: int,
        save_path: str,
        n_eval_episodes: int = 5,
        n_eval_steps: int = np.inf,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env: Env = env
        self.save_freq = save_freq
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_steps = n_eval_steps
        self.best_mean_reward = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Called at each step during training
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            model_path = os.path.join(self.save_path, f"checkpoint_{step}.zip")
            self.model.save(model_path)
            if self.verbose > 1:
                print(f"[Checkpoint] Saved model at {model_path}")

            # Evaluate model
            _ = self.env.reset()
            ep_rewards, ep_lengths = evaluate_policy_limited(
                self.model,
                self.env,
                self.n_eval_episodes,
                self.n_eval_steps,
            )

            if self.verbose > 0:
                print(
                    f"[Eval] Step {step} | Reward: {np.mean(ep_rewards):.2f} +/- {np.std(ep_rewards):.2f} | Length: {np.mean(ep_lengths):.2f} +/- {np.std(ep_lengths):.2f}"
                )

            mean_reward = np.mean(ep_rewards)
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = os.path.join(self.save_path, "best_model.zip")
                self.model.save(best_path)
                if self.verbose > 1:
                    print(
                        f"[Best] New best model with reward {mean_reward:.2f} saved to {best_path}"
                    )

        return True


def evaluate_policy_limited(
    model: BaseAlgorithm, env: Env, n_eval_episodes=10, n_eval_steps=1000
):
    ep_rewards = []
    ep_lengths = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        while not done and (ep_length < n_eval_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)

    return ep_rewards, ep_lengths


def copy_env(env: str | Env):
    print(env.format())
