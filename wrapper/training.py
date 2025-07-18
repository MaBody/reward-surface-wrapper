from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np


class EvalCheckpointCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        save_freq: int,
        save_path: str,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
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
            ep_rewards, ep_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
                render=False,
                warn=False,
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
