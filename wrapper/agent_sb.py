from abc import ABC
from numpy.typing import ArrayLike
from wrapper.agent import AgentWrapper
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import gymnasium as gym
import numpy as np


class SB3Wrapper(AgentWrapper):
    def __init__(self, agent: BaseAlgorithm):
        self.agent = agent

    def initialize(self, weights: list[ArrayLike]) -> "SB3Wrapper":
        """Makes new agent with provided weights"""
        # Convert list of arrays to a single flat vector and load into policy
        vector = torch.cat([torch.as_tensor(w).flatten() for w in weights])
        self.agent.policy.load_from_vector(vector)
        return self

    def get_weights(self) -> list[ArrayLike]:
        # Returns the parameters as a list of numpy arrays (one per parameter tensor)
        return [
            param.detach()
            for param in self.agent.policy.parameters()
            # param.detach().cpu().numpy() for param in self.agent.policy.parameters()
        ]

    def estimate_reward(self, **kwargs) -> float:
        """
        Estimate average reward over n_steps using the current policy.
        Args:
            kwargs : dict
                For SB3, this contains env (str | Env) and n_steps (int)
        Returns:
            float: Average reward
        """
        env: str | gym.Env = kwargs["env"]
        if isinstance(env, str):
            env = gym.make(env)
        n_steps = kwargs["n_steps"]
        render = kwargs.get("render")
        n_episodes = kwargs.get("n_episodes", 1)

        obs, _ = env.reset()
        total_reward = 0.0

        episode_rewards, episode_lengths = evaluate_policy(
            self.agent,
            env,
            n_eval_episodes=n_episodes,
            render=render,
            deterministic=True,
            return_episode_rewards=True,
        )
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
            episode_lengths
        )

        # for _ in range(n_episodes):
        #     epoch_reward = 0.0
        #     step_count = 0
        #     while step_count < n_steps:
        #         action, _ = self.agent.predict(obs, deterministic=True)
        #         obs, reward, terminated, truncated, _ = env.step(action)
        #         done = terminated or truncated
        #         epoch_reward += reward
        #         step_count += 1
        #         if render:
        #             env.render()
        #         if done:
        #             obs, _ = env.reset()
        #     total_reward += epoch_reward / n_steps
        # total_reward /= n_episodes

        return mean_reward
