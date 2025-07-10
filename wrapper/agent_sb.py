from abc import ABC
from numpy.typing import ArrayLike
from wrapper.agent import AgentWrapper
from stable_baselines3.common.base_class import BaseAlgorithm


class StableBaselinesWrapper(AgentWrapper):

    def __init__(self, agent: BaseAlgorithm):
        self.agent = agent

    # @classmethod
    def initialize(self, weights: list[ArrayLike]) -> AgentWrapper:
        # makes new agent with provided weights
        return self.agent.policy.load_from_vector(weights)

    def get_weights(self) -> list[ArrayLike]:
        # returns weights as a list of (torch|numpy) arrays
        return self.agent.policy.parameters_to_vector()

    def estimate_loss(self, n_steps: int, **kwargs) -> float:
        # evaluates policy for n_steps and returns loss estimate
        pass
