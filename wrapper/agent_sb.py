from abc import ABC
from numpy.typing import ArrayLike
from wrapper.agent import AgentWrapper


class StableBaselinesWrapper(AgentWrapper):
    @classmethod
    def initialize(cls, weights: list[ArrayLike]) -> AgentWrapper:
        # makes new agent with provided weights
        pass

    def get_weights(self) -> list[ArrayLike]:
        # returns weights as a list of (torch|numpy) arrays
        pass

    def estimate_loss(self, n_steps: int, **kwargs) -> float:
        # evaluates policy for n_steps and returns loss estimate
        pass
