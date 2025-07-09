from abc import ABC
from numpy.typing import ArrayLike


class AgentWrapper(ABC):
    @classmethod
    def initialize(cls, weights: list[ArrayLike]) -> AgentWrapper:
        # makes new agent with provided weights
        raise NotImplementedError()

    def get_weights(self) -> list[ArrayLike]:
        # returns weights as a list of (torch|numpy) arrays
        raise NotImplementedError()

    def estimate_loss(self, n_steps: int) -> float:
        # evaluates policy for n_steps and returns loss estimate
        raise NotImplementedError()
