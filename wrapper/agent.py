from abc import ABC
from numpy.typing import ArrayLike
from typing import Self


class AgentWrapper(ABC):
    @classmethod
    def initialize(cls, weights: list[ArrayLike]) -> Self:
        # makes new agent with provided weights
        raise NotImplementedError()

    def get_weights(self) -> list[ArrayLike]:
        # returns weights as a list of (torch|numpy) arrays
        raise NotImplementedError()

    def estimate_loss(self, **kwargs) -> float:
        # evaluates policy for n_steps (and other params) and returns loss estimate
        raise NotImplementedError()
