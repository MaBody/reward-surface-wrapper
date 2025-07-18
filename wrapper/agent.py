from abc import ABC
import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Self
from pathlib import Path
from wrapper import utils
import itertools


class AgentWrapper(ABC):
    @classmethod
    def initialize(cls, weights: list[ArrayLike]) -> Self:
        # makes new agent with provided weights
        raise NotImplementedError()

    def get_weights(self) -> list[ArrayLike]:
        # returns weights as a list of (torch|numpy) arrays
        raise NotImplementedError()

    def estimate_reward(self, **kwargs: dict) -> float:
        # evaluates policy for n_steps (and other params) and returns loss estimate
        raise NotImplementedError()

    def compute_surface(
        self,
        rollout_kwargs: dict,
        directions: tuple[list[ArrayLike] | Path] = None,
        direction_dir: Path = None,
        grid_size: int = 25,
    ):
        """direction_dir is only used if directions is None (optional, to store new directions)."""

        assert grid_size % 2 == 1, "Grid size must be odd."

        self.rollout_kwargs = rollout_kwargs

        # Initialize directions
        self.directions = None
        if directions is not None:
            if isinstance(directions[0], (str, Path)):
                self.directions = [utils.readz(handle) for handle in directions]
            else:
                self.directions = directions
        else:
            self.directions = utils.initialize_directions(
                self.get_weights(), direction_dir
            )

        # Initialize offsets for the grid
        self.offsets = utils.initialize_offsets(grid_size)

        # Compute loss on (offsets, offsets) grid
        points = self._compute_points_loop()

        return points

    def _compute_points_loop(self) -> ArrayLike:
        losses = []
        for offset_pair in itertools.product(self.offsets, self.offsets):
            loss = self._compute_point(offset_pair)
            losses.append(loss)
        n = len(self.offsets)
        losses = np.array(losses).reshape((n, n))
        return losses

    def _compute_point(self, offset_pair: tuple[float]) -> float:
        weights = []
        for p, p_dir_0, p_dir_1 in zip(self.get_weights(), *self.directions):
            print(offset_pair)
            p_alt = p + p_dir_0 * offset_pair[0] + p_dir_1 * offset_pair[1]
            weights.append(p_alt)
        agent_alt = self.initialize(weights)

        return agent_alt.estimate_reward(**self.rollout_kwargs)
