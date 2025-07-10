import os
import torch
import argparse
import json
import numpy as np
from pathlib import Path
from numpy.typing import ArrayLike
from wrapper.agent import AgentWrapper
import itertools


def initialize_directions(
    weights: list[ArrayLike],
    out_dir: Path = None,
) -> tuple[list[ArrayLike]]:
    direction_1 = [filter_normalize(p) for p in weights]
    direction_2 = [filter_normalize(p) for p in weights]
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.savez((out_dir / "direction_1.npz"), *direction_1)
        np.savez((out_dir / "direction_2.npz"), *direction_2)
    return direction_1, direction_2


def filter_normalize(param: list[ArrayLike]):
    # TODO: verify with loss landscapes code
    ndims = len(param.shape)
    if ndims == 1 or ndims == 0:
        # don't do any random direction for scalars
        return np.zeros_like(param)
    elif ndims == 2:
        direction = np.random.normal(size=param.shape)
        direction /= np.sqrt(np.sum(np.square(direction), axis=0, keepdims=True))
        direction *= np.sqrt(np.sum(np.square(param), axis=0, keepdims=True))
        return direction
    elif ndims == 4:
        direction = np.random.normal(size=param.shape)
        direction /= np.sqrt(
            np.sum(np.square(direction), axis=(0, 1, 2), keepdims=True)
        )
        direction *= np.sqrt(np.sum(np.square(param), axis=(0, 1, 2), keepdims=True))
        return direction
    else:
        raise AssertionError(
            f"only 1, 2, 4 dimentional filters allowed, got {param.shape}"
        )


def initialize_offsets(grid_size: int):
    n = grid_size // 2
    offsets = (np.arange(0, grid_size) - n) / n
    return offsets


def compute_point(
    agent: AgentWrapper,
    directions: tuple[list[ArrayLike]],
    offsets: tuple[float],
    rollout_kwargs: dict,
) -> float:
    weights = []
    for p, p_directions in zip(agent.get_weights(), zip(directions)):
        p_alt = p + p_directions[0] * offsets[0] + p_directions[1] * offsets[1]
        weights.append(p_alt)
    agent_alt = AgentWrapper.initialize(weights)

    return agent_alt.estimate_loss(rollout_kwargs)


def compute_points_loop(
    agent: AgentWrapper,
    directions: tuple[list[ArrayLike]],
    offsets: ArrayLike,
    rollout_kwargs: dict,
) -> ArrayLike:
    losses = []
    for offset_tuple in itertools.product(offsets, offsets):
        loss = compute_point(agent, directions, offset_tuple, rollout_kwargs)
        losses.append(loss)

    losses = np.array(losses).reshape((len(offsets), len(offsets)))
    return losses


def readz(fname):
    outvecs = []
    with np.load(fname) as data:
        for item in data:
            outvecs.append(data[item])
    return outvecs


def compute_surface(
    agent: AgentWrapper,
    directions: tuple[list[ArrayLike] | Path] = None,
    out_dir: Path = None,
    grid_size: int = 25,
    rollout_kwargs: dict = {"n_steps": 1},
):
    assert grid_size % 2 == 1, "Grid size must be odd."
    assert "n_steps" in rollout_kwargs, "Parameter 'n_steps' is required."
    # Initialize directions
    _directions = None
    if directions is not None:
        if isinstance(directions[0], Path):
            _directions = [readz(handle) for handle in directions]
        else:
            _directions = directions
    else:
        _directions = initialize_directions(agent.get_weights(), out_dir)

    # Initialize offsets for the grid
    offsets = initialize_offsets(grid_size)

    # Compute loss on (offsets, offsets) grid
    points = compute_points_loop(agent, _directions, offsets, rollout_kwargs)
    return points
