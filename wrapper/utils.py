import os
import torch
import argparse
import json
import numpy as np
from pathlib import Path
from numpy.typing import ArrayLike
import itertools


def initialize_directions(
    weights: list[ArrayLike],
    direction_dir: Path = None,
) -> tuple[list[ArrayLike]]:
    direction_1 = [filter_normalize(p) for p in weights]
    direction_2 = [filter_normalize(p) for p in weights]
    if direction_dir is not None:
        os.makedirs(direction_dir, exist_ok=True)
        np.savez((direction_dir / "direction_1.npz"), *direction_1)
        np.savez((direction_dir / "direction_2.npz"), *direction_2)
    return direction_1, direction_2


def filter_normalize(param: ArrayLike):
    # TODO: verify with loss landscapes code
    ndims = len(param.shape)
    if ndims == 1 or ndims == 0:
        # don't do any random direction for scalars
        return torch.zeros_like(param)
    elif ndims == 2:
        direction = torch.normal(0, 1, size=param.shape)
        direction /= torch.sqrt(
            torch.sum(torch.square(direction), axis=0, keepdims=True)
        )
        direction *= torch.sqrt(torch.sum(torch.square(param), axis=0, keepdims=True))
        return direction
    elif ndims == 4:
        direction = torch.normal(0, 1, size=param.shape)
        direction /= torch.sqrt(
            torch.sum(torch.square(direction), axis=(0, 1, 2), keepdims=True)
        )
        direction *= torch.sqrt(
            torch.sum(torch.square(param), axis=(0, 1, 2), keepdims=True)
        )
        return direction
    else:
        raise AssertionError(
            f"only 1, 2, 4 dimentional filters allowed, got {param.shape}"
        )


def initialize_offsets(grid_size: int):
    n = grid_size // 2
    offsets = (np.arange(0, grid_size) - n) / n
    return offsets


def readz(fname):
    outvecs = []
    with np.load(fname) as data:
        for item in data:
            outvecs.append(data[item])
    return outvecs
