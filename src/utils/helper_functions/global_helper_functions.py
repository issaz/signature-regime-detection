import os
from pathlib import Path
from itertools import cycle, islice
from typing import Callable

import numpy as np
from tqdm import tqdm


def get_project_root() -> Path:
    """
    Returns path root of project

    :return: Path object
    """
    return get_source_root().parent


def get_source_root() -> Path:
    """
    Returns /src root of project

    :return: Path object
    """
    return Path(__file__).parent.parent.parent


def mkdir(filepath):
    """
    Makes a directory corresponding to :param filepath: if it does not exist.

    :param filepath:    Filepath to make directory of.
    :return:            None
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def map_over_matrix(v: np.ndarray, func: Callable, **kwargs) -> np.ndarray:
    """
    Maps the given function over all pairs in the vector v.

    :param v:       Vector of indexes to map over
    :param func:    Callable to map over
    :param kwargs:  Named arguments for callable
    :return:        Matrix of f(i,j)s.
    """
    n = v.shape[0]
    res = np.zeros((n, n))

    for i, vi in enumerate(v):
        for j, vj in enumerate(v[i:]):
            this_result = func(vi, vj, **kwargs)
            res[i, j] = this_result

    return res


def map_over_matrix_vector(v: np.ndarray, odim: int, func: Callable, **kwargs) -> np.ndarray:
    """
    Maps the given function over all pairs in the vector v.

    :param v:       Vector of indexes to map over
    :param odim:    Output dimension
    :param func:    Callable to map over
    :param kwargs:  Named arguments for callable
    :return:        Matrix of f(i,j)s.
    """
    n = v.shape[0]
    res = np.zeros((n, n, odim))

    for i, vi in enumerate(v):
        for j, vj in enumerate(v[i:]):
            this_result = func(v[i], v[j], **kwargs)
            res[i, j] = this_result
            res[j, i] = this_result

    return res
