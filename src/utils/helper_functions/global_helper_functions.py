import os
from itertools import cycle, islice
from pathlib import Path


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
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
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
