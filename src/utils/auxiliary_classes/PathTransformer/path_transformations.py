from functools import partial

import numpy as np


def numeraire(s_type: str, p: np.ndarray) -> np.ndarray:
    """
    Helper function for path transformations

    :param p:       Slice of path dimension
    :param s_type:  Standardisation type
    :return:        Standardised path
    """
    if s_type == "initial":
        return p / p[0]
    elif s_type == "L1":
        return p / np.sum(p)
    elif s_type == "max":
        return p / max(p)
    elif s_type == "min":
        return p / min(p)
    elif s_type == "mean-var":
        return (p - np.mean(p)) / np.std(p)
    elif s_type == "min-max":
        return (p - min(p)) / (max(p) - min(p))
    elif s_type == "none":
        return p


def standardise_path_transform(path: np.ndarray, s_type="initial") -> np.ndarray:
    """
    Standardises path by a given value

    :param path:   Path to be standardised
    :param s_type: Type of standardisation to complete
    :return:       Standardised path
    """

    d = path.shape[1]
    num = partial(numeraire, s_type)

    for di in range(1, d):
        path[:, di] = num(path[:, di])

    return path


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Returns lead_lag transform of N x d path, where N is the number of stream elements and d is the dimension.

    Note by convention we assume the first dimension is the time component which is ignored.

    :param path:        ARRAY, N x d, path to have transform taken of
    :return:            ARRAY, 2N-1 x 2d, lead-lag transformed path
    """
    # Assume path is N x d
    length, dimension = path.shape
    dimension        -= 1
    lead_lag_length   = 2 * length - 1
    lead_lag_dim      = 2 * dimension

    # Instantiate lead-lag path
    res = np.zeros((lead_lag_length, lead_lag_dim))

    # Assign points
    for j in range(lead_lag_length):
        i = j // 2
        if j % 2 == 0:
            res[j, :] = np.concatenate([path[i, 1:], path[i, 1:]])
        else:
            res[j, :] = np.concatenate([path[i, 1:], path[i + 1, 1:]])

    return res


def time_normalisation_transform(path: np.ndarray) -> np.ndarray:
    """
    Given a path ((t_0, x_0), ... (t_n, x_n)), returns ((0, x_0), (1/n, x_1), ..., (1, x_n)).

    :param path:    Path in R^d, where the first component corresponds to the time index.
    :return:        Time-normalised path.
    """

    path[:, 0] = (path[:, 0] - path[0, 0])/(path[-1, 0] - path[0, 0])

    return path


def time_difference_transform(path: np.ndarray) -> np.ndarray:
    """
    Given a path ((t_0, x_0), ... (t_n, x_n)), returns ((0, x_0), (t_1-t_0, x_1), ..., (t_n-t_{n-1}, x_n)).

    :param path:    Path in R^d, where the first component corresponds to the time index.
    :return:        Time-difference transform
    """

    times = path[:, 0].copy()
    path[0, 0] = 0.
    path[1:, 0] = np.diff(times)

    return path


def translation_transform(path: np.ndarray, all_channels=False) -> np.ndarray:
    """
    Subtracts the starting point for the given set of paths.

    :param path:            Path in R^d. First component includes time.
    :param all_channels:    Whether to translate time too
    :return:                Paths with subtracted initial point
    """

    ind_ = 0 if all_channels else 1

    res = path.copy()
    res[:, ind_:] -= path[0, ind_:]

    return res


def difference_transform(path: np.ndarray) -> np.ndarray:
    """
    For a given path, returns the difference path in its state space coordinates

    :param path:    Path to be differenced
    :return:        Differenced path
    """

    d = path.shape[1]

    for di in range(1, d):
        path[:, di] = np.diff(path[:, di], prepend=[path[0, di]])

    return path


def returns_transform(path: np.ndarray) -> np.ndarray:
    """
    For a given path, returns the real returns path in its state space coordinates

    :param path:    Path
    :return:        Path of real returns path
    """

    d = path.shape[1]

    for di in range(1, d):
        path[1:, di] = np.diff(path[:, di])/path[:-1, di]
        path[0, di]  = 0.

    return path


def invisibility_transform(path: np.ndarray) -> np.ndarray:
    """
    Preserves the absolute value of a stream of data after taking its signature.

    :param path:    Path to take invisibility transform of.
    :return:        Invisibility-transformed path.
    """

    n = path.shape[0]

    itran = np.concatenate(([0.], [1.]*n)).reshape(-1, 1)
    e_path = np.concatenate(([path[0]], path), axis=0)

    return np.concatenate((e_path, itran), axis=1)


def increment_transform(path: np.ndarray) -> np.ndarray:
    """
    Returns the path of absolute increments Y associated to a path X. This is given by Y_0 = X_0 and
    Y_i = Y_{i-1} + |X_i - X_{i-1}|.

    :param path:    Path to calculate increment transform of.
    :return:        Transformed path.
    """

    d = path.shape[1]

    for di in range(1, d):
        path[:, di] = np.concatenate(([0.], np.abs(np.diff(path[:, di])).cumsum())) + path[0, di]

    return path


def cumulant_transform(path: np.ndarray) -> np.ndarray:
    """
    Returns the cumulant transform associated to a poth.

    :param path:  Path to be transformed
    :return:      Path of cumulants with time
    """

    d = path.shape[1]

    for di in range(1, d):
        path[:, di] = path[:, di].cumsum()

    #zeros = np.zeros(d).reshape(-1, d)

    #return np.concatenate((zeros, path))
    return path


def realised_variance_transform(path: np.ndarray) -> np.ndarray:
    """
    Returns the total realised variance associated to each component of a path

    :param path:   Path to be transformed
    :return:       (Path, rv) pair
    """

    n, d = path.shape

    res = np.zeros((n, 1 + 2*(d-1)))

    for di in range(1, d):
        rv          = np.zeros(n)
        log_returns = np.diff(np.log(path[:, di]))
        rv[1:]      = np.power(log_returns, 2).cumsum()*7*252/np.arange(1, n)

        res[:, 2*di-1] = path[:, di]
        res[:, 2*di]   = rv

    res[:, 0] = path[:, 0]

    return res


def squared_log_returns_transform(path: np.ndarray, s_type="none") -> np.ndarray:
    """
    Returns the path of squared log returns associated to a path

    :param path:    Input path
    :param s_type:  Standardisation type
    :return:        Path with attached squared log returns
    """

    n, d = path.shape
    res = np.zeros((n, 1 + 2*(d-1)))
    num = partial(numeraire, s_type)

    for di in range(1, d):
        log_returns = np.zeros(n)
        log_returns[1:] = np.diff(np.log(path[:, di]))**2

        res[:, 2*di-1] = path[:, di]
        res[:, 2*di]   = num(log_returns)

    res[:, 0] = path[:, 0]

    return res


def ewma_volatility_transform(path: np.ndarray, lambd=0.5, s_type="none") -> np.ndarray:
    """
    Computes EWMA volatility transform on path data

    :param path:    Path to be transformed
    :param lambd:   Smoothing parameter
    :param s_type:  Standardisation type
    :return:        Path with (standardised) EWMA of squared log returns
    """
    n, d = path.shape
    res = np.zeros((n, 1 + 2*(d-1)))

    for di in range(1, d):
        # Get path of squared log returns
        log_returns = np.zeros(n)
        log_returns[1:] = np.diff(np.log(path[:, di]))**2

        # Recursive formula for the path
        res[:, 2*di-1] = path[:, di]
        res[0, 2*di]   = log_returns[0]

        for j, lr in enumerate(log_returns[1:]):
            res[j+1, 2*di] = lambd*res[j, 2*di] + (1-lambd)*lr

    # Append time
    res[:, 0] = path[:, 0]

    return res


def scaling_transform(path: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    n, d = path.shape

    assert len(sigmas) == d-1, "Not enough scalings for each dimension of the path object."

    res = np.zeros((n, d))

    for di in range(1, d):
        res[:, di] = path[:, di]*sigmas[di-1]

    res[:, 0] = path[:, 0].copy()

    return res
