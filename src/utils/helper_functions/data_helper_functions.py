import numpy as np


def get_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Returns the vector of log-returns

    :param prices:  Array. Prices associated to an asset
    :return:        Array. Vector of log-returns with prepended 0.
    """

    log_prices = np.log(prices)

    return np.diff(log_prices, prepend=log_prices[0])


def reweighter(num_elements: int, factor: int) -> np.ndarray:
    """
    Takes a number of elements in a vector and splits then progressively via the factor parameter.
    Higher factors mean more recent observations receive more weight.

    :param num_elements:    Number of elements to reweight
    :param factor:          Factor parameter
    :return:                Vector of indexes corresponding to reweight
    """
    x = np.arange(num_elements)

    if factor <= 1:
        return x

    split_pct = 1.0 - 1.0 / factor
    res, new, old = [], [], []
    curr_index = num_elements // factor
    i = 1

    while curr_index > 0:
        old, new = np.split(x, [int(split_pct * x.shape[0])])
        res += list(np.repeat(old, i))

        x = new
        curr_index //= factor
        i += 1

    return np.array(res + list(np.repeat(new, i)))


def ema(s, n):
    s    = np.array(s)
    ema_ = []
    j    = 1

    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema_.append(sma)

    ema_.append(( (s[n] - sma) * multiplier) + sma)

    for i in s[n+1:]:
        tmp = ((i - ema_[j]) * multiplier) + ema_[j]
        j = j + 1
        ema_.append(tmp)

    return ema_
