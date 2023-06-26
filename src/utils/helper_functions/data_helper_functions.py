from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats


def date_transformer(x: str) -> pd.Timestamp:
    """
    Transforms date column of local data to pd.Timestamp.

    :param x:   Date to be transformed
    :return:    Timestamp object
    """

    acceptable_formats = ["%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"]
    try:
        return pd.Timestamp(datetime.strptime(x, "%d/%m/%Y %H:%M"))
    except ValueError:
        if ("AM" in x) or ("PM" in x):
            incl = "AM" if "AM" in x else "PM"
            val = 12 if incl == "PM" else 0
            ind = x.index(incl)
            this_time = str(int(x[ind-3:ind-1]) + val)

            if this_time == "24":
                this_time = "00"

            string = x[:ind-3] + this_time + ":00"

            for formats in acceptable_formats:
                try:
                    return pd.Timestamp(datetime.strptime(string, formats))
                except ValueError:
                    pass
            return pd.NaT


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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    s = np.array(s)
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema
