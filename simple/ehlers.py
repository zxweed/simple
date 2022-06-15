import numpy as np
from numba import njit


@njit
def supersmoother_fast(source: np.array, period: int) -> np.array:
    a = np.exp(-1.414 * np.pi / period)
    b = 2 * a * np.cos(1.414 * np.pi / period)
    newseries = np.copy(source)
    for i in range(2, source.shape[0]):
        newseries[i] = (1 + a ** 2 - b) / 2 * (source[i] + source[i - 1]) \
                       + b * newseries[i - 1] - a ** 2 * newseries[i - 2]
    return newseries


@njit
def reflex_fast(ssf, period):
    rf = np.full_like(ssf, 0)
    ms = np.full_like(ssf, 0)
    sums = np.full_like(ssf, 0)
    for i in range(ssf.shape[0]):
        if i >= period:
            slope = (ssf[i - period] - ssf[i]) / period
            my_sum = 0
            for t in range(1, period + 1):
                my_sum = my_sum + (ssf[i] + t * slope) - ssf[i - t]
            my_sum /= period
            sums[i] = my_sum

            ms[i] = 0.04 * sums[i] * sums[i] + 0.96 * ms[i - 1]
            if ms[i] > 0:
                rf[i] = sums[i] / np.sqrt(ms[i])
    return rf


def reflex(source: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Reflex indicator by John F. Ehlers

    :param source: np.ndarray
    :param period: int - default: 20
    :return: np.ndarray
    """

    ssf = supersmoother_fast(source, period / 2)
    rf = reflex_fast(ssf, period)

    return rf
