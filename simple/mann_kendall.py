"""ported from pyMannKendall: a python package for non parametric Mann Kendall family of trend tests"""
import numpy as np
from numba import njit


@njit(nogil=True)
def __preprocessing(x):
    x = np.asarray(x).astype(np.float64)
    dim = x.ndim

    if dim == 1:
        c = 1

    else:
        print('Please check your dataset.')

    return x, c


# Missing Values Analysis
def __missing_values_analysis(x, method='skip'):
    if method.lower() == 'skip':
        if x.ndim == 1:
            x = x[~np.isnan(x)]

        else:
            x = x[~np.isnan(x).any(axis=1)]

    n = len(x)

    return x, n


# vectorization approach to calculate mk score, S
@njit(nogil=True)
def __mk_score(x, n):
    s = 0

    demo = np.ones(n) 
    for k in range(n-1):
        s = s + np.sum(demo[k+1:n][x[k+1:n] > x[k]]) - np.sum(demo[k+1:n][x[k+1:n] < x[k]])

    return s


@njit(nogil=True)
def original_series(X, period):
    result = np.zeros_like(X)
    for i in range(period, len(X)):
        s = __mk_score(X[i-period:i], period)
        Tau = s / (0.5 * period * (period - 1))
        result[i] = Tau

    return result
