# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed
from simple.types import TTrade
from numpy.typing import NDArray


@np.vectorize
@njit(nogil=True)
def symlog(x):
    """Symmetrical log transform with a linear section from -1 to 1"""
    if x > 1:
        return np.log(x) + 1
    elif x < -1:
        return -np.log(-x) - 1
    else:
        return x


def WPrice(A, B, vA, vB, L) -> np.ndarray:
    """Weighted average price"""
    return ((A[:L].astype(float) * vA[:L]).sum(0) + (B[:L].astype(float) * vB[:L]).sum(0)) / (vB[:L] + vA[:L]).sum(0)


def IWPrice(A, B, vA, vB, L) -> np.ndarray:
    """Inverse weighted average price"""
    return ((A[:L] / vA[:L]).sum(0) + (B[:L] / vB[:L]).sum(0)) / (1 / vA[:L] + 1 / vB[:L]).sum(0)


def Ratio(vA, vB, L) -> np.ndarray:
    """Imbalance ratio"""
    return (vA[:L].sum(0) - vB[:L].sum(0)) / (vA[:L].sum(0) + vB[:L].sum(0))


@njit(nogil=True)
def vPIN(T: NDArray[TTrade], period: int = 1000) -> NDArray[float]:
    """Some version of Volume-Synchronized Probability of Informed Trading - paper by Easley, Lopez de Prado, O’Hara"""

    A = B = 0
    resultA = np.zeros(len(T), dtype=np.float32)
    resultA[:period] = np.nan

    # during init stage we can't calculate anything, but cumulate the values
    for i in range(period):
        if T.Size[i] < 0:
            A += -T.Size[i]
        else:
            B += T.Size[i]

    for i in range(period, len(T), 1):
        k = i - period
        if T.Size[i] < 0:
            A += -T.Size[i]
        else:
            B += T.Size[i]

        if T.Size[k] < 0:
            A -= -T.Size[k]
        else:
            B -= T.Size[k]

        resultA[i] = (B - A) / (B + A) * 100

    return resultA


@njit(nogil=True)
def cPIN(T: NDArray[TTrade], period: int = 1000) -> NDArray[float]:
    """Tick-synchronized buy/sell count imbalance ratio"""

    A = B = 0
    resultA = np.zeros(len(T), dtype=np.float32)

    # during init stage we can't calculate anything, but cumulate the values
    resultA[:period] = np.nan
    for i in range(period):
        if T.Size[i] < 0:
            A += 1
        else:
            B += 1

    for i in range(period, len(T), 1):
        k = i - period
        if T.Size[i] < 0:
            A += 1
        else:
            B += 1

        if T.Size[k] < 0:
            A -= 1
        else:
            B -= 1

        resultA[i] = (B - A) / (B + A) * 100

    return resultA


@njit(nogil=True)
def tickSpeed(T: NDArray[TTrade], period: int = 1000) -> NDArray[float]:
    """Tick speed indicator (change of price in dollars per second)"""
    resultA = np.zeros(len(T), dtype=np.float32)
    resultA[:period] = np.nan

    for i in range(period, len(T), 1):
        k = i - period
        t0, t1 = T.DT[k], T.DT[i]
        delta = np.int64(t1 - t0)
        resultA[i] = (T.Price[i] - T.Price[k]) * 1e6 / delta if delta > 0 else 0

    return resultA


@njit(nogil=True)
def vwap(T: NDArray[TTrade], period: int, destA: np.array = None) -> NDArray[float]:
    """Volume Weighted Average Price"""
    turnover = 0.
    size = 0.
    sizeA = np.abs(T.Size)

    if destA is None:
        destA = np.zeros(len(T.Price))

    for i in range(period):
        size += sizeA[i]
        turnover += sizeA[i] * T.Price[i]

    for i in range(period, len(T.Price)):
        k = i - period
        size += sizeA[i] - sizeA[k]
        turnover += sizeA[i] * T.Price[i] - sizeA[k] * T.Price[k]
        destA[i] = turnover / size

    destA[:period] = destA[period]
    return destA


@njit(nogil=True)
def sma(src: np.ndarray, period: int, dest: np.ndarray):
    """A fast SMA detrender that stores the response in the series given by the last parameter, rather than returning it (to save memory)"""
    n: int = 0
    mean: float = 0.

    for i in range(period):
        n += 1
        delta = src[i] - mean
        mean += delta / n
        dest[i] = src[i] - mean

    for i in range(period, len(src)):
        mean += (src[i] - src[i - period]) / period
        dest[i] = src[i] - mean


def PSMA1(src: np.ndarray, period: int) -> np.ndarray:
    """Parallel calculation of average detrenders by column"""

    dest = np.zeros_like(src)
    with Parallel(n_jobs=-1, require='sharedmem') as P:
        SMA = delayed(sma)
        if src.dtype.isbuiltin == 1:
            P(SMA(src[idx], period, dest[idx]) for idx in range(src.shape[0]))
        else:
            P(SMA(src[col], period, dest[col]) for col in src.dtype.names)
            dest.dtype = np.dtype(list(('MA{0}({1})'.format(period, desc[0]), desc[1]) for desc in src.dtype.descr))

    return dest


def getColumn(src_list: [], name: str) -> np.ndarray:
    for src in src_list:
        if name in src.dtype.names:
            return src[name]


def PSMA(src_list: [], periods: [int], dest=None) -> np.ndarray:
    # create all source columns
    dtype = []
    src = None
    for src in src_list:
        dtype = dtype + src.dtype.descr

    # append dest type with all result columns
    cmd = []
    for period in periods:
        for src in src_list:
            for desc in src.dtype.descr:
                src_name = desc[0]
                dest_name = 'MA{0}({1})'.format(period, src_name)
                cmd.append((src_name, period, dest_name))
                dtype.append((dest_name, desc[1]))

    # copy source data
    if dest is None:
        dest = np.zeros(len(src), dtype=dtype)
        for src in src_list:
            S = list(src.dtype.names)
            dest[S] = src[S]
    else:
        dest.dtype = dtype

    # parallel calculation of moving averages
    with Parallel(n_jobs=12, require='sharedmem') as P:
        SMA = delayed(sma)
        P(SMA(getColumn(src_list, src_name), period, dest[dest_name]) for src_name, period, dest_name in cmd)

    return dest


def vx(X):
    return X.view(X.dtype.descr[0][1]).reshape(len(X), len(X.dtype.names))


def RGG(V):
    """Returns the series with colors for the signal"""
    return np.where(V < 0, 'red', np.where(V > 0, 'green', 'gray'))


def pdMA(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """Moving average detrending"""
    M = X - X.rolling(p).mean()
    M.columns = ['MA{}({})'.format(p, c) for c in X.columns]
    return M


def pdEMA(X: pd.DataFrame, p: int) -> pd.DataFrame:
    """Exponential Moving average detrending"""
    M = X - X.ewm(p).mean()
    M.columns = ['EMA{}({})'.format(p, c) for c in X.columns]
    return M


def hurst(X: np.array) -> float:
    """Returns the Hurst Exponent of the time series vector X"""
    lags = range(2, min(100, len(X) - 1))

    # Calculate the array of the variances of the lagged differences
    tau = [np.std(np.subtract(X[lag:], X[:-lag])) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    fit = np.polyfit(np.log(lags), np.log(tau), 1)
    return fit[0]
