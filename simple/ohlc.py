import numpy as np
from numba import njit
from numpy.typing import NDArray
from simple.types import TTrade, TDebounce, TOHLC


@njit(nogil=True)
def resampleVolume(T: NDArray[TTrade], threshold: int, OHLC: NDArray[TOHLC]) -> int:
    t = c = 0
    volume = 0

    while t < len(T):
        open = high = low = T.PriceA[t]
        buy = sell = 0
        while t < len(T) and volume < threshold:
            volume += np.abs(T.VolumeA[t])
            if T.PriceA[t] > high:
                high = T.PriceA[t]
            if T.PriceA[t] < low:
                low = T.PriceA[t]
            t += 1
            if T.VolumeA[t] > 0:
                buy += T.VolumeA[t]
            else:
                sell += T.VolumeA[t]

        OHLC['DT'][c] = T.DateTimeA[t]
        OHLC['Open'][c] = open
        OHLC['High'][c] = high
        OHLC['Low'][c] = low
        OHLC['Close'][c] = T.PriceA[t]
        OHLC['Volume'][c] = volume
        OHLC['Buy'][c] = buy
        OHLC['Sell'][c] = sell
        c += 1
        volume = 0

    return c


def getStepPrice(PriceA: NDArray[float]) -> float:
    X = np.around(sorted(np.unique(np.abs(np.diff(PriceA)))), 8)
    stepPrice = X[X > 0][0]
    return stepPrice


@njit(nogil=True)
def midPrice(T: NDArray[TTrade], stepPrice: float) -> NDArray[float]:
    dest = np.zeros_like(T.PriceA)

    for t in range(len(T)):
        dest[t] = T.PriceA[t] - stepPrice / 2 if T.VolumeA[t] > 0 else T.PriceA[t] + stepPrice / 2
    return dest


@njit(nogil=True)
def _isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Backported from https://github.com/numba/numba/pull/7067, until issue is resolved."""

    if np.isnan(x) and np.isnan(y):
        return equal_nan
    elif np.isinf(x) and np.isinf(y):
        return (x > 0) == (y > 0)
    elif np.isinf(x) or np.isinf(y):
        return False
    else:
        return abs(x - y) <= atol + rtol * abs(y)


@njit(nogil=True)
def resampleDebounce(MidA: NDArray[float], T: NDArray[TTrade], DebA: NDArray[TDebounce]) -> int:
    t = c = 0
    DebA.Price[c] = MidA[t]
    DebA.DT[c] = T.DateTimeA[t]

    while t < len(MidA):
        while t < len(MidA) and _isclose(MidA[t], DebA.Price[c]):
            DebA.Count[c] += 1
            DebA.Volume[c] += np.abs(T.VolumeA[t])
            if T.VolumeA[t] > 0:
                DebA.BuyCount[c] += 1
                DebA.BuySize[c] += T.VolumeA[t]
            else:
                DebA.SellCount[c] += 1
                DebA.SellSize[c] += T.VolumeA[t]

            DebA.Duration[c] = T.DateTimeA[t] - DebA.DT[c]
            t += 1

        DebA.DT[c] = T.DateTimeA[t]
        c += 1
        if t < len(MidA):   # prepare the next candle if available
            DebA.Price[c] = MidA[t]
            DebA.DT[c] = T.DateTimeA[t]
            DebA.Index[c] = c

    return c


def debounce(T: NDArray[TTrade]) -> NDArray[TDebounce]:
    """Drops bounce trades (that not change best bidask prices)"""

    stepPrice = getStepPrice(T.PriceA)
    MidA = midPrice(T, stepPrice)
    DebA = np.zeros(len(MidA), dtype=TDebounce).view(np.recarray)
    c = resampleDebounce(MidA, T, DebA)
    return np.resize(DebA, c).view(np.recarray)


@njit(nogil=True)
def resampleRenko(P: np.array, step: int = 1) -> int:
    k = t = 0
    price = P[k]
    low = np.trunc(price / step) * step
    high = low + step
    result = []
    while t < len(P):
        while t < len(P) and low <= P[t] < high:
            t += 1

        price = P[t] if t < len(P) else P[t - 1]
        delta = price - P[k]
        if delta > 0:
            result.append((k, t - 1, low, high))  # TODO: indexes must be stored as int32
        else:
            result.append((k, t - 1, high, low))
        low = np.trunc(price / step) * step
        high = low + step
        k = t

    return result


def renko(T: NDArray[TTrade], step: int = 1) -> np.array:
    stepPrice = getStepPrice(T.PriceA)
    MidA = midPrice(T, stepPrice)
    RenkoL = resampleRenko(MidA, step)
    return RenkoL


def ohlcVolume(T: NDArray[TTrade], threshold: int) -> np.array:
    OHLC = np.zeros(len(T), dtype=TOHLC)
    c = resampleVolume(T, threshold, OHLC)
    OHLC.resize(c, refcheck=False)
    return OHLC.view(np.recarray)


def tickDebounce(T: NDArray[TTrade]) -> np.array:
    Result = np.zeros(len(T), dtype=float)
    c = resampleDebounce(T, Result)
    Result.resize(c, refcheck=False)
    return Result.view(np.recarray)


def tickVolume(T: NDArray[TTrade], threshold: int) -> np.array:
    OHLC = np.zeros(len(T), dtype=TOHLC)
    c = resampleVolume(T, threshold, OHLC)
    OHLC.resize(c, refcheck=False)
    return OHLC[['DT', 'Close']].view(np.recarray)


@njit
def npJoin(S1, S2: np.array) -> int:
    """
    Returns indexes for joining two timeseries

    :param S1 - Indexes of first timeseries
    :param S2 - Indexed of second timeseries
    """

    j = 0
    Idx = np.zeros(len(S1), dtype=np.int32)
    for s in range(len(S1)):
        while j < len(S2) - 1 and S1[s] >= S2[j]:
            j += 1
        Idx[s] = j - 1

    return Idx
