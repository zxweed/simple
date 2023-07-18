import numpy as np
from numba import njit
from numpy.typing import NDArray
from simple.types import TTrade, TDebounce, TOHLC


@njit(nogil=True)
def resampleVolume(T: NDArray[TTrade], threshold: int, OHLC: NDArray[TOHLC]) -> int:
    t = c = size = count = 0

    while t < len(T):
        open = high = low = T.Price[t]
        buySize = sellSize = buyCount = sellCount = 0
        while t < len(T) and size < threshold:
            size += np.abs(T.Size[t])
            count += 1
            if T.Price[t] > high:
                high = T.Price[t]
            if T.Price[t] < low:
                low = T.Price[t]
            t += 1
            if T.Size[t] > 0:
                buySize += T.Size[t]
                buyCount += 1
            else:
                sellSize += T.Size[t]
                sellCount += 1

        OHLC['DateTime'][c] = T.DateTime[t]
        OHLC['Open'][c] = open
        OHLC['High'][c] = high
        OHLC['Low'][c] = low
        OHLC['Close'][c] = T.Price[t]

        OHLC['Size'][c] = size
        OHLC['BuySize'][c] = buySize
        OHLC['SellSize'][c] = sellSize

        OHLC['Count'][c] = count
        OHLC['BuyCount'][c] = buyCount
        OHLC['SellCount'][c] = sellCount
        c += 1
        size = count = 0

    return c-1


def getStepPrice(Price: NDArray[float]) -> float:
    X = np.around(sorted(np.unique(np.abs(np.diff(Price)))), 8)
    stepPrice = X[X > 0][0]
    return stepPrice


@njit(nogil=True)
def midPrice(T: NDArray[TTrade], stepPrice: float) -> NDArray[float]:
    dest = np.zeros_like(T.Price)

    for t in range(len(T)):
        dest[t] = T.Price[t] - stepPrice / 2 if T.Size[t] > 0 else T.Price[t] + stepPrice / 2
    return dest


@njit(nogil=True)
def midPrice2(T: NDArray[TTrade]) -> NDArray[float]:
    dest = np.zeros_like(T.Price)
    bid = ask = 0
    for t in range(len(T)):
        if T.Size[t] > 0:
            ask = T.Price[t]
        else:
            bid = T.Price[t]

        dest[t] = (bid + ask) / 2 if bid > 0 and ask > 0 else np.nan
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
    DebA.DateTime[c] = T.DateTime[t]

    while t < len(MidA):
        while t < len(MidA) and _isclose(MidA[t], DebA.Price[c]):
            DebA.Count[c] += 1
            DebA.Size[c] += T.Size[t]
            if T.Size[t] > 0:
                DebA.BuyCount[c] += 1
                DebA.BuySize[c] += T.Size[t]
            else:
                DebA.SellCount[c] += 1
                DebA.SellSize[c] += T.Size[t]

            DebA.Duration[c] = T.DateTime[t] - DebA.DateTime[c]
            t += 1

        DebA.DateTime[c] = T.DateTime[t]
        c += 1
        if t < len(MidA):   # prepare the next candle if available
            DebA.Price[c] = MidA[t]
            DebA.DateTime[c] = T.DateTime[t]
            DebA.Index[c] = c

    return c


def debounce(T: NDArray[TTrade], step_price: float = None) -> NDArray[TDebounce]:
    """Drops bounce trades (that not change best bidask prices)"""

    if step_price is None:
        step_price = getStepPrice(T['Price'])

    MidA = midPrice(T, step_price)
    DebA = np.zeros(len(MidA), dtype=TDebounce).view(np.recarray)
    c = resampleDebounce(MidA, T, DebA)
    return np.resize(DebA, c).view(np.recarray)


#@njit(nogil=True)
def resampleRenko2(T: NDArray[TTrade], DebA: NDArray[TDebounce], step: float = 1) -> int:
    k = t = c = 0
    price = T.Price[k]
    low = np.trunc(price / step) * step
    high = low + step
    result = []
    DebA.Price[c] = T.Price[t]
    DebA.DateTime[c] = T.DateTime[t]

    while t < len(T):
        # iterate over ticks in range
        while t < len(T) and low <= T.Price[t] < high:
            DebA.Count[c] += 1
            DebA.Size[c] += T.Size[t]
            if T.Size[t] > 0:
                DebA.BuyCount[c] += 1
                DebA.BuySize[c] += T.Size[t]
            else:
                DebA.SellCount[c] += 1
                DebA.SellSize[c] += T.Size[t]

            DebA.Duration[c] = T.DateTime[t] - DebA.DateTime[c]
            t += 1

        if c > 907305 or t > 907305:
            print(c, t)

        # Price changed more than range - create new renko box
        if t < len(DebA):
            DebA.DateTime[c] = T.DateTime[t]
            c += 1
            DebA.Price[c] = T.Price[t]
            DebA.DateTime[c] = T.DateTime[t]
            DebA.Index[c] = c

        price = T.Price[t] if t < len(T.Price) else T.Price[t - 1]
        delta = price - T.Price[k]
        if delta > 0:
            result.append((k, t - 1, low, high))  # TODO: indexes must be stored as int32
        else:
            result.append((k, t - 1, high, low))
        low = np.trunc(price / step) * step
        high = low + step
        k = t

    return result



@njit(nogil=True)
def resampleRenko(P: np.array, step: float = 1) -> int:
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
    stepPrice = getStepPrice(T.Price)
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
    return OHLC[['DateTime', 'Close']].view(np.recarray)


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
