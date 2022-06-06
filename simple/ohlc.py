import numpy as np
from numba import njit

# candle record structure
TOHLC = np.dtype([('DT', '<M8[us]'),
                  ('Open', float),
                  ('High', float),
                  ('Low', float),
                  ('Close', float),
                  ('Volume', float),
                  ('Buy', int),
                  ('Sell', int)])


@njit(nogil=True)
def resampleVolume(T: np.array, threshold: int, OHLC: np.array) -> int:
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


def getStepPrice(PriceA: np.array) -> float:
    X = np.around(sorted(np.unique(np.abs(np.diff(PriceA)))), 8)
    stepPrice = X[X > 0][0]
    return stepPrice


@njit(nogil=True)
def midPrice(T: np.array, stepPrice: float) -> np.array:
    dest = np.zeros_like(T.PriceA)

    for t in range(len(T)):
        dest[t] = T.PriceA[t] - stepPrice / 2 if T.VolumeA[t] > 0 else T.PriceA[t] + stepPrice / 2
    return dest


@njit(nogil=True)
def resampleDebounce(dest: np.array) -> int:
    t = c = 0
    while t < len(dest):
        while t < len(dest) and dest[t] == dest[c]:
            t += 1

        price = dest[t] if t < len(dest) else dest[t-1]
        c += 1
        dest[c] = price

    return c + 1  # c is the last index of debounced value, so return length


@njit(nogil=True)
def resampleRenko(P: np.array, step: int = 1) -> int:
    k = t = 0
    price = P[k]
    low = np.trunc(price / step) * step
    high = low + step
    result = []
    while t < len(P):
        while t < len(P) and P[t] >= low and P[t] < high:
            t += 1

        price = P[t] if t < len(P) else P[t - 1]
        delta = price - P[k]
        if delta > 0:
            result.append((k, t-1, low, high))  # TODO: indexes must be stored as int32
        else:
            result.append((k, t-1, high, low))
        low = np.trunc(price / step) * step
        high = low + step
        k = t

    return result


def debounce(T: np.array) -> np.array:
    """Drops bounce trades (that not change best bidask prices)"""
    stepPrice = getStepPrice(T.PriceA)
    MidA = midPrice(T, stepPrice)
    c = resampleDebounce(MidA)
    return np.resize(MidA, c).view(np.recarray)


def renko(T: np.array, step: int = 1) -> np.array:
    stepPrice = getStepPrice(T.PriceA)
    MidA = midPrice(T, stepPrice)
    RenkoL = resampleRenko(MidA, step)
    return np.array(RenkoL).view(np.recarray)


def ohlcVolume(T: np.array, threshold: int) -> np.array:
    OHLC = np.zeros(len(T), dtype=TOHLC)
    c = resampleVolume(T, threshold, OHLC)
    OHLC.resize(c, refcheck=False)
    return OHLC.view(np.recarray)


def tickDebounce(T: np.array) -> np.array:
    Result = np.zeros(len(T), dtype=float)
    c = resampleDebounce(T, Result)
    Result.resize(c, refcheck=False)
    return Result.view(np.recarray)


def tickVolume(T: np.array, threshold: int) -> np.array:
    OHLC = np.zeros(len(T), dtype=TOHLC)
    c = resampleVolume(T, threshold, OHLC)
    OHLC.resize(c, refcheck=False)
    return OHLC[['DT', 'Close']].view(np.recarray)