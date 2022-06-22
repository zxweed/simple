import numpy as np
from numba import njit
from numba.typed import List
from numba.types import int64, float64, Tuple
from simple.types import TTrade, TPairTrade
from numpy.typing import NDArray


fp32 = np.float32
default_fee = 0.05
default_delay = 1000

signal_type = Tuple((int64, int64, float64, float64))
trade_type = Tuple((int64, int64, float64, float64, int64, int64, float64, float64, int64))


@njit(nogil=True)
def backtestMarket(ts, A, B, signal, threshold, delay=default_delay, maxpos=1) -> List[trade_type]:
    """Fast&simple vectorized backtester for market orders"""

    buys = List.empty_list(signal_type)
    sells = List.empty_list(signal_type)
    trades = List.empty_list(trade_type)
    pos: int = 0

    for i in range(len(ts) - 1):
        delta_pos: int = 0

        # Conditions for open position
        if signal[i] > threshold:
            delta_pos = min(maxpos - pos, 1)
        elif signal[i] < -threshold:
            delta_pos = -min(maxpos + pos, 1)

        # Find delayed price index for open position
        k = i + 1
        if delta_pos > 0:
            while k < len(ts) - 1 and A[i] == A[k] and ts[k] - ts[i] < delay:
                k += 1

        elif delta_pos < 0:
            while k < len(ts) - 1 and B[i] == B[k] and ts[k] - ts[i] < delay:
                k += 1

        # There can be more than one position (until maxpos reached)
        for _ in range(abs(delta_pos)):
            midprice = (A[k] + B[k]) / 2
            if delta_pos > 0:
                buys.append((k, ts[k], A[k], midprice))
            elif delta_pos < 0:
                sells.append((k, ts[k], B[k], midprice))

            if len(sells) > 0 and len(buys) > 0:
                buy = buys.pop(0)
                sell = sells.pop(0)
                if delta_pos < 0:
                    trades.append((*buy, *sell, -delta_pos))
                else:
                    trades.append((*sell, *buy, -delta_pos))

        pos += delta_pos

    return trades


def usInt(ts) -> np.ndarray:
    """Convert time series to microseconds"""

    if ts.dtype == np.dtype('<M8[us]'):
        return ts.astype(np.int64)
    elif ts.dtype == np.dtype('<M8[ns]'):
        return ts.astype(np.int64)//1000
    else:
        return ts


def npTrades(trades: List) -> NDArray[TPairTrade]:
    """Converts trades from limit-backtester to structured array"""

    return np.array(trades, dtype=TPairTrade).view(np.recarray)


def npBacktestMarket(ts, A, B, signal, threshold, delay=default_delay, maxpos=1) -> NDArray[TPairTrade]:
    """Converts trades from the IOC-backtester to structured array"""

    return npTrades(backtestMarket(usInt(ts), A, B, signal, threshold, delay=delay, maxpos=maxpos))


@njit(nogil=True)
def backtestLimit(T: NDArray[TTrade], qA: NDArray[float], qB: NDArray[float]) -> List[trade_type]:
    """Vectorized backtester for limit order strategies"""

    buys = List.empty_list(signal_type)
    sells = List.empty_list(signal_type)
    trades = List.empty_list(trade_type)
    ts = T.DT.view(np.int64)
    pos: int = 0

    for i in range(len(ts) - 1):
        price = T.Price[i]

        if price > qA[i]:
            delta_pos = -min(pos + 1, 1)
        elif price < qB[i]:
            delta_pos = min(1 - pos, 1)
        else:
            delta_pos = 0

        k = i + 1
        if delta_pos > 0:
            buys.append((k, ts[k], qB[k], qB[k]))
        elif delta_pos < 0:
            sells.append((k, ts[k], qA[k], qA[k]))

        if len(sells) > 0 and len(buys) > 0:
            buy = buys.pop(0)
            sell = sells.pop(0)
            if delta_pos < 0:
                trades.append((*buy, *sell, -delta_pos))
            else:
                trades.append((*sell, *buy, -delta_pos))

        pos += delta_pos

    return trades


def npBacktestLimit(T: NDArray[TTrade], qA: NDArray[float], qB: NDArray[float]) -> NDArray[TPairTrade]:
    """Converts trades from the limit-backtester to structured array"""

    return npTrades(backtestLimit(T, qA, qB))


def getProfit(trades: NDArray[TPairTrade], fee_percent=default_fee, inversed: bool = False) -> NDArray[TProfit]:
    """Returns profit of trades"""

    P = np.zeros(len(trades), dtype=TProfit).view(np.recarray)
    if inversed:
        P.RawPnL = (1/trades.Price0 - 1/trades.Price1) * trades.Size * 1000
        P.MidPnL = (1/trades.MidPrice0 - 1/trades.MidPrice1) * trades.Size * 1000
        P.Fee = fee_percent/100 * abs(trades.Size) * (1/trades.Price1 + 1/trades.Price0) * 1000
    else:
        P.RawPnL = (trades.Price1 - trades.Price0) * trades.Size
        P.MidPnL = (trades.MidPrice1 - trades.MidPrice0) * trades.Size
        P.Fee = fee_percent/100 * (trades.Price1 + trades.Price0) * abs(trades.Size)

    P.Profit = P.RawPnL - P.Fee
    P.Index = trades.X1
    P.DT = trades.T1.astype('M8[us]')

    return P


def getLong(trades: NDArray[TPairTrade]) -> dict:
    """Returns long trades only in dictionary format"""

    LongEntry = trades[['X0', 'Price0']][trades.Size > 0]
    LongExit = trades[['X1', 'Price1']][trades.Size < 0]
    return {'x': np.concatenate((LongEntry.X0, LongExit.X1)),
            'y': np.concatenate((LongEntry.Price0, LongExit.Price1))}


def getShort(trades: NDArray[TPairTrade]) -> dict:
    """Returns short trades only in dictionary format"""

    ShortEntry = trades[['X0', 'Price0']][trades.Size < 0]
    ShortExit = trades[['X1', 'Price1']][trades.Size > 0]
    return {'x': np.concatenate((ShortEntry.X0, ShortExit.X1)),
            'y': np.concatenate((ShortEntry.Price0, ShortExit.Price1))}
