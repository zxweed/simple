import numpy as np
from typing import List
from numba import njit
from numba.typed import List
from numba.types import int64, float64, Tuple


def npTrades(trades: List) -> np.ndarray:
    """Converts trades from the limit-backtester to structured array"""

    TPairTrade = [('X0', int), ('Price0', float), ('X1', int), ('Price1', float),
                  ('Size', float), ('Profit', float), ('Fee', float)]
    return np.array(trades, dtype=TPairTrade).view(np.recarray)


def getLong(trades: np.array):
    LongEntry = trades[['X0', 'Price0']][trades.Size > 0]
    LongExit = trades[['X1', 'Price1']][trades.Size < 0]
    return {'x': np.concatenate((LongEntry.X0, LongExit.X1)),
            'y': np.concatenate((LongEntry.Price0, LongExit.Price1))}


def getShort(trades: np.array):
    ShortEntry = trades[['X0', 'Price0']][trades.Size < 0]
    ShortExit = trades[['X1', 'Price1']][trades.Size > 0]
    return {'x': np.concatenate((ShortEntry.X0, ShortExit.X1)),
            'y': np.concatenate((ShortEntry.Price0, ShortExit.Price1))}


fp32 = np.float32
default_fee = 0.05
default_delay = 1000

signal_type = Tuple((int64, int64, float64, float64))
trade_type = Tuple((int64, int64, float64, float64, int64, int64, float64, float64, int64))


@njit(nogil=True)
def backtestIOC(ts, A, B, signal, threshold, delay=default_delay, maxpos=1) -> list:
    """Fast&simple vectorized backtester w/ IOC orders modeling"""

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

        # IOC check
        k = i + 1
        if delta_pos > 0:
            while k < len(ts) - 1 and A[i] == A[k] and ts[k] - ts[i] < delay:
                k += 1

            if A[k] > A[i] and ts[k] - ts[i] < delay:
                delta_pos = 0

        elif delta_pos < 0:
            while k < len(ts) - 1 and B[i] == B[k] and ts[k] - ts[i] < delay:
                k += 1

            if B[k] < B[i] and ts[k] - ts[i] < delay:
                delta_pos = 0

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


def npBacktestIOC(ts, A, B, signal, threshold, delay=default_delay, maxpos=1) -> np.ndarray:
    """Converts trades from the IOC-backtester to structured array"""

    trades = backtestIOC(usInt(ts), A, B, signal, threshold, delay=delay, maxpos=maxpos)
    TPairTrade = [('X0', np.int64), ('T0', np.int64), ('Price0', float), ('MidPrice0', float),
                  ('X1', np.int64), ('T1', np.int64), ('Price1', float), ('MidPrice1', float),
                  ('Size', float)]

    return np.array(trades, dtype=TPairTrade).view(np.recarray)


@njit(nogil=True)
def backtestLimit(PriceA, qA, qB, fee_percent=default_fee) -> list:
    """Vectorized backtester for limit order strategies"""

    buys = [(int(x), fp32(x)) for x in range(0)]
    sells = [(int(x), fp32(x)) for x in range(0)]
    trades = [(int(x), fp32(x), int(x), fp32(x), int(x), fp32(x), fp32(x)) for x in range(0)]

    pos: int = 0

    for i in range(len(PriceA) - 1):
        price = PriceA[i]

        if price > qA[i]:
            delta_pos = -min(pos + 1, 1)
        elif price < qB[i]:
            delta_pos = min(1 - pos, 1)
        else:
            delta_pos = 0

        k = i + 1
        if delta_pos > 0:
            buys.append((k, qB[k]))
        elif delta_pos < 0:
            sells.append((k, qA[k]))

        if len(sells) > 0 and len(buys) > 0:
            k_buy, buy = buys.pop(0)
            k_sell, sell = sells.pop(0)
            d_rawPnL = sell - buy
            fee = fee_percent / 100 * (sell + buy)
            d_PnL = d_rawPnL - fee
            if delta_pos < 0:
                trades.append((k_buy, buy, k_sell, sell, -delta_pos, d_PnL, fee))
            else:
                trades.append((k_sell, sell, k_buy, buy, -delta_pos, d_PnL, fee))

        pos += delta_pos

    return trades