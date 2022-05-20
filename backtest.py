import numpy as np
from typing import List
from numba import njit


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