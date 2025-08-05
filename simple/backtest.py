import pandas as pd
import numpy as np
from numpy.typing import NDArray

from numba import njit
from numba.typed import List
from numba.types import int64, float64, Tuple
from sklearn.metrics import mean_squared_error, r2_score

from .types import TTrade, TPairTrade, TProfit, TBidAskDT, TOHLC
from .pretty import pmap


fp32 = np.float32
default_fee = 0.015
default_delay = 1000

signal_type = Tuple((int64, int64, float64, float64))
trade_type = Tuple((int64, int64, float64, float64, int64, int64, float64, float64, int64))


@njit(nogil=True)
def backtestMarket(ts, A, B, signal, threshold, maxpos=1, hold=None) -> List[trade_type]:
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

        # Check for position hold if specified
        if hold is not None:
            last_ts = buys[-1][1] if len(buys) > 0 else sells[-1][1] if len(sells) > 0 else 0
            if last_ts > 0 and ts[i] > last_ts + hold:
                delta_pos = -pos

        # Find delayed price index for open position
        k = i + 1
        if delta_pos > 0:
            while k < len(ts) - 1 and A[i] == A[k] and ts[k] - ts[i] < default_delay:
                k += 1

        elif delta_pos < 0:
            while k < len(ts) - 1 and B[i] == B[k] and ts[k] - ts[i] < default_delay:
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


def asInt(ts) -> np.ndarray:
    """Convert time series to microseconds"""

    if ts.dtype == np.dtype('M8[us]'):
        return ts.astype(np.int64)
    elif ts.dtype == np.dtype('M8[ms]'):
        return ts.astype(np.int64) * 1000
    elif ts.dtype == np.dtype('M8[ns]'):
        return ts.astype(np.int64)//1000
    else:
        return ts


def npTrades(trades: List) -> NDArray[TPairTrade]:
    """Converts trades from limit-backtester to structured array"""

    return np.array(trades, dtype=TPairTrade).view(np.recarray)


def npBacktestMarket(T: NDArray[TBidAskDT], signal: NDArray[np.float64], threshold: float,
                     maxpos: int = 1, hold: int = None) -> NDArray[TPairTrade]:
    """The numpy wrapper for taker orders backtester"""

    trades = backtestMarket(asInt(T['DateTime']), T['Ask'], T['Bid'], signal, threshold, maxpos=maxpos, hold=hold)
    return npTrades(trades)


@njit(nogil=True)
def backtestLimit(ts, high, low, qA, qB, maxpos=1) -> List[trade_type]:
    """Vectorized backtester for limit order strategies"""

    buys = List.empty_list(signal_type)
    sells = List.empty_list(signal_type)
    trades = List.empty_list(trade_type)
    pos: int = 0

    for i in range(1, len(ts) - 1):
        # Conditions for position open (up to the specified maxpos size)
        if high[i] > qA[i]:   # sell
            delta_pos = -min(pos + maxpos, 1)
        elif low[i] < qB[i]:  # buy
            delta_pos = min(maxpos - pos, 1)
        else:
            delta_pos = 0

        k = i - 1   # limit order are executed by previous price, not the future
        if delta_pos > 0:
            buys.append((i, ts[i], qB[k], qB[k]))
        elif delta_pos < 0:
            sells.append((i, ts[i], qA[k], qA[k]))

        if len(sells) > 0 and len(buys) > 0:
            buy = buys.pop(0)
            sell = sells.pop(0)
            if delta_pos < 0:
                trades.append((*buy, *sell, -delta_pos))
            else:
                trades.append((*sell, *buy, -delta_pos))

        pos += delta_pos

    return trades


def npBacktestLimit(T: NDArray[TTrade], qA: NDArray[np.float64], qB: NDArray[np.float64]) -> NDArray[TPairTrade]:
    """Returns trades from the limit-backtester as structured array"""

    ts, high, low = asInt(T['DateTime']), T['Price'], T['Price']
    return npTrades(backtestLimit(ts, high, low, qA, qB))

def npBacktestLimitOHLC(C: NDArray[TOHLC], qA: NDArray[np.float64], qB: NDArray[np.float64]) -> NDArray[TPairTrade]:
    """Returns trades from the limit-backtester as structured array"""

    ts, high, low = asInt(C['DateTime']), C['High'], C['Low']
    return npTrades(backtestLimit(ts, high, low, qA, qB))


def npOHLC(C) -> NDArray[TOHLC]:
    """
    Converts an DataFrame of OHLC (with Open, High, Low, Close fields) to numpy structured array 

    Args:
        C (ndarray): An array of records.

    Returns:
        ndarray: An array of OHLC records.
    """
    if 'DateTime' in C.columns:
        dt = 'DateTime'
    elif 'timestamp' in C.columns:
        dt = 'timestamp'
    return np.core.records.fromarrays([C[dt], C.Open, C.High, C.Low, C.Close], dtype=TOHLC)


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
    P.DateTime = trades.T1.astype('M8[us]')

    return P


def getProfitDict(trades: NDArray[TPairTrade], fee_percent=default_fee, inversed: bool = False) -> dict:
    """Returns profit values for trades in dictionary form"""

    P: NDArray[TProfit] = getProfit(trades, fee_percent, inversed)
    Profit = P['RawPnL'] - P['Fee']
    return {
        'Profit': Profit.sum(),
        'Count': len(P),
        'PRatio': (Profit > 0).mean() if len(Profit) > 0 else 0,
        'AvgMid': P['MidPnL'].mean() if len(P) > 0 else 0,
        'RawPnL': P['RawPnL'].sum() if len(P) > 0 else 0,
        'Fee': P['Fee'].sum() if len(P) > 0 else 0,
        'MidPnL': P['MidPnL'].sum() if len(P) > 0 else 0,
        'Sharpe': P['Profit'].sum() / P['Profit'].std() if len(P) > 1 else 0
    }


def pdThresholdMarket(T: NDArray[TBidAskDT], signal, maxpos=1, inversed=False, parallel=True) -> pd.DataFrame:
    """Parallel evaluation of thresholds*signals by 2D-grid"""

    TS = asInt(T['DateTime'])

    @njit(nogil=True)
    def internalProfit(param):
        """Calculates profit metrics"""

        if len(param) == 3:
            level, index, threshold = param
            trades = backtestMarket(TS, T['Ask'], T['Bid'], signal[level], threshold, maxpos)
        else:
            index, threshold = param
            trades = backtestMarket(TS, T['Ask'], T['Bid'], signal, threshold, maxpos)

        if inversed:
            rawPnL = sum([(1 / t[2] - 1 / t[6]) * t[8] for t in trades]) * 1000
            midPnL = sum([(1 / t[3] - 1 / t[7]) * t[8] for t in trades]) * 1000
            fee = default_fee / 100 * sum([abs(t[8] / t[2] + t[8] / t[6]) for t in trades]) * 1000
        else:
            rawPnL = sum([(t[6] - t[2]) * t[8] for t in trades])
            midPnL = sum([(t[7] - t[3]) * t[8] for t in trades])
            fee = default_fee / 100 * sum([(t[2] + t[6]) * abs(t[8]) for t in trades])
        return rawPnL, midPnL, rawPnL - fee, fee, len(trades), trades

    # create parameter grid
    if len(signal.shape) == 1:
        Thresholds = np.linspace(0, np.percentile(np.abs(signal), 99.98), 100)
        Param = [(index, threshold) for index, threshold in enumerate(Thresholds)]
        prefix = ['Index', 'Threshold']

    elif len(signal.shape) == 2:
        Thresholds = [np.linspace(0, np.percentile(np.abs(y), 99.98), 100) for y in signal]
        Levels = range(len(signal))
        Param = [(level, index, threshold) for level, thresholds in zip(Levels, Thresholds) for index, threshold in enumerate(thresholds)]
        prefix = ['Level', 'Index', 'Threshold']

    else:
        print('The signal must be 1D or 2D numpy array')
        return

    X = pmap(internalProfit, Param, require='sharedmem') if parallel else map(internalProfit, Param)

    # result as DataFrame
    F = pd.DataFrame(Param).join(pd.DataFrame(X), rsuffix='_')
    F.columns = prefix + ['Raw', 'Ideal', 'Profit', 'Fee', 'TradesCnt', 'Trades']
    return F


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


def pdAccuracy(X: NDArray, Y: NDArray) -> float:
    """Directional accuracy"""
    Q = pd.DataFrame(np.sign(X) == np.sign(Y)).value_counts()
    return np.round(Q[True] / Q.sum() * 100, 3)


def getMetrics(y_true: NDArray, y_pred: NDArray) -> tuple:
    """Calculates RMSE, Correlation and Directional Accuracy metrics"""

    rmse = np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 6)
    correlation = np.round(np.corrcoef(y_true, y_pred)[0, 1] * 100, 3)
    accuracy = pdAccuracy(y_true, y_pred)
    rsquared = np.round(r2_score(y_true, y_pred) * 100, 3)
    return rmse, correlation, accuracy, rsquared


def dictMetrics(y_true: NDArray, y_pred: NDArray, sep: int) -> dict:
    """Returns dictionary with the statistical metrics"""

    T0, T1 = y_true[:sep], y_true[sep:]
    Y0, Y1 = y_pred[:sep], y_pred[sep:]

    rmse0, corr0, acc0, rsquared0 = getMetrics(T0, Y0)
    rmse1, corr1, acc1, rsquared1 = getMetrics(T1, Y1)
    return {
        'RMSE(t)': rmse0, 'Corr(t)': corr0, 'Acc(t)': acc0, 'R²(t)' : rsquared0,
        'RMSE(v)': rmse1, 'Corr(v)': corr1, 'Acc(v)': acc1, 'R²(v)' : rsquared1
    }


def npDeals(Trades: NDArray[TPairTrade]) -> NDArray[TTrade]:
    """Returns one-sided deals from paired trades"""
    Enter = Trades[['T0', 'Price0', 'Size']]
    Exit = np.rec.fromarrays([Trades['T1'], Trades['Price1'], -Trades['Size']], dtype=TTrade)
    Enter.dtype.names = Exit.dtype.names

    Deals = np.concatenate([Enter, Exit])
    Deals.sort(order=['DateTime'])
    return Deals.view(np.recarray)


@njit(nogil=True)
def npPositions(Spreads: NDArray[TBidAskDT], Deals: NDArray[TTrade]) -> NDArray[np.float32]:
    """Returns position curve from tick trades"""

    P = np.zeros(len(Spreads))
    for i in range(len(Deals)):
        tm = Deals[i]['DateTime']
        k = np.searchsorted(Spreads['DateTime'], tm)
        P[k] += Deals[i]['Size']
    return P.cumsum()


@njit(nogil=True)
def npEquity(Deals: NDArray[TTrade], Spread: NDArray[TBidAskDT] = None, fee=default_fee) -> NDArray[np.float32]:
    if Spread is not None:
        Eq = np.zeros(len(Spread))
        k = money = pos = 0
        for i in range(len(Spread)):
            while k < len(Deals) and Deals[k].DateTime <= Spread[i].DateTime:
                pos += Deals[k].Size
                t = Deals[k].Size * Deals[k].Price
                money -= t + abs(t) * fee/100
                k += 1
            Eq[i] = money + pos * (Spread[i].BidPrice + Spread[i].AskPrice) / 2

    # trades-only equity line
    else:
        Eq = np.cumsum(Deals.Size) * Deals.Price - np.cumsum(Deals.Size * Deals.Price)
        Fee = np.cumsum(np.abs(Deals.Size) * Deals.Price * fee/100)
        Eq -= Fee

    return Eq
