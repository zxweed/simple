import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from requests import get

api = 'https://api.binance.com/api/v3/klines'
db = 'quotes.hdf'


def _OHLC(tm, ticker, interval):
    url = f'{api}?&symbol={ticker}&interval={interval}&startTime={int(tm.timestamp() * 1000)}'
    df = pd.DataFrame(get(url).json())
    if len(df) > 0:
        df = df.set_index(df[0].astype('M8[ms]'))[[1, 2, 3, 4, 5]].apply(pd.to_numeric)
        df.index.name = 'DateTime'
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df


def getOHLC(ticker, start_time, end_time, minutes=5):
    TM = pd.date_range(start_time, end_time, freq=f'{500*minutes}min')
    return pd.concat(Pool(64).map(partial(_OHLC, ticker=ticker, interval=f'{minutes}m'), TM)).sort_index()


def storeOHLC(ticker, start_time, end_time, minutes=5):
    OHLC = getOHLC(ticker, start_time, end_time, minutes)
    OHLC.to_hdf(db, key=ticker, complevel=5)
