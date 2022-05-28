import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from requests import get
from datetime import datetime, timedelta
import time
import warnings
from tables import NaturalNameWarning

warnings.filterwarnings('ignore', category=NaturalNameWarning)

api = 'https://api.binance.com/api/v3'
db = 'quotes.hdf'


def request(url):
    while True:
        try:
            r = get(url)
            if r.status_code == 200:
                return r
            time.sleep(0.5)
        except Exception as E:
            print(url, E)
            time.sleep(np.random.randint(1, 10))


def getSymbols():
    symbols = []
    js = request(f'{api}/exchangeInfo').json()
    for symbol in js['symbols']:
        symbols.append(symbol['baseAsset'] + symbol['quoteAsset'])
    return sorted(symbols)


def _OHLC(tm, ticker, interval):
    url = f'{api}/klines?&symbol={ticker}&interval={interval}&startTime={int(tm.timestamp() * 1000)}'
    df = pd.DataFrame(request(url).json())
    if len(df) > 0:
        df = df.set_index(df[0].astype('M8[ms]'))[[1, 2, 3, 4, 5]].apply(pd.to_numeric)
        df.index.name = 'DateTime'
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df


def getOHLC(ticker, start_time, end_time, minutes=5):
    print(ticker, end=':')
    TM = pd.date_range(start_time + timedelta(minutes=minutes), end_time, freq=f'{500*minutes}min')
    interval = f'{minutes}m' if minutes < 60 else f'{minutes//60}h' if minutes < 1440 else f'{minutes//1440}d'
    L = Pool(24).map(partial(_OHLC, ticker=ticker, interval=interval), TM)
    if L is not None and len(L) > 0 and L[0] is not None:
        print(len(L), end=' ')
        return pd.concat(L).sort_index()


def storeOHLC(ticker, start_time, end_time, minutes=5):
    OHLC = getOHLC(ticker, start_time, end_time, minutes)
    OHLC.to_hdf(db, key=ticker, complevel=5)


def updateOHLC(ticker, minutes=5):
    try:
        OHLC = pd.read_hdf(db, key=ticker)
    except:
        OHLC = pd.DataFrame()

    if len(OHLC) > 0:
        tm = OHLC.index.max()
    else:
        tm = pd.to_datetime('2022-01-01')

    P = getOHLC(ticker, tm, datetime.now(), minutes)
    if P is not None and len(P) > 0:
        OHLC = pd.concat((OHLC, P))
        OHLC.to_hdf(db, key=ticker, complevel=5)
