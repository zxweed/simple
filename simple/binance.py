import pandas as pd
import numpy as np
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial
from requests import get
from datetime import datetime, timedelta
import time

api = 'https://api.binance.com/api/v3/klines?&symbol={ticker}&interval={interval}&startTime={startTime}'
hist_api = 'https://data.binance.vision/data/futures/um/monthly/klines/{ticker}/{frame}/{ticker}-{frame}-{month}.zip'


def _HistOHLC(month, ticker, frame, close_only=False):
    x = pd.read_csv(hist_api.format(month=month, ticker=ticker, frame=frame), header=None,
                    names=['DT', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseDT', 'BaseVolume',
                           'TradeCount', 'TakerBase', 'TakerQuote', 'Ignore'])

    x.DT = x.DT.astype('M8[ms]')
    x.CloseDT = x.CloseDT.astype('M8[ms]')
    x.set_index('DT', inplace=True)
    x.name = ticker
    return x[['Close']].rename(columns={'Close': ticker}) if close_only else x


def getHistMonth(start_date, end_date, ticker, frame, close_only=False):
    M = [s.strftime('%Y-%m') for s in pd.date_range(start_date, end_date, freq='MS')]
    X = pd.concat(ThreadPool(16).map(partial(_HistOHLC, ticker=ticker, frame=frame, close_only=close_only), M))
    return X


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
    url = api.format(ticker=ticker, interval=interval, startTime=int(tm.timestamp() * 1000))
    df = pd.DataFrame(request(url).json())
    if len(df) > 0:
        df = df.set_index(df[0].astype('M8[ms]'))[[1, 2, 3, 4, 5]].apply(pd.to_numeric)
        df.index.name = 'DateTime'
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df


def getOHLC(tm, ticker):
    url = api.format(ticker=ticker, interval=interval, startTime=int(tm.timestamp() * 1000))
    df = pd.DataFrame(requests.get(url).json())
    if len(df) > 0:
        df = df.set_index(df[0].astype('M8[ms]'))[[1, 2, 3, 4, 5]].apply(pd.to_numeric)
        df.index.name = 'DateTime'
        df.columns = pd.MultiIndex.from_product(([ticker], ['Open', 'High', 'Low', 'Close', 'Volume']))
        return df


def getClose(tm, ticker):
    url = api.format(ticker=ticker, interval=interval, startTime=int(tm.timestamp() * 1000))
    df = pd.DataFrame(requests.get(url).json())
    if len(df) > 0:
        df = df.set_index(df[0].astype('M8[ms]'))[[4]].apply(pd.to_numeric)
        df.index.name = 'DateTime'
        df.columns = [ticker]
        return df


def getHist(startDate, endDate, ticker):
    TM = pd.date_range(startDate, endDate, freq='2500min')
    
    with closing(ThreadPool(16)) as P:
        return pd.concat(P.map(partial(getClose, ticker=ticker), TM))