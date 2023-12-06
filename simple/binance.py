import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool
from functools import partial
from requests import get
import time
from contextlib import closing

api_endpoint = 'https://api.binance.com/api/v3'
api = api_endpoint + '/klines?&symbol={ticker}&interval={interval}&startTime={startTime}'

hist_endpoint = 'https://data.binance.vision/data'
hist_fut_api = hist_endpoint + '/futures/um/monthly/klines/{ticker}/{frame}/{ticker}-{frame}-{month}.zip'
hist_spot_api = hist_endpoint + '/spot/monthly/klines/{ticker}/{frame}/{ticker}-{frame}-{month}.zip'


def _HistOHLC(month, ticker, frame, close_only=False, spot=False):
    try:
        if spot:
            url = hist_spot_api.format(month=month, ticker=ticker, frame=frame)
        else:
            url = hist_fut_api.format(month=month, ticker=ticker, frame=frame)

        header = None if month <= '2022-03' else 'infer'
        x = pd.read_csv(url, header=header)
        x.columns = [
            'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseDT',
            'QuoteVolume', 'Count', 'TakerBuyVolume', 'TakerBuyQuoteVolume', 'Ignore']

        x.DateTime = x.DateTime.astype('M8[ms]')
        x.CloseDT = x.CloseDT.astype('M8[ms]')
        x.set_index('DateTime', inplace=True)
        x.name = ticker
        return x[['Close']].rename(columns={'Close': ticker}) if close_only else x
    except Exception as E:
        print(ticker, url, E)


def getHistMonth(start_date, end_date, ticker, frame, close_only=False, spot=False):
    months = [s.strftime('%Y-%m') for s in pd.date_range(start_date, end_date, freq='MS')]
    lst = ThreadPool(16).map(partial(_HistOHLC, ticker=ticker, frame=frame, close_only=close_only, spot=spot), months)
    lst = [item for item in lst if item is not None]
    if len(lst) > 0:
        return pd.concat(lst)


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
    url = f'{api_endpoint}/exchangeInfo'
    js = request(url).json()
    return sorted([symbol['symbol'] for symbol in js['symbols'] if symbol['status'] == 'TRADING'])


def _OHLC(tm, ticker, interval):
    url = api.format(ticker=ticker, interval=interval, startTime=int(tm.timestamp() * 1000))
    df = pd.DataFrame(request(url).json())
    if len(df) > 0:
        df = df.set_index(df[0].astype('M8[ms]'))[[1, 2, 3, 4, 5]].apply(pd.to_numeric)
        df.index.name = 'DateTime'
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df


def getOHLC(tm, ticker, interval):
    url = api.format(ticker=ticker, interval=interval, startTime=int(tm.timestamp() * 1000))
    df = pd.DataFrame(get(url).json())
    if len(df) > 0:
        df = df.set_index(df[0].astype('M8[ms]'))[[1, 2, 3, 4, 5]].apply(pd.to_numeric)
        df.index.name = 'DateTime'
        df.columns = pd.MultiIndex.from_product(([ticker], ['Open', 'High', 'Low', 'Close', 'Volume']))
        return df


def getClose(tm, ticker, interval, startTime):
    url = api.format(ticker=ticker, interval=interval, startTime=int(tm.timestamp() * 1000))
    df = pd.DataFrame(get(url).json())
    if len(df) > 0:
        df = df.set_index(df[0].astype('M8[ms]'))[[4]].apply(pd.to_numeric)
        df.index.name = 'DateTime'
        df.columns = [ticker]
        return df


def getHist(startDate, endDate, ticker):
    TM = pd.date_range(startDate, endDate, freq='2500min')

    with closing(ThreadPool(16)) as P:
        return pd.concat(P.map(partial(getClose, ticker=ticker), TM))