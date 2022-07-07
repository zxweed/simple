# -*- coding: utf-8 -*-
import pandas as pd
import urllib.request
from datetime import datetime, timedelta
import time
import requests
from io import StringIO

parser = lambda x: datetime.strptime(x, '%Y%m%d %H:%M:%S')

# time periods for resampling
periods = {'tick': 1, '1min': 2, '5min': 3, '10min': 4, '15min': 5, '30min': 6, 
           'hour': 7, 'daily': 8, 'week': 9, 'month': 10}

url_dict = 'https://www.finam.ru/cache/N72Hgd54/icharts/icharts.js'
url_addr = 'http://export.finam.ru/data.txt?'
hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0)'}


def getOHLC(em, market, period, start_date, end_date) -> pd.DataFrame:
    """Request OHLC data from FINAM server"""

    # http url request
    params = urllib.parse.urlencode([
        ('market', market), ('em', em),
        ('df', start_date.day), ('mf', start_date.month - 1), ('yf', start_date.year),
        ('dt', end_date.day), ('mt', end_date.month - 1), ('yt', end_date.year),
        ('p', period), ('dtf', 1), ('tmf', 3), ('sep', 3), ('sep2', 1), ('datf', 5),
        ('at', 1),   # include headers=1
        ('MSOR', 0)  # 1 for close time, 0 for open time
    ])

    print(url_addr + params, start_date, end_date)
    resp = requests.get(url_addr + params, headers=hdr)

    OHLC = pd.read_csv(StringIO(resp.text), header=0, sep=';', parse_dates=[[0, 1]], date_parser=parser, index_col=0)
    OHLC.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    OHLC.index.names = ['DT']
    return OHLC


def getTick(em, market, start_date, end_date) -> pd.DataFrame:
    """Request tick data from FINAM server"""

    day = start_date
    T = pd.DataFrame()
    while day <= end_date:
        # http url request
        params = urllib.parse.urlencode([
            ('market', market), ('em', em),
            ('df', day.day), ('mf', day.month - 1), ('yf', day.year),
            ('dt', day.day), ('mt', day.month - 1), ('yt', day.year),
            ('p', 1), ('dtf', 1), ('tmf', 3), ('sep', 3),
            ('sep2', 1), ('datf', 7), ('at', 1)
        ])

        try:
            # print(url_addr + params, day)
            # print(day, end=' ')
            data = pd.read_csv(url_addr + params, header=0, sep=';', parse_dates=[[1, 2]], date_parser=parser, index_col=0)
            data.columns = ['Ticker', 'Last', 'Volume']
            del data['Ticker']
            data.index.names = ['DT']
            T = T.append(data)

        except Exception as E:
            print('E', end=' ')

        time.sleep(1)
        day += timedelta(days=1)

    return T


def getDict() -> pd.DataFrame:
    """Create the dictionary with ticker codes"""

    r = urllib.request.urlopen(url_dict)
    dict = r.readlines()

    # extract IDs
    str_id = dict[0].decode('cp1251')
    IDs = str_id[str_id.find('[') + 1 : str_id.find(']')].split(',')

    # extract names
    str_name = dict[1].decode('cp1251')
    P1 = str_name.find("[\'") + 2
    P2 = str_name.find("\']")
    Names = str_name[P1:P2].split('\',\'')

    # extract tickers
    str_code = dict[2].decode('cp1251')
    P1 = str_code.find("[\'") + 2
    P2 = str_code.find("\']")
    Tickers = str_code[P1:P2].split('\',\'')

    # market codes
    str_market = dict[3].decode('cp1251')
    P1 = str_market.find('[') + 1
    P2 = str_market.find(']')
    Markets = str_market[P1:P2].split(',')

    # now create dictionary
    D = pd.DataFrame({'Market': Markets, 'EM': IDs, 'Ticker': Tickers, 'Name': Names})
    D = D[D.Market != "-1"]
    D.EM = D.EM.astype(int)
    D.Market = D.Market.astype(int)
    D.set_index('EM', inplace=True)

    return D
