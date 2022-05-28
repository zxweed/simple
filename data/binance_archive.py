import os
import sys
import time
import urllib
import warnings

import requests
import lxml
import pathlib
import pandas as pd
from functools import partial
from datetime import datetime as dt
from typing import List
from enum import Enum, unique
from glob import glob
from bs4 import BeautifulSoup as BS


# defaults
S3_ARCHIVE_URL = 's3-ap-northeast-1.amazonaws.com'
BINANCE_URL = 'data.binance.vision'
_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9,ru;q=0.8,bg;q=0.7'
    }


@unique
class Source(Enum):
    FUTURES = 'futures'
    SPOT = 'spot'


@unique
class Timeframe(Enum):
    TRADES = 'trades'
    AGG_TRADES = 'aggTrades'
    MINUTE1 = '1m'
    MINUTE3 = '3m'
    MINUTE5 = '5m'
    MINUTE15 = '15m'
    MINUTE30 = '30m'
    HOUR1 = '1h'
    DAY1 = '1d'


def get_all_data_urls_by_instrument(url: str) -> List[str]:
    """ find all data urls by instrument """

    r = requests.get(url, headers=_headers)
    soup = BS(r.text, 'lxml')
    keys = soup.find_all('key')
    return [x.text for x in keys if x.text.endswith('.zip')]


def download_all_data_by_instrument(name: str, storage_dir: str, all_instruments_url: str,
                                    binance_url: str, tf: str, start_dt: str = None):
    """ download all data files by instrument"""

    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    # s3 urls
    link_storage_url = all_instruments_url + f'{name}/{tf + "/" if tf not in ("trades", "aggTrades") else ""}'
    data_urls = get_all_data_urls_by_instrument(link_storage_url)

    # filter out old data
    if start_dt is not None:
        # print(f'filter out data older than {start_dt}')
        data_urls = [i for i in data_urls
                     if dt.strptime(i.split(tf)[-1][1:-4], '%Y-%m') >= dt.strptime(start_dt, '%Y-%m-%d')]

    # no data found
    if len(data_urls) == 0:
        print(f'{name}: no data in {link_storage_url}')
        return

    # download zipped data archive
    for data_url in data_urls:
        link = f"https://{binance_url}/{data_url}"
        fname = data_url.split("/")[-1]

        if not os.path.exists(os.path.join(storage_dir, fname)):
            download_file_by_url(link, f'{storage_dir}/{fname}')
        else:
            # todo: check content-length between local and s3 files
            # print('exist', fname)
            pass


def download_file_by_url(url: str, save_path: str):
    """download file by url"""
    # todo: sync requests -> async httpx ( https://github.com/projectdiscovery/httpx )
    while True:
        try:
            dl_file = urllib.request.urlopen(url)
            length = dl_file.getheader('content-length')
            if length:
                length = int(length)
                blocksize = max(4096, length // 100)

            with open(save_path, 'wb') as out_file:
                dl_progress = 0
                print("\nFile Download: {}".format(save_path))
                while True:
                    buf = dl_file.read(blocksize)
                    if not buf:
                        break
                    dl_progress += len(buf)
                    out_file.write(buf)
                    done = int(50 * dl_progress / length)
                    sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50 - done)))
                    sys.stdout.flush()
        except BaseException as e:
            print(f'save error')
            print(save_path)
            print(url)
            print(e)
            os.remove(save_path)
            time.sleep(5)
            continue
        # success download
        break


def get_all_instrument_names(url: str) -> List[str]:
    """find all exchange's instruments"""

    new_url = url
    output = []
    marker = '&marker=' + url.split('=')[-1]
    last_instrument = ''
    while True:
        if last_instrument != '':
            new_url = url + marker + last_instrument + '/'
        r = requests.get(new_url, headers=_headers)
        soup = BS(r.text, 'lxml')
        keys = soup.find_all('prefix')
        res = [x.text.split('/')[-2] for x in keys[1:]]
        output += res
        if len(res) < 1000:
            break
        else:
            last_instrument = res[-1]
            continue
    return output


def combine_data_to_df(files: List[str], from_date: str = None, to_date: str = None,
                       tf: Timeframe = None) -> pd.DataFrame:
    """ aggragate all files into one dataframe """

    if tf == Timeframe.TRADES:
        col_names = ['TradeId', 'Price', 'Quantity', 'QuoteQty', 'Open_time', 'isBuyerMaker']
    elif tf == Timeframe.AGG_TRADES:
        col_names = ['AggTradeId', 'Price', 'Quantity', 'firstTradeId', 'lastTradeId',
                     'Open_time', 'isBuyerMaker', 'isBestPriceMatch']
    else:
        col_names = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume',
                     'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']

    df_lst = []
    for filename in files:
        df_ = pd.read_csv(filename, names=col_names, header=None, index_col=False)
        df_lst.append(df_)

    full_df = pd.concat(df_lst, ignore_index=True)

    # add datetime index and sort asc
    full_df = full_df.set_index(full_df['Open_time'].astype('M8[ms]'))
    full_df.index.name = 'DateTime'
    full_df.sort_index(ascending=True, inplace=True)

    # filter out by dates
    if from_date is not None:
        full_df = full_df.loc[full_df.index >= dt.strptime(from_date, '%Y-%m-%d')]

    if to_date is not None:
        full_df = full_df.loc[full_df.index <= dt.strptime(to_date, '%Y-%m-%d')]

    # output
    if tf == Timeframe.TRADES:
        return full_df[['TradeId', 'Price', 'Quantity', 'isBuyerMaker']]
    elif tf == Timeframe.AGG_TRADES:
        return full_df[['AggTradeId', 'Price', 'Quantity', 'isBuyerMaker']]
    else:
        return full_df[['Open', 'High', 'Low', 'Close', 'Volume']]


def find_all_files(path: str, suffix: str = 'zip') -> List[str]:
    """ find all files by path"""
    return glob(os.path.join(path, f'*.{suffix}'))


def get_ohlc(symbol: str = 'BTCUSDT', from_date: str = '2022-01-01', to_date: str = '2022-05-01',
             tf: Timeframe = Timeframe.MINUTE1, source_type: Source = Source.SPOT,
             storage_path: str = 'binance_data') -> pd.DataFrame:

    # storage path
    aggregate_period = 'monthly'    # monthly/daily
    current_dir = pathlib.Path(__file__).parent.resolve()
    storage_dir_partial = partial(os.path.join(current_dir, storage_path, source_type.value,
                                               '{name}', aggregate_period, tf.value).format_map)
    # s3 url
    all_instruments_url = f"https://{S3_ARCHIVE_URL}/{BINANCE_URL}?delimiter=/&prefix=data/{source_type.value}/{'um/' if source_type.value=='futures' else ''}{aggregate_period}/{tf.value if tf.value in ('trades', 'aggTrades') else 'klines'}/"

    # download
    storage_dir = storage_dir_partial({'name': symbol})
    download_all_data_by_instrument(symbol, storage_dir, all_instruments_url, BINANCE_URL, tf.value, from_date)
    # print('download completed =============================================')

    # dataframe
    files_ = find_all_files(storage_dir)
    df_ = combine_data_to_df(files_, from_date, to_date, tf)
    return df_


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    pd.set_option('display.expand_frame_repr', False)  # print out all columns

    # SPOT examples
    df1 = get_ohlc('ANCUSDT', '2022-04-01', '2022-05-01', Timeframe.MINUTE1, Source.SPOT)
    print(df1.head(3))

    df2 = get_ohlc('ANCUSDT', '2022-04-01', '2022-05-01', Timeframe.TRADES, Source.SPOT)
    print(df2.head(3))

    df3 = get_ohlc('ANCUSDT', '2022-04-01', '2022-05-01', Timeframe.AGG_TRADES, Source.SPOT)
    print(df3.head(3))

    # FUTURES examples
    df4 = get_ohlc('ANCUSDT', '2022-04-01', '2022-05-01', Timeframe.MINUTE1, Source.FUTURES)
    print(df4.head(3))

    df5 = get_ohlc('ANCUSDT', '2022-04-01', '2022-05-01', Timeframe.TRADES, Source.FUTURES)
    print(df5.head(3))

    df6 = get_ohlc('ANCUSDT', '2022-04-01', '2022-05-01', Timeframe.AGG_TRADES, Source.FUTURES)
    print(df6.head(3))
