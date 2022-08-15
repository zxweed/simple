import numpy as np
from numba import njit
from datetime import datetime, date
import MySQLdb
import os, zlib, time, random
from contextlib import closing
from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count
from simple.pretty import tqdmParallel
from simple.types import TTrade


# Base for time format conversion
US = 24 * 60 * 60 * 1000 * 1000
BASE_US = 2209161600000000

# Database connection parameters
host = 'localhost'
username = 'gb_tickdb'
socket = '/run/mysqld/mysqld.sock'

# Types of database structures
TArrayHeader = [('RefCount', np.int32), ('Length', np.int32)]

THeaderMap = [('DT', float), ('Pos', np.int32)]
TSnapRow = [('Price', np.float32), ('Size', np.int32)]


def jobs():
    return min(cpu_count() // 2, 32)


def connect():
    """Enables reconnect"""
    while True:
        try:
            return MySQLdb.connect(host=host, user=username, db=username,
                                   passwd=os.environ['HISTDB_PASSWORD'],
                                   unix_socket=socket)
        except Exception as E:
            print(repr(E), end=' ', flush=True)
            time.sleep(random.randint(1, 5))


def getTickerID(ticker: str) -> int:
    """
    Returns ID by ticker

    :param ticker: ticker
    :return: ID
    """

    with closing(connect()) as db:
        try:
            cur = db.cursor()
            if '@' in ticker:
                T = ticker.split('@')
                cur.execute(f'select TickerID from TradeBoard where Ticker="{T[0]}" and Market="{T[1]}"')
            else:
                cur.execute(f'select TickerID from TradeBoard where Ticker="{ticker}"')
            return cur.fetchone()[0]
        except:
            return None


def iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False


def getWhere(tickerID: any, field: str, startDate: datetime = None, endDate: datetime = None):
    """Returns <where> string condition for date/time parameters"""

    if iterable(tickerID):
        whereTickerID = f'TickerID in ({str(tickerID)[1:-1]})'
    else:
        whereTickerID = f'TickerID = {tickerID}'

    if startDate is not None and endDate is not None:
        where = f'where {whereTickerID} and {field} >= "{startDate}" and {field} < "{endDate}"'

    elif startDate is not None and endDate is None:
        where = f'where {whereTickerID} and {field} >= "{startDate}"'
    elif startDate is None and endDate is not None:
        where = f'where {whereTickerID} and {field} < "{endDate}"'
    else:
        where = f'where {whereTickerID}'

    return where


def npTradeT(tickerID: any, startDate: date = None, endDate: date = None, progress: bool = True) -> np.array:
    """
    Returns array of tick prices

    :param tickerID: ID 
    :param startDate: begin of interval
    :param endDate: end of interval
    """

    with closing(connect()) as db:
        cur = db.cursor()
        whereTickerID = getWhere(tickerID, 'Day', startDate, endDate)
        sql = f'select DateTimeA, LocalTimeA, PriceA, VolumeA, OpenIntA from TradeT {whereTickerID} order by Day'
        cur.execute(sql)

    def fromBuf(f, count):
        if f is not None:
            buf = zlib.decompress(f)
            if (len(buf) - 8) // 8 == count:
                return np.frombuffer(buf, float, count, offset=8)

    TradeT = np.zeros(0, dtype=TTrade)
    for f in tqdm(cur.fetchall()) if progress else cur.fetchall():
        buf = zlib.decompress(f[0])
        count = (len(buf) - 8) // 8

        k = len(TradeT)
        TradeT.resize(k + count, refcheck=False)
        TradeT['DT'][k:] = (np.frombuffer(buf, float, count, offset=8) * US - BASE_US).astype('M8[us]')
        TM = fromBuf(f[1], count)
        TradeT['LocalDT'][k:] = (TM * US - BASE_US).astype('M8[us]') if f[1] is not None and TM is not None else None
        TradeT['Price'][k:] = fromBuf(f[2], count)
        TradeT['Size'][k:] = fromBuf(f[3], count)
        TradeT['OpenInt'][k:] = fromBuf(f[4], count)

    return TradeT.view(np.recarray)


@njit(nogil=True)
def clipFrame(Price: np.array, Size: np.array, nrows: np.array, height: int) -> tuple:
    """
    Fast clip height of orderbook snapshot

    :param Price: Prices array
    :param Size: Sizes array
    :param nrows: Heights array
    :param height: Target height
    """

    L = len(nrows)
    A1 = np.zeros(shape=(height, L), dtype=np.float32)
    B1 = np.zeros(shape=(height, L), dtype=np.float32)
    vA1 = np.zeros(shape=(height, L), dtype=np.int32)
    vB1 = np.zeros(shape=(height, L), dtype=np.int32)

    p1 = 0
    p2 = 0
    for k, h in enumerate(nrows):
        p2 += h
        aPrice = Price[p1:p2]
        aSize = Size[p1:p2]
        p1 = p2

        # look for middle of orderbook
        i = np.searchsorted(aSize, 0)
        k1 = i - height
        k2 = i + height

        # clip height
        A1[:, k] = aPrice[k1:i][::-1]
        B1[:, k] = aPrice[i:k2]
        vA1[:, k] = aSize[k1:i][::-1]
        vB1[:, k] = aSize[i:k2]

    return A1, vA1, B1, vB1


def unpackFrame(f: tuple, height: int = 10) -> tuple:
    """Unpack orderbook frame"""

    # HeaderMap
    buf = zlib.decompress(f[0])
    count = (len(buf) - 8) // 12
    HeaderMap = np.frombuffer(buf, dtype=THeaderMap, count=count, offset=8)
    ts = (HeaderMap['DT'] * US - BASE_US).astype('M8[us]')

    buf = zlib.decompress(f[1])
    C = (np.roll(HeaderMap['Pos'], -1) - HeaderMap['Pos'])
    C[-1:] = len(buf) - HeaderMap['Pos'][-1:]

    S = np.frombuffer(buf, dtype=TSnapRow, count=len(buf) // 8)

    A, vA, B, vB = clipFrame(S['Price'], S['Size'], C // 8, height)
    return ts, A, vA, B, vB


def getFrame(minute: datetime, tickerID: int, height: int = 5) -> tuple:
    """Reads one minute frame from database"""

    with closing(connect()) as db:
        try:
            cur = db.cursor()
            sql = f'select Header, Snap from SnapMin where TickerID={tickerID} and DT = "{minute}" order by DT limit 1'
            cur.execute(sql)
            return unpackFrame(cur.fetchmany(1)[0], height)

        except:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)


def unpackHeader(f: tuple) -> np.array:
    """Unpack header of orderbook frame"""

    # HeaderMap
    buf = zlib.decompress(f[1])
    count = (len(buf) - 8) // 12
    HeaderMap = np.frombuffer(buf, dtype=THeaderMap, count=count, offset=8)

    if len(HeaderMap) > 1:
        ts = (HeaderMap['DT'] * US - BASE_US).astype('M8[us]')
        rows = np.diff(HeaderMap['Pos'] // 8)
        size = rows.sum() + rows[0]
        return size, len(ts), ts, rows
    else:
        return 0, 0, None, None


def getHeader(tickerID: int, minute: datetime) -> tuple:
    """Reads header of orderbook frame"""

    with closing(connect()) as db:
        cur = db.cursor()
        sql = f'select DT, Header from SnapMin where TickerID={tickerID} and DT="{minute}" order by DT limit 1'
        cur.execute(sql)

        for c in cur.fetchmany(1):
            size, cnt, ts, rows = unpackHeader(c)
            return int(c[0].timestamp()), cnt, tickerID, ts, rows


def getMinutesRange(tickerID: any, startDate: datetime = None, endDate: datetime = None) -> np.array:
    """
    Returns the array of all available frames for interval    
    :param tickerID: ID 
    :param startDate: begin of interval
    :param endDate: end of interval
    """

    with closing(connect()) as db:
        cur = db.cursor()
        sql = 'select TickerID, DT from SnapMin ' + getWhere(tickerID, 'DT', startDate, endDate) + ' order by DT'
        cur.execute(sql)
        return cur.fetchall()


def getSnapHeaders(tickerID: any, startDate: datetime = None, endDate: datetime = None) -> np.array:
    """Returns the array of frame headers for interval"""

    MinutesRange = getMinutesRange(tickerID, startDate, endDate)
    with Parallel(n_jobs=jobs()) as P:
        FUNC = delayed(getHeader)
        F = P(FUNC(tickerID, minute) for tickerID, minute in MinutesRange)

    H = np.array([(f[0], f[1], f[2]) for f in F if f is not None], dtype=[('minute', int), ('cnt', int), ('tickerID', int)])
    H['cnt'] = H['cnt'].cumsum()
    return H


def flatSnapS(tickerID: any, startDate: date = None, endDate: date = None, height: int = 10, progress: bool = True) -> tuple:
    """
    Returns prices and volumes of orderbook snapshots as separate series    

    :param tickerID: ID 
    :param startDate: begin of interval
    :param endDate: end of interval
    :param height: orderbook height
    :param progress: set to True to show progressbar
    """

    H = getSnapHeaders(tickerID, startDate, endDate)
    K = H['cnt'][-1]  # Total orderbook snapshots count

    ts = np.zeros(K, dtype='M8[us]')
    A = np.zeros(shape=(height, K), dtype=np.float32)
    B = np.zeros(shape=(height, K), dtype=np.float32)
    vA = np.zeros(shape=(height, K), dtype=np.int32)
    vB = np.zeros(shape=(height, K), dtype=np.int32)

    def internalSnapS(h, height, ts, A, vA, B, vB):
        """callback function fills slices of result series"""
        minute, pos, ticker_id = h
        ts1, A1, vA1, B1, vB1 = getFrame(datetime.fromtimestamp(minute), ticker_id, height)
        s = slice(pos-len(ts1), pos)
        ts[s], A[:, s], vA[:, s], B[:, s], vB[:, s] = ts1, A1, -vA1, B1, vB1

    P = tqdmParallel(total=len(H), require='sharedmem', n_jobs=jobs()) if progress else Parallel(n_jobs=jobs(), require='sharedmem')
    PROC = delayed(internalSnapS)
    P(PROC(h, height, ts, A, vA, B, vB) for h in H)

    return ts, A, vA, B, vB

