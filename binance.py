import pandas as pd
import asyncio
import aiohttp

api = 'https://api.binance.com/api/v3/klines'


async def asyncOHLC(tm, ticker, interval):
    async with aiohttp.ClientSession() as session:
        url = f'{api}?&symbol={ticker}&interval={interval}&startTime={int(tm.timestamp() * 1000)}'

        async with session.get(url) as response:
            json = await response.json()
            df = pd.DataFrame(json)
            if len(df) > 0:
                df = df.set_index(df[0].astype('M8[ms]'))[[1, 2, 3, 4, 5]].apply(pd.to_numeric)
                df.index.name = 'DateTime'
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return df


async def getOHLC(ticker, start_time, end_time, minutes=5):
    TM = pd.date_range(start_time, end_time, freq=f'{500*minutes}min')
    tasks = [asyncio.create_task(asyncOHLC(tm, ticker, f'{minutes}m')) for tm in TM]
    results = await asyncio.gather(*tasks)
    return pd.concat(results).sort_index()
