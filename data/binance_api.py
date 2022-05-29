import hmac
import time
import hashlib
import requests
import json
from urllib.parse import urlencode
from pprint import pprint


KEY = ""
SECRET = ""
BASE_URL = 'https://fapi.binance.com'               # production base url
# BASE_URL = "https://testnet.binancefuture.com"    # testnet base url


def hashing(query_string):
    return hmac.new(
        SECRET.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def get_timestamp():
    return int(time.time() * 1000)


def dispatch_request(http_method):
    session = requests.Session()
    session.headers.update(
        {"Content-Type": "application/json;charset=utf-8", "X-MBX-APIKEY": KEY}
    )
    return {
        "GET": session.get,
        "DELETE": session.delete,
        "PUT": session.put,
        "POST": session.post,
    }.get(http_method, "GET")


# used for sending request requires the signature
def send_signed_request(http_method, url_path, payload={}):
    query_string = urlencode(payload)
    # replace single quote to double quote
    query_string = query_string.replace("%27", "%22")
    if query_string:
        query_string = "{}&timestamp={}".format(query_string, get_timestamp())
    else:
        query_string = "timestamp={}".format(get_timestamp())

    url = (
        BASE_URL + url_path + "?" + query_string + "&signature=" + hashing(query_string)
    )
    print("{} {}".format(http_method, url))
    params = {"url": url, "params": {}}
    response = dispatch_request(http_method)(**params)
    return response.json()


# used for sending public data request
def send_public_request(url_path, payload={}):
    query_string = urlencode(payload, True)
    url = BASE_URL + url_path
    if query_string:
        url = url + "?" + query_string
    print("{}".format(url))
    response = dispatch_request("GET")(url=url)
    return response.json()


###################################################################################
if __name__ == '__main__':
    # get klines
    response = send_public_request(
        "/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1d"}
    )
    print(response)

    # get account informtion
    response = send_signed_request("GET", "/fapi/v2/account")
    print(response)

    # order book real-time
    response = send_signed_request("GET", "/fapi/v1/depth", {"symbol": "BTCUSDT"})
    print(response)

    # 24h statistics by instrument
    response = send_signed_request("GET", "/fapi/v1/ticker/24hr", {"symbol": "BTCUSDT"})
    print(response)

    # get 24h top quoted volume instruments
    response = send_signed_request("GET", "/fapi/v1/ticker/24hr")
    r = sorted(response, key=lambda x: float(x['volume']) * float(x['lastPrice']), reverse=True)
    pprint([{i['symbol']: float(i['volume']) * float(i['lastPrice'])} for i in r][:10])

    #  recent trades
    response = send_signed_request('GET', '/fapi/v1/trades', {"symbol": "BTCUSDT"})
    pprint(response[0])

    # recent aggTrades
    response = send_signed_request('GET', '/fapi/v1/aggTrades', {"symbol": "BTCUSDT"})
    pprint(response[0])

    # defi composite index
    response = send_signed_request("GET", "/fapi/v1/indexInfo")
    defi_components = [i for i in response if i['symbol'] == 'DEFIUSDT'][0]['baseAssetList']
    defi_top5 = sorted(defi_components, key=lambda x: x['weightInPercentage'], reverse=True)[:5]
    pprint(defi_top5)
