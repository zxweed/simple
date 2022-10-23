import numpy as np

# Tick trade record structure
TShortTrade = np.dtype([
    ('DT', 'M8[us]'),
    ('Price', 'f8'),
    ('Size', 'f8')])

TTrade = np.dtype([
    ('DT', 'M8[us]'),
    ('LocalDT', 'M8[us]'),
    ('Price', float),
    ('Size', float),
    ('OpenInt', float)])

# Candle record structure
TOHLC = np.dtype([
    ('DT', '<M8[us]'),
    ('Open', float),
    ('High', float),
    ('Low', float),
    ('Close', float),

    ('Size', float),
    ('BuySize', float),
    ('SellSize', float),

    ('Count', int),
    ('BuyCount', int),
    ('SellCount', int)])

# Debounced timeseries record structure
TDebounce = np.dtype([
    ('DT', '<M8[us]'),
    ('Index', int),
    ('Price', float),
    ('Duration', '<m8[us]'),

    ('Size', float),
    ('BuySize', float),
    ('SellSize', float),

    ('Count', int),
    ('BuyCount', int),
    ('SellCount', int)
])

TDebounceSpread = np.dtype([
    ('Ask', float),
    ('Bid', float),
    ('Mean', float),
    ('AskLiq', float),
    ('BidLiq', float)
])

# Paired trade record structure
TPairTrade = [
    ('X0', np.int64), ('T0', 'M8[us]'), ('Price0', float), ('MidPrice0', float),
    ('X1', np.int64), ('T1', 'M8[us]'), ('Price1', float), ('MidPrice1', float),
    ('Size', float)]

# Profit record structure
TProfit = [
    ('Index', np.int64),
    ('DT', 'M8[us]'),
    ('RawPnL', float),
    ('MidPnL', float),
    ('Fee', float),
    ('Profit', float)
]
