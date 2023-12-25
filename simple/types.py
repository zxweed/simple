from numpy import dtype, byte, double, int64

# Tick trade record structure (short and long versions)
TShortTrade = dtype([
    ('DateTime', 'M8[us]'),
    ('Price', double),
    ('Size', double)])

TTrade = dtype([
    ('DateTime', 'M8[us]'),
    ('LocalDT', 'M8[us]'),
    ('Price', float),
    ('Size', float),
    ('OpenInt', float)])

TBidAskDT = dtype([
    ('DateTime', 'M8[us]'),
    ('Bid', double),
    ('Ask', double),
    ('Act', byte)
])

TOHLC = dtype([
    ('DateTime', 'M8[us]'),
    ('Open', float),
    ('High', float),
    ('Low', float),
    ('Close', float)
])

# Candle record structure with additional aggressive sum/count fields
TExtOHLC = dtype([
    ('DateTime', 'M8[us]'),
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
TDebounce = dtype([
    ('DateTime', 'M8[us]'),
    ('Index', int),
    ('Price', float),
    ('Duration', 'timedelta64[us]'),

    ('Size', float),
    ('BuySize', float),
    ('SellSize', float),

    ('Count', int),
    ('BuyCount', int),
    ('SellCount', int)
])

TDebounceSpread = dtype([
    ('Ask', float),
    ('Bid', float),
    ('Mean', float),
    ('AskLiq', float),
    ('BidLiq', float)
])

# Paired trade record structure
TPairTrade = dtype([
    ('X0', int64), ('T0', 'M8[us]'), ('Price0', float), ('MidPrice0', float),
    ('X1', int64), ('T1', 'M8[us]'), ('Price1', float), ('MidPrice1', float),
    ('Size', float)
])

# Profit record structure
TProfit = dtype([
    ('Index', int64),
    ('DateTime', 'M8[us]'),
    ('RawPnL', float),
    ('MidPnL', float),
    ('Fee', float),
    ('Profit', float)
])
