from numpy import dtype, byte, double, int64

# Tick trade record structure (short and long versions)
TShortTrade = dtype([
    ('DateTime', 'datetime64[us]'),
    ('Price', double),
    ('Size', double)])

TTrade = dtype([
    ('DateTime', 'datetime64[us]'),
    ('LocalDT', 'datetime64[us]'),
    ('Price', float),
    ('Size', float),
    ('OpenInt', float)])

TBidAskDT = dtype([
    ('DateTime', 'datetime64[us]'),
    ('Bid', double),
    ('Ask', double),
    ('Act', byte)
])

# Candle record structure with additional aggressive sum/count fields
TOHLC = dtype([
    ('DateTime', 'datetime64[us]'),
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
    ('DateTime', 'datetime64[us]'),
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
    ('X0', int64), ('T0', 'datetime64[us]'), ('Price0', float), ('MidPrice0', float),
    ('X1', int64), ('T1', 'datetime64[us]'), ('Price1', float), ('MidPrice1', float),
    ('Size', float)
])

# Profit record structure
TProfit = dtype([
    ('Index', int64),
    ('DateTime', 'datetime64[us]'),
    ('RawPnL', float),
    ('MidPnL', float),
    ('Fee', float),
    ('Profit', float)
])
