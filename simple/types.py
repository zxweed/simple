import numpy as np

# Tick trade record structure
TTrade = np.dtype([
    ('DateTimeA', 'M8[us]'),
    ('LocalTimeA', 'M8[us]'),
    ('PriceA', float),
    ('VolumeA', float),
    ('OpenIntA', float)])

# Candle record structure
TOHLC = np.dtype([
    ('DT', '<M8[us]'),
    ('Open', float),
    ('High', float),
    ('Low', float),
    ('Close', float),
    ('Volume', float),
    ('Buy', int),
    ('Sell', int)])

# Debounced timeseries record structure
TDebounce = np.dtype([
    ('DT', '<M8[us]'),
    ('Index', int),
    ('Price', float),
    ('Duration', '<m8[us]'),

    ('Volume', float),
    ('BuySize', float),
    ('SellSize', float),

    ('Count', int),
    ('BuyCount', int),
    ('SellCount', int)
])