import numpy as np

# Tick trade record structure
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