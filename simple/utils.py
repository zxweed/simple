import numpy as np
from numpy.typing import NDArray
from itertools import zip_longest, starmap
import inspect

def getName(var) -> str:
    """Returns global name for specified variable"""
    frame = inspect.currentframe()

    while frame:
        for name, value in frame.f_globals.items():
            if value is var and not name.startswith('_'):
                return name
        frame = frame.f_back


def addSuffixes(X: NDArray, suffix: str = None) -> list:
    """Returns names list with suffix 'on {variable_name}' for combined datasets"""
    return [f'{name} on {getName(X) if suffix is None else suffix}' for name in X.dtype.names]


def dtypeSuffixes(X: NDArray, suffix: str = None) -> list:
    """Returns dtype with suffixed names for combined datasets"""
    types = [X.dtype[name] for name in X.dtype.names]
    return zip(addSuffixes(X, suffix), types)


def flatList(list_of_lists: [list]) -> list:
    """Creates flat list with all items from list of lists"""
    return [item for sublist in list_of_lists for item in sublist]


def npCombine(list_of_arrays: list, suffixes: list = []):
    """Combines several structure arrays into one"""

    data = list(zip_longest(list_of_arrays, suffixes))
    dtype = np.dtype(flatList(starmap(dtypeSuffixes, data)))
    R = np.zeros(len(list_of_arrays[0]), dtype=dtype)
    
    for x, suffix in data:
        name = addSuffixes(x, suffix)
        R[name] = x
        
    return R.view(np.recarray)


def vx(X: NDArray) -> NDArray:
    x = X.astype([desc for desc in X.dtype.descr if desc[0] != ''])
    dtype = x.dtype.descr[0][1]
    return x.view(dtype).reshape(len(x), len(x.dtype.names))
