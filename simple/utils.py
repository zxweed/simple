import numpy as np
from numpy.typing import NDArray
from itertools import zip_longest, starmap, product
from inspect import currentframe, getfullargspec
from joblib import delayed
from typing import Union
from .pretty import tqdmParallel


def getName(var) -> str:
    """Returns global name for specified variable"""
    frame = currentframe()

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


def common_type(types: Union[list, set]) -> type:
    """
    Find the common type among the given types.
    Args:
        types (list or set): A list of types.
    Returns:
        type: The common type among the given types.
    """
    if all(issubclass(t, int) for t in types):
        return int
    elif any(issubclass(t, str) for t in types):
        return np.dtype('U32')
    elif any(issubclass(t, float) for t in types):
        return float
    else:
        return object


def inclusive_range(*args, count: int = 10):
    """
    Generates an inclusive range of numbers based on the given arguments.

    Args:
        *args: The arguments can be passed in any of the following formats:
            - (stop): Generates numbers from 0 to stop-1 with a step of 1.
            - (start, stop): Generates numbers from start to stop-1 with a step of 1.
            - (start, stop, step): Generates numbers from start to stop-1 with a step of step.
        count: The number of elements in the generated range. Defaults to 10.

    Yields:
        int: The next number in the inclusive range.

    Raises:
        TypeError: If no arguments are passed or if more than 3 arguments are passed.
    """
    nargs = len(args)
    if nargs == 0:
        raise TypeError("you need to write at least a value")
    elif nargs == 1:
        stop = args[0]
        start = 0
        step = 1
    elif nargs == 2:
        (start, stop) = args
        step = (stop - start) / (count - 1)
    elif nargs == 3:
        (start, stop, step) = args
    else:
        raise TypeError(f"Inclusive range was expected at most 3 arguments, but got {nargs}")
    i = start
    while i < stop + step/2:
        yield i
        i += step


def gridrun(func: callable, count: int = 10, **kwargs) -> NDArray:
    """Parallel grid search for all parameters combinations"""

    spec = getfullargspec(func)
    def_list = [] if spec.defaults is None else reversed(spec.defaults)
    defaults = dict(reversed(list(zip_longest(reversed(spec.args), def_list, fillvalue=100))))

    # create list with all parameter combinations
    X = product(*(inclusive_range(*v, count=count) if type(v) is tuple else v for v in defaults.values()))
    params = [dict(zip(spec.args, x)) for x in X]

    # run parallel grid search
    with tqdmParallel(total=len(params), **kwargs) as P:
        FUNC = delayed(func)
        log = P(FUNC(**arg) for arg in params)

    # stack parameters and results
    result_list = [
        (*param.values(), *(result.values() if type(result) is dict else (result,)))
        for param, result in zip(params, log)
    ]

    # create structured array for result
    columns = spec.args + (list(log[0].keys()) if type(log[0]) is dict else ['value'])
    dtype = [(col, common_type(set([type(r[i]) for r in result_list]))) for i, col in enumerate(columns)]
    return np.array(result_list, dtype=dtype).view(np.recarray)
