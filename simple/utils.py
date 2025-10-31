import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union, Generator, Tuple, List
from itertools import zip_longest, starmap, product
from functools import partial
from inspect import currentframe, getfullargspec, getargvalues
from joblib import Parallel, delayed
from psutil import cpu_percent
from os import cpu_count
from os.path import getsize, splitext
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from multiprocessing.shared_memory import SharedMemory
from json import dumps, loads


class tqdmParallel(Parallel):
    """
    Enhanced Parallel class that shows a progress bar during parallel processing.
    This class extends joblib.Parallel to provide visual feedback on task completion.
    """

    def __init__(self, progress=True, total=None, postfix=None, n_jobs=cpu_count()//2, **kwargs):
        self.progress = progress
        self.total = total
        self.lasttime = datetime.now()
        self.postfix = postfix if type(postfix) is dict else {'name': postfix} if postfix is not None else None
        super().__init__(n_jobs=n_jobs, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self.progress, total=self.total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks

        if datetime.now() > self.lasttime + timedelta(milliseconds=500):
            postfix = {'cpu': f'{cpu_percent():1.0f}%'}

            # add user specified postfix values to the progress bar
            if self.postfix is not None:
                # evaluate postfix values if callables was specified
                if isinstance(self.postfix, dict):
                    user_postfix = {key: value() if callable(value) else value for key, value in self.postfix.items()}
                else:
                    user_postfix = {'value': self.postfix() if callable(self.postfix) else self.postfix}
                postfix = {**postfix, **user_postfix}  # merge dict operator with python 3.8 compatibility

            self._pbar.set_postfix(postfix, refresh=True)
            self.lasttime = datetime.now()


def tpl(x):
    """Convert value/dict to tuple. Useful for functions that expect tuple inputs"""
    return x if isinstance(x, tuple) else tuple(x.values()) if isinstance(x, dict) else (x,)


def plist(*args):
    """
    Creates product parameter list from a bunch of iterables/values
    Useful for generating all possible combinations of input parameters

    Examples:
        >>> plist([1, 2], ['a', 'b'])
        [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

        >>> plist([1, 2], 'x', [True, False])
        [(1, 'x', True), (1, 'x', False), (2, 'x', True), (2, 'x', False)]

        >>> plist(range(2), ['red', 'blue'])
        [(0, 'red'), (0, 'blue'), (1, 'red'), (1, 'blue')]
    """
    return list(product(*(p if iterable(p) else [p] for p in args)))


def pmap(func: callable, *args, params: List[tuple] = None, combine: bool = False, **kwargs):
    """
    Parallel map/starmap implementation via tqdmParallel

    Parameters
    ----------
    func : callable
        Function to apply to each parameter combination
    *args
        Positional arguments to be passed to the func
    params : List[tuple], optional
        List of parameter combinations, by default None
    combine : bool, optional
        If True, combine parameter list with results, by default False
    **kwargs
        Additional arguments to be passed to the tqdmParallel:
        desc : str, optional
            Progress bar description, by default empty
        n_jobs : int
            Parallel jobs count, by default half of the available cores
        progress : bool
            True to show progress bar, enabled by default
    """
    param_list = params if params is not None else plist(*args)
    with tqdmParallel(total=len(param_list), **kwargs) as P:
        FUNC = delayed(func)
        result = P(FUNC(*tpl(param)) for param in param_list)

    # combine parameter list with results if specified
    return [(*p, *tpl(v)) for p, v in zip(param_list, result)] if combine else result


def prun(indicator: callable, src, period: int, threads: int = cpu_count()//2, progress: bool = True) -> np.array:
    """Parallel calculate any indicator by slicing source series"""

    page_size = (len(src) - period + (threads - 1)) // threads
    start_indexes = [t * page_size for t in range(threads)]
    result_indexes = list(range(period, len(src), page_size))

    with tqdmParallel(total=len(start_indexes), n_jobs=threads, require='sharedmem', progress=progress) as P:
        FUNC = delayed(indicator)
        slice_size = period + page_size
        X = P(FUNC(src[start:start + slice_size], period) for start in start_indexes)

    # store all slices to the result series
    result = np.zeros(len(src))
    for page_start, x in zip(result_indexes, X):
        m = page_start + page_size
        page_stop = m if m < len(result) else len(result)
        result[page_start:page_stop] = x[period:]

    # fill prefix with first value
    result[:period] = result[period]
    return result


def npMMF(filename: str, dtype: np.dtype = None, mode: str = 'r') -> NDArray:
    """Returns specified filename as memory-mapped array of specified or stored dtype"""
    if dtype is None:
        name, ext = splitext(filename)
        with open(f'{name}.dtype') as f:
            s = f.readline()
        dtype = np.dtype([tuple(row) for row in loads(s)])

    rec_count = getsize(filename) // np.dtype(dtype).itemsize
    X = np.memmap(filename, mode=mode, shape=rec_count, dtype=dtype)
    return X.view(np.recarray)


def toMMF(filename: str, A: NDArray):
    """Stores array to memory-mapped file"""
    mf = np.memmap(filename, mode='w+', shape=A.shape, dtype=A.dtype)
    mf[:] = A
    mf.flush()

    s_dtype = dumps(A.dtype.descr)
    name, ext = splitext(filename)
    with open(f'{name}.dtype', 'w') as f:
        f.write(s_dtype)


def asShared(X: np.ndarray, shm_name:str=None) -> np.ndarray:
    """Create shared memory copy of the array to improve further parallel performance"""
    try:
        # Try to attach to existing shared memory
        shm = SharedMemory(shm_name)
    except FileNotFoundError:
        # If it doesn't exist, create a new shared memory region
        shm = SharedMemory(shm_name, create=True, size=len(X) * X.dtype.itemsize)
    
    result = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
    result[:] = X
    return result, shm


def common_type(types: Union[list, set]) -> type:
    """
    Find the common type among the given types.
    Args:
        types (list): A list of types.
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


def iterable(obj):
    """Check if an object is iterable (except string because iterating over characters does not make sense here)"""
    if isinstance(obj, str):
        return False

    try:
        iter(obj)
        return True
    except TypeError:
        return False


def getName(var) -> str:
    """Returns global name for specified variable"""
    frame = currentframe()

    while frame:
        for name, value in frame.f_globals.items():
            if value is var and not name.startswith('_'):
                return name
        frame = frame.f_back
    return None


def getFuncParams() -> dict:
    """Returns parameters dictionary of the calling function"""
    args_info = getargvalues(currentframe().f_back)
    params = {arg: args_info.locals[arg] for arg in args_info.args}
    return params


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


def npCombine(list_of_arrays: list, suffixes: list = None):
    """Combines several structure arrays into one"""
    if suffixes is None:
        suffixes = []

    data = list(zip_longest(list_of_arrays, suffixes))
    dtype = np.dtype(flatList(starmap(dtypeSuffixes, data)))
    R = np.zeros(len(list_of_arrays[0]), dtype=dtype)

    for x, suffix in data:
        name = addSuffixes(x, suffix)
        R[name] = x

    return R.view(np.recarray)


def vx(X: NDArray) -> NDArray:
    """Converts a structured array into a 2D array of values"""
    x = X.astype([desc for desc in X.dtype.descr if desc[0] != ''])
    dtype = x.dtype.descr[0][1]
    return x.view(dtype).reshape(len(x), len(x.dtype.names))


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
    while i < stop + step / 2:
        yield i
        i += step


def gridrun(func: callable, count: int = 10, **kwargs) -> NDArray:
    """Parallel grid search for all parameters combinations"""

    spec = getfullargspec(func)
    def_list = [] if spec.defaults is None else reversed(spec.defaults)
    defaults = dict(reversed(list(zip_longest(reversed(spec.args), def_list, fillvalue=100))))

    # create list with all parameter combinations
    X = product(*(inclusive_range(*v, count=count) if isinstance(v, tuple) else v for v in defaults.values()))
    params = [dict(zip(spec.args, x)) for x in X]

    # run parallel grid search
    with tqdmParallel(total=len(params), **kwargs) as P:
        FUNC = delayed(func)
        log = P(FUNC(**arg) for arg in params)

    # stack parameters and results
    result_list = [
        (*param.values(), *(result.values() if isinstance(result, dict) else (result,)))
        for param, result in zip(params, log)
    ]

    # create structured array for result
    columns = spec.args + (list(log[0].keys()) if isinstance(log[0], dict) else ['value'])
    dtype = [(col, common_type({type(r[i]) for r in result_list})) for i, col in enumerate(columns)]
    return np.array(result_list, dtype=dtype).view(np.recarray)


def npDateTime(T: np.dtype, new_dtype: object = 'M8[us]') -> np.dtype:
    """Replace DateTime/DT field's dtype with another specified dtype"""
    return np.dtype([(d[0], new_dtype if d[0].lower() in ['datetime', 'dt', 'timestamp', 'ts'] else d[1]) for d in T.descr])


def asDateTime(X: NDArray, new_dtype: object = 'M8[us]') -> NDArray:
    """Convert a DateTime/DT fields in structured array to datetime64 dtype"""
    return X.view(npDateTime(X.dtype, new_dtype)).view(np.recarray)


def rolling(array: np.ndarray, window: int, skip_na: bool = False, as_array: bool = False) -> Union[Generator[np.ndarray, None, None], np.ndarray]:
    """
    Roll a fixed-width window over an array.
    The result is either a 2-D array or a generator of slices, controlled by `as_array` parameter.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    window : int
        Size of the rolling window.
    skip_na : bool, optional
        If False, the sequence starts with (window-1) windows filled with nans. If True, those are omitted.
        Default is False.
    as_array : bool, optional
        If True, return a 2-D array. Otherwise, return a generator of slices. Default is False.

    Returns
    -------
    np.ndarray or Generator[np.ndarray, None, None]
        Rolling window matrix or generator

    Examples
    --------
    >>> rolling(np.array([1, 2, 3, 4, 5]), 2, as_array=True)
    array([[nan,  1.],
           [ 1.,  2.],
           [ 2.,  3.],
           [ 3.,  4.],
           [ 4.,  5.]])

    Usage with numpy functions

    >>> arr = rolling(np.array([1, 2, 3, 4, 5]), 2, as_array=True)
    >>> np.sum(arr, axis=1)
    array([nan,  3.,  5.,  7.,  9.])
    """
    if not any(isinstance(window, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong window type ({type(window)}) int expected')

    window = int(window)

    if array.size < window:
        raise ValueError('array.size should be bigger than window')

    def rows_gen():
        if not skip_na:
            yield from (prepend_na(array[:i + 1], (window - 1) - i) for i in np.arange(window - 1))

        starts = np.arange(array.size - (window - 1))
        yield from (array[start:end] for start, end in zip(starts, starts + window))

    return np.array([row for row in rows_gen()]) if as_array else rows_gen()


def nans(shape: Union[int, Tuple[int]], dtype=np.float64) -> np.ndarray:
    """
    Return a new array of a given shape and type, filled with np.nan values.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., (2, 3) or 2.
    dtype: data-type, optional

    Returns
    -------
    np.ndarray
        Array of np.nans of the given shape.

    Examples
    --------
    >>> nans(3)
    array([nan, nan, nan])
    >>> nans((2, 2))
    array([[nan, nan],
           [nan, nan]])
    >>> nans(2, np.datetime64)
    array(['NaT', 'NaT'], dtype=datetime64)
    """
    if np.issubdtype(dtype, np.integer):
        dtype = np.float
    arr = np.empty(shape, dtype=dtype)
    arr.fill(np.nan)
    return arr


def prepend_na(array: np.ndarray, n: int) -> np.ndarray:
    """
    Return a copy of array with nans inserted at the beginning.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    n : int
        Number of elements to insert.

    Returns
    -------
    np.ndarray
        New array with nans added at the beginning.

    Examples
    --------
    >>> prepend_na(np.array([1, 2]), 2)
    array([nan, nan,  1.,  2.])
    """
    return np.hstack(
        (
            nans(n, array[0].dtype) if len(array) and hasattr(array[0], 'dtype') else nans(n),
            array
        )
    )


def rolling_apply(func: Callable, window: int, *arrays: np.ndarray, n_jobs: int = 1, progress: bool = True, **kwargs) -> np.ndarray:
    """
    Roll a fixed-width window over an array or a group of arrays, producing slices.
    Apply a function to each slice / group of slices, transforming them into a value.
    Perform computations in parallel, optionally.
    Return a new np.ndarray with the resulting values.

    Parameters
    ----------
    func : Callable
        The function to apply to each slice or a group of slices.
    window : int
        Window size.
    *arrays : list
        List of input arrays.
    n_jobs : int, optional
        Parallel tasks count for joblib. If 1, joblib won't be used. Default is 1.
    **kwargs : dict
        Input parameters (passed to func, must be named).

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> rolling_apply(sum, 2, arr)
    array([nan,  3.,  5.,  7.,  9.])
    >>> arr2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> func = lambda a1, a2, k: (sum(a1) + max(a2)) * k
    >>> rolling_apply(func, 2, arr, arr2, k=-1)
    array([  nan,  -5.5,  -8.5, -11.5, -14.5])
    """
    if not any(isinstance(window, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong window type ({type(window)}) int expected')

    window = int(window)

    if max(len(x.shape) for x in arrays) != 1:
        raise ValueError('Wrong array shape. Supported only 1D arrays')

    if len({array.size for array in arrays}) != 1:
        raise ValueError('Arrays must be the same length')

    def _apply_func_to_arrays(idxs):
        return func(*[array[idxs[0]:idxs[-1] + 1] for array in arrays], **kwargs)

    array = arrays[0]
    rolls = rolling(
        array if len(arrays) == n_jobs == 1 else np.arange(len(array)),
        window=window,
        skip_na=True
    )

    if n_jobs == 1:
        if len(arrays) == 1:
            arr = list(map(partial(func, **kwargs), rolls))
        else:
            arr = list(map(_apply_func_to_arrays, rolls))
    else:
        f = delayed(_apply_func_to_arrays)
        P = tqdmParallel(total=len(array)-window) if progress else Parallel(n_jobs=-1)
        arr = P(f(idxs[[0, -1]]) for idxs in rolls)

    return prepend_na(arr, n=window-1)
