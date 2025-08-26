import numpy as np
from numpy.typing import NDArray
from typing import Union, List
from itertools import zip_longest, starmap, product
from inspect import currentframe, getfullargspec
from joblib import Parallel, delayed
from psutil import cpu_percent
from os import cpu_count
from os.path import getsize, splitext, exists
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


def npMMF(filename: str, dtype: np.dtype = None) -> NDArray:
    """Returns memory-mapped array of specified filename and dtype"""
    if dtype is None:
        name, ext = splitext(filename)
        with open(f'{name}.dtype') as f:
            s = f.readline()
        dtype = np.dtype([tuple(row) for row in loads(s)])

    rec_count = getsize(filename) // np.dtype(dtype).itemsize
    X = np.memmap(filename, mode='r+', shape=rec_count, dtype=dtype)
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
    if type(obj) == str:
        return False

    try:
        iter(obj)
        return True
    except:
        return False


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


def npDateTime(T: np.dtype, new_dtype: object = 'M8[us]') -> np.dtype:
    """Replace DateTime/DT field's dtype with another specified dtype"""
    return np.dtype([(d[0], new_dtype if d[0].lower() in ['datetime', 'dt', 'timestamp', 'ts'] else d[1]) for d in T.descr])

def asDateTime(X: NDArray, new_dtype: object = 'M8[us]') -> NDArray:
    return X.view(npDateTime(X.dtype, new_dtype)).view(np.recarray)
