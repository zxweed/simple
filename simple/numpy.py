from simple.pretty import tqdmParallel
from typing import Callable, Union, Generator, Tuple
from joblib import delayed, Parallel
import numpy as np
from functools import partial


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