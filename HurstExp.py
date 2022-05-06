import numpy as np
from numba import njit
from scipy import signal


def hurst_exponent(source: np.ndarray, min_chunksize: int = 8, max_chunksize: int = 200, num_chunksize: int = 5, method: int = 1) -> float:
    """
    Hurst Exponent

    :param source: np.ndarray
    :param min_chunksize: int - default: 8
    :param max_chunksize: int - default: 200
    :param num_chunksize: int - default: 5
    :param method: int - default: 1 - 0: RS | 1: DMA | 2: DSOD
    :param source_type: str - default: "close"

    :return: float
    """

    if method == 0:
        h = HurstRS(np.diff(source), min_chunksize, max_chunksize, num_chunksize)
    elif method == 1:
        h = HurstDMA(source, min_chunksize, max_chunksize, num_chunksize)
    elif method == 2:
        h = HurstDSOD(source)
    else:
        raise NotImplementedError('The method choose is not implemented.')

    return None if np.isnan(h) else h


@njit(nogil=True)
def HurstRS(X: np.array, period: int, min_chunksize: int = 8, max_chunksize: int = 200, num_chunksize: int = 5) -> np.array:
    """
    Estimates the Hurst (H) exponent using the R/S method from the time series.
    The R/S method consists of dividing the series into pieces of equal size
    `series_len` and calculating the rescaled range. This repeats the process
    for several `series_len` values and adjusts data regression to obtain the H.
    `series_len` will take values between `min_chunksize` and `max_chunksize`,
    the step size from `min_chunksize` to `max_chunksize` can be controlled
    through the parameter `step_chunksize`.
    Parameters
    ----------
    X : 1D-array
        A time series to calculate hurst exponent, must have more elements
        than `min_chunksize` and `max_chunksize`.
    period: int
        Length of sliding window
    min_chunksize : int
        This parameter allows you to control the minimum window size.
    max_chunksize : int
        This parameter allows you to control the maximum window size.
    num_chunksize : int
        This parameter allows you to control the size of the step from minimum to
        maximum window size. Bigger step means fewer calculations.
    out : 1-element-array, optional
        one element array to store the output.
    Returns
    -------
    H : float
        A estimation of Hurst exponent.
    References
    ----------
    Hurst, H. E. (1951). Long term storage capacity of reservoirs. ASCE Transactions, 116(776), 770-808.
    Alessio, E., Carbone, A., Castelli, G. et al. Eur. Phys. J. B (2002) 27:
    197. http://dx.doi.org/10.1140/epjb/e20020150
    """

    max_chunksize += 1
    result = np.zeros(len(X), dtype=np.float64)
    rs_tmp = np.empty(period, dtype=np.float64)
    chunk_size_list = np.linspace(min_chunksize, max_chunksize, num_chunksize).astype(np.int64)
    rs_values_list = np.empty(num_chunksize, dtype=np.float64)

    for p in range(len(X) - period):
        k = p + period
        x = X[p:k]

        # 1. The series is divided into chunks of chunk_size_list size
        for i in range(num_chunksize):
            chunk_size = chunk_size_list[i]

            # 2. it iterates on the indices of the first observation of each chunk
            number_of_chunks = int(len(x) / chunk_size)

            for idx in range(number_of_chunks):
                # next means no overlapping
                # convert index to index selection of each chunk
                ini = idx * chunk_size
                end = ini + chunk_size
                chunk = x[ini:end]

                # 2.1 Calculate the RS (chunk_size)
                z = np.cumsum(chunk - np.mean(chunk))
                rs_tmp[idx] = np.divide(
                    np.max(z) - np.min(z),  # range
                    np.nanstd(chunk)  # standard deviation
                )

            # 3. Average of RS(chunk_size)
            rs_values_list[i] = np.nanmean(rs_tmp[:idx + 1])

        # 4. calculate the Hurst exponent.
        h, c = np.linalg.lstsq(
            a=np.vstack((np.log(chunk_size_list), np.ones(num_chunksize))).T,
            b=np.log(rs_values_list)
        )[0]

        result[k] = h

    result[:period] = result[period]
    return result


def HurstDMA(X: np.array, period: int, min_chunksize=8, max_chunksize=200, num_chunksize=5) -> np.array:
    """Estimate the Hurst exponent using R/S method.

    Estimates the Hurst (H) exponent using the DMA method from the time series.
    The DMA method consists on calculate the moving average of size `series_len`
    and subtract it to the original series and calculating the standard
    deviation of that result. This repeats the process for several `series_len`
    values and adjusts data regression to obtain the H. `series_len` will take
    values between `min_chunksize` and `max_chunksize`, the step size from
    `min_chunksize` to `max_chunksize` can be controlled through the parameter
    `step_chunksize`.

    Parameters
    ----------
    X
    min_chunksize
    max_chunksize
    num_chunksize

    Returns
    -------
    hurst_exponent : float
        Estimation of hurst exponent.

    References
    ----------
    Alessio, E., Carbone, A., Castelli, G. et al. Eur. Phys. J. B (2002) 27:
    197. https://dx.doi.org/10.1140/epjb/e20020150

    """

    result = np.zeros(len(X), dtype=np.float64)
    max_chunksize += 1
    n_list = np.arange(min_chunksize, max_chunksize, num_chunksize, dtype=np.int64)
    dma_list = np.empty(len(n_list))
    factor = 1 / (period - max_chunksize)

    for p in range(len(X) - period):
        k = p + period
        x = X[p:k]
        # sweeping n_list
        for i, n in enumerate(n_list):
            b = np.divide([n - 1] + (n - 1) * [-1], n)  # do the same as:  y - y_ma_n
            noise = np.power(signal.lfilter(b, 1, x)[max_chunksize:], 2)
            dma_list[i] = np.sqrt(factor * np.sum(noise))

        h, const = np.linalg.lstsq(
            a=np.vstack([np.log10(n_list), np.ones(len(n_list))]).T,
            b=np.log10(dma_list), rcond=None
        )[0]
        result[k] = h

    result[:period] = result[period]
    return result


def HurstDSOD(X, period):
    """Estimate Hurst exponent on data timeseries.

    The estimation is based on the discrete second order derivative. Consists on
    get two different noise of the original series and calculate the standard
    deviation and calculate the slope of two point with that values.
    source: https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc

    Parameters
    ----------
    x : numpy array
        time series to estimate the Hurst exponent for.

    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given time series.

    References
    ----------
    Istas, J.; G. Lang (1994), “Quadratic variations and estimation of the local
    Hölder index of data Gaussian process,” Ann. Inst. Poincaré, 33, pp. 407–436.


    Notes
    -----
    This hurst_ets is data literal transduction of wfbmesti.m of wavelet toolbox
    from matlab.
    """
    result = np.zeros(len(X), dtype=np.float64)
    for p in range(len(X) - period):
        k = p + period
        x = X[p:k]
        y = np.cumsum(np.diff(x, axis=0), axis=0)

        # second order derivative
        b1 = [1, -2, 1]
        y1 = signal.lfilter(b1, 1, y, axis=0)
        y1 = y1[len(b1) - 1:]  # first values contain filter artifacts

        # wider second order derivative
        b2 = [1, 0, -2, 0, 1]
        y2 = signal.lfilter(b2, 1, y, axis=0)
        y2 = y2[len(b2) - 1:]  # first values contain filter artifacts

        s1 = np.mean(y1 ** 2, axis=0)
        s2 = np.mean(y2 ** 2, axis=0)

        result[k] = 0.5 * np.log2(s2 / s1)

    result[:period] = result[period]
    return result
