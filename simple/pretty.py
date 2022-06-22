# -*- coding: utf-8 -*-
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import inspect
import psutil
from datetime import datetime, timedelta
from IPython.display import clear_output, display, Javascript
import matplotlib as mpl


class tqdmParallel(Parallel):
    """Show progress bar when parallel processing"""

    def __init__(self, n_jobs=-1, progress=True, total=None, postfix=None, *args, **kwargs):
        self._progress = progress
        self._total = total
        self._lasttime = datetime.now()
        self._postfix = postfix if type(postfix) == dict else {'name': postfix} if postfix is not None else None
        super().__init__(n_jobs=n_jobs, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._progress, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks

        if datetime.now() > self._lasttime + timedelta(milliseconds=500):
            postfix = {'cpu': f'{psutil.cpu_percent():1.0f}%'}
            if self._postfix is not None:
                postfix |= self._postfix

            self._pbar.set_postfix(postfix, refresh=True)
            self._lasttime = datetime.now()


def pmap(func: callable, params, **kwargs):
    """Parallel map implementation via tqdmParallel"""

    with tqdmParallel(total=len(params), **kwargs) as P:
        FUNC = delayed(func)
        return P(FUNC(param) for param in params)


tpl = lambda x: x if isinstance(x, tuple) else tuple(x.values()) if isinstance(x, dict) else (x,)
"""Convert value/dict to tuple"""


def pxmap(func: callable, param, **kwargs):
    """Another parallel map implementation, returns DataFrame, supports many parameters and results"""

    param_names = inspect.getfullargspec(func).args
    param_list = list(param)

    with tqdmParallel(total=len(param_list),  **kwargs) as P:
        FUNC = delayed(func)
        X = list(P(FUNC(*tpl(param)) for param in param_list))

    # append result names
    if len(X) > 0:
        if type(X[0]) == dict:
            param_names.extend(X[0].keys())
        elif type(X[0]) == tuple:
            param_names.extend([f'result_{n}' for n in range(len(X[0]))])
        else:
            param_names.append('result')

    return pd.DataFrame([(*tpl(x[0]), *tpl(x[1])) for x in zip(param_list, X)], columns=param_names)


def Corr(X, Y):
    """Correlation coefficient"""
    return np.corrcoef(np.nan_to_num(X), np.nan_to_num(Y))[0, 1] * 100


def background(self, scale='Linear', cmap='RdYlGn', **css):
    """For use with `DataFrame.style.apply` this function will apply a heatmap color gradient *elementwise* to the calling DataFrame

    self : pd.DataFrame
        The calling DataFrame. This argument is automatically passed in by the `DataFrame.style.apply` method

    cmap : colormap or str
        The colormap to use

    css : dict
        Any extra inline css key/value pars to pass to the styler

    Returns
    -------
    pd.DataFrame
        The css styles to apply

    """
    cvals = self.fillna(0).values.ravel().copy()

    q = np.percentile(cvals, 99.9)
    vmax = max(cvals.max(), abs(np.clip(cvals, -q, q).min()))
    if scale == 'Linear':
        norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax, clip=True)  # Linear colorscale
    else:
        norm = mpl.colors.SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax, clip=True,
                                     base=2.17)  # Logarithmic color scale
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    styles = []
    for c in cvals:
        rgb = mapper.to_rgba(c)
        style = ["{}: {}".format(key.replace('_', '-'), value) for key, value in css.items()]
        style.append("background-color: {}".format(mpl.colors.rgb2hex(rgb)))
        styles.append('; '.join(style))
    styles = np.asarray(styles).reshape(self.shape)
    return pd.DataFrame(styles, index=self.index, columns=self.columns)


def pp(X: pd.DataFrame):
    """Pretty print pandas dataframe"""
    return X.style.apply(background, axis=None)

