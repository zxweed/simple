# -*- coding: utf-8 -*-
from tqdm.auto import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from joblib import Parallel, delayed
import inspect
from datetime import datetime, timedelta
from IPython.display import clear_output, display, Javascript


class tqdmParallel(Parallel):
    """Show progress bar when parallel processing"""

    def __init__(self, n_jobs=-1, progress=True, total=None, *args, **kwargs):
        self._progress = progress
        self._total = total
        self._lasttime = datetime.now()
        super().__init__(n_jobs=n_jobs, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._progress, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks

        if datetime.now() > self._lasttime + timedelta(milliseconds=500):
            self._pbar.refresh()
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


def FeatureImportance(model, names=None, top=20, palette='Blues_r'):
    """Feature importance chart"""
    names = model.feature_name() if names is None else names

    f = pd.Series(model.feature_importance('gain'), index=names).sort_values(ascending=False)[:top]
    sns.barplot(y=f.index, x=f, palette=palette, ax=plt.subplot(121)).set(ylabel='Feature by GAIN');

    f = pd.Series(model.feature_importance('split'), index=names).sort_values(ascending=False)[:top]
    sns.barplot(y=f.index, x=f, palette=palette, ax=plt.subplot(122)).set(ylabel='Feature by SPLIT');


def Corr(X, Y):
    """Correlation coefficient"""
    return np.corrcoef(np.nan_to_num(X), np.nan_to_num(Y))[0, 1] * 100


def Linear(X, Y, ax=None):
    """Linear regression chart"""
    ax = sns.regplot(x=X, y=Y, ax=ax, fit_reg=True, scatter_kws={'color': 'green', 'alpha': min(100 / len(X), 1)})
    # ax.axvline(0, color='gray', linestyle='--')
    # ax.axhline(0, color='gray', linestyle='--')
    return Corr(X, Y)


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


thousands_fmt = FuncFormatter(lambda x, pos: "{:.0f}k".format(abs(x) / 1000))
millions_fmt = FuncFormatter(lambda x, pos: "{:.0f}m".format(abs(x) / 1e6))


def favicon(url):
    """Change favicon for Jupyter page"""
    display(Javascript('''
        var link = document.createElement('link'), oldLink = document.getElementById('dynamic-favicon');
        link.id = 'dynamic-favicon';
        link.rel = 'shortcut icon';
        link.href = "%s";
        if (oldLink) {
          document.head.removeChild(oldLink);
        }
        document.head.appendChild(link);
    ''' % url
                       ))


_RMSE = []
_LastTime = datetime.now()


def chart_callback(env):
    """Callback function for LightGBM chart"""
    global _LastTime, _RMSE
    c = env.evaluation_result_list
    T, V = c[0], c[1]
    _RMSE.append([T[2], V[2]])

    if datetime.now() > _LastTime + timedelta(seconds=2):
        clear_output(wait=True)
        plt.plot(_RMSE)
        plt.legend([
            '{0} {1} = {2:.5f}'.format(T[0], T[1], T[2]),
            '{0} {1} = {2:.5f}'.format(V[0], V[1], V[2]),
        ], loc='upper right', shadow=True)
        plt.title('Best {0} {1} = {2:.5f}'.format(V[0], V[1], np.array(_RMSE)[:, 1].min()))
        plt.show(block=False)
        _LastTime = datetime.now()


chart_callback.before_iteration = False
chart_callback.order = 0

# global visualization options
plt.rcParams['figure.dpi'] = 80
plt.rcParams['figure.subplot.left'] = 0.04
plt.rcParams['figure.subplot.right'] = 0.95
plt.rcParams['figure.subplot.top'] = 0.96
plt.rcParams['figure.subplot.bottom'] = 0.05
plt.rcParams['figure.figsize'] = (16, 5)

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

plt.rcParams['axes.spines.top'] = False
plt.rcParams['grid.color'] = 'lightgray'

pd.set_option('display.max_columns', 2000)
#pd.set_option('precision', 2)

np.set_printoptions(precision=4, linewidth=160, edgeitems=5)
