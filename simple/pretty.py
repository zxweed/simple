# -*- coding: utf-8 -*-
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
from numpy.typing import NDArray
from typing import Union, List
from itertools import product, zip_longest, cycle, product
from math import ceil, sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from joblib import Parallel, delayed
from psutil import cpu_percent
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from multiprocessing.shared_memory import SharedMemory


class tqdmParallel(Parallel):
    """
    Enhanced Parallel class that shows a progress bar during parallel processing.
    This class extends joblib.Parallel to provide visual feedback on task completion.
    """

    def __init__(self, progress=True, total=None, postfix=None, n_jobs=-1, **kwargs):
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


def pmap(func: callable, *args, params: List[tuple] = None, **kwargs):
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
    desc : str, optional
        Progress bar description, by default ""
    **kwargs
        Additional arguments to be passed to the tqdmParallel
    """
    param_list = params if params is not None else plist(*args)
    with tqdmParallel(total=len(param_list), **kwargs) as P:
        FUNC = delayed(func)
        return P(FUNC(*tpl(param)) for param in param_list)


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


def Corr(X: NDArray, Y: NDArray) -> float:
    """Correlation coefficient"""
    return np.corrcoef(np.nan_to_num(X), np.nan_to_num(Y))[0, 1] * 100


def background(self, scale='Linear', cmap='RdYlGn', **css) -> pd.DataFrame:
    """For use with `DataFrame.style.apply` this function will apply a heatmap
       color gradient *elementwise* to the calling DataFrame

    self : pd.DataFrame
        The calling DataFrame. This argument is automatically passed in by the `DataFrame.style.apply` method

    cmap : colormap or str
        The colormap to use

    css : dict
        Any extra inline css key/value pars to pass to the styler

    Returns
    -------
    pd.DataFrame
        The css styles to apply to the calling DataFrame
    """
    cvals = self.fillna(0).values.ravel().copy()

    q = np.percentile(cvals, 99.9)
    vmax = max(cvals.max(), abs(np.clip(cvals, -q, q).min()))
    if scale == 'Linear':
        # Linear colorscale
        norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax, clip=True)
    else:
        # Logarithmic color scale
        norm = mpl.colors.SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax, clip=True, base=2.17)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    styles = []
    for c in cvals:
        rgb = mapper.to_rgba(c)
        style = ["{}: {}".format(key.replace('_', '-'), value) for key, value in css.items()]
        style.append("background-color: {}".format(mpl.colors.rgb2hex(rgb)))
        styles.append('; '.join(style))
    styles = np.asarray(styles).reshape(self.shape)
    return pd.DataFrame(styles, index=self.index, columns=self.columns)


def hline(row, color='black', width='1px'): 
    return [f"border-bottom: {width} solid {color};" for _ in row]

def add_vlines(indices: list, props = 'border-left: 1px solid lightgray;') -> Styler:
    """
    Add vertical lines to the styled DataFrame
    Usage example:
        P.style.format(custom_format).set_table_styles(add_vlines([1,5,6]))
    """
    styles = []
    for i in indices:
        styles.append({'selector': f'th.col{i}', 'props': props})
        styles.append({'selector': f'td.col{i}', 'props': props})
    return styles


def pp(X: pd.DataFrame, float_format: str = None, h_subset=None) -> Styler:
    """Pretty print pandas dataframe with ability to stroke some cells"""

    if float_format is None:
        float_format = pd.options.display.float_format
    if float_format is None:
        float_format = '{:,.2f}'
    x = X.style.format(float_format).apply(background, axis=None)
    return x if h_subset is None else x.apply(hline, axis=1, subset=h_subset)


def get_sides(n: int) -> tuple:
    """
    Returns the width and height of a rectangle that contains n subplots.

    Parameters:
    n (int): The number of subplots.

    Returns:
    tuple: A tuple containing the width and height of the rectangle.
    """
    # Calculate the initial values for x and y
    x = ceil(sqrt(n))
    y = x

    # Decrease y until the rectangle fits the number of subplots
    while x * y >= n:
        y -= 1

    # Ensure the rectangle is not too small
    y += 1

    # Adjust x and y if the rectangle is too large
    if (x - 1) * (y + 1) >= n:
        x -= 1
        y += 1

    return x, y


def plotHeatmaps(df: NDArray, x_name: str, y_name: str, value_name: str,
                 z_name: str = None, g_name: str = None, 
                 value_max: float = None,
                 fig_width=16, text_color='blue', stroke: bool = False) -> plt.figure:
    """
    Create grid figure with heatmap subplots.

    Parameters:
        df (NDArray or DataFrame): The input data.
        x_name (str): The name of the column to be used as x-axis for each subplot.
        y_name (str): The name of the column to be used as y-axis for each subplot.
        value_name (str): The name of the column to be used as values.
        z_name (str, optional): The name of the column to be used as z-value for grid.
        g_name (str, optional): The name of the column to be used as g-value for grid.
        value_max (float, optional): The maximum value for the color map.
        fig_width (int, optional): The width of the figure.
        text_color (str, optional): The color of the text.
        stroke (bool, optional): Whether to add a stroke to the text.

    Returns:
        plt.figure: The figure object.
    """

    Z = np.unique(df[z_name]) if z_name else None
    G = np.unique(df[g_name]) if g_name else None

    # determine number of rows and columns
    if not z_name and not g_name:
        # single plot
        rows, cols = 1, 1
        one = df
        param = [0]
    elif z_name and g_name:
        # subplot grid by z-value and g-value
        rows, cols = len(Z), len(G)
        one = df[(df[z_name] == Z[0]) & (df[g_name] == G[0])]
        param = product(Z, G)
    elif g_name and not z_name:
        # one subplot per z-value (no grid, just rectangular sequence)
        rows, cols = get_sides(len(G))
        one = df[df[g_name] == G[0]]
        param = G
    elif z_name and not g_name:
        # one subplot per g-value (no grid, just rectangular sequence)
        rows, cols = get_sides(len(Z))
        one = df[df[z_name] == Z[0]]
        param = Z

    # determine width and height of figure
    pvt = one.pivot(columns=x_name, index=y_name, values=value_name).values
    k1 = rows / cols
    k2 = pvt.shape[0] / pvt.shape[1]
    fig_height = fig_width * k1 * k2
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex='all', sharey='all')
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    values = df[value_name]
    h = value_max if value_max is not None else values[values > 0].max()
    path_effects = [pe.Stroke(linewidth=2, foreground='black'), pe.Normal()] if stroke else None

    # iterate over z/g-values (one value combination for each subplot)
    for p, ax in [(param[0], axs)] if isinstance(axs, plt.Axes) else tqdm(zip_longest(param, axs.flatten()), total=len(axs.flatten())):
        # filter one slice for each subplot
        if z_name and not g_name:
            one = df[df[z_name] == p]
            text = f'{z_name}={p}\n'
        elif g_name and not z_name:
            one = df[df[g_name] == p]
            text = f'{g_name}={p}\n'
        elif z_name and g_name:
            one = df[(df[z_name] == p[0]) & (df[g_name] == p[1])]
            text = f'{z_name}={p[0]}\n{g_name}={p[1]}\n'
        else:
            one = df
            text = ''
        pvt = one.pivot(columns=x_name, index=y_name, values=value_name).values

        if len(pvt) > 0:
            ax.imshow(pvt, cmap='RdYlGn', vmin=-h, vmax=h)

            # place text in the center of heatmap
            y, x = [(s-1)/2 for s in pvt.shape]
            text += f'\nmax({value_name})={pvt.max():,.1f}\nmean({value_name})={pvt.mean():,.1f}'
            ax.text(x, y, text, color=text_color, ha='center', va='center', path_effects=path_effects)
            
            # add x and y labels
            xlabels = one[x_name].unique()
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels)

            ylabels = one[y_name].unique()
            ax.set_yticks(range(len(ylabels)))
            ax.set_yticklabels(one[y_name].unique())

        # the number of subplots can be more than the data, but it is necessary to set parameters for them too
#        ax.tick_params(left=True, right=False, labelleft=True, labelbottom=True, bottom=True)

    plt.close(fig)
    return fig


def plotImportance(model, names=None, top=20, palette='Blues_r', ax=None):
    """
    Plot the feature importance chart.

    Args:
        model: The machine learning model.
        names: A list of feature names. If None, it will check if the model has a 'feature_names_' attribute.
        top: The number of top features to display.
        palette: The color palette for the barplot.
        ax: The axis to plot the chart on. If None, a new axis will be created.
    """
    # Check if the model has a 'feature_names_' attribute and names is None
    names = model.feature_names_ if hasattr(model, 'feature_names_') and names is None else names

    # Check if the model has a 'feature_importances_' attribute
    if hasattr(model, 'feature_importances_'):
        # Sort the feature importances in descending order and select the top features
        f = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)[:top]
        sns.barplot(y=f.index, hue=f.index, x=f, palette=palette, ax=ax).set(ylabel='Feature importance')
    else:
         print("Selected model does not support feature_importances_")


def getROC(model, X_test, y_test):
    """Calculate the Receiver Operating Characteristic (ROC) curve for a multi-class classification model.

    Parameters:
        model (object): The trained classification model.
        X_test (array-like): The feature matrix of the test set.
        y_test (array-like): The true labels of the test set.

    Returns dictionary containing the false positive rate (fpr), true positive rate (tpr)
    and the area under the ROC curve (roc_auc) for each class.
    """
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_binarized.shape[1]
    fpr = [0] * n_classes
    tpr = [0] * n_classes
    roc_auc = [0] * n_classes
    y_scores = model.predict_proba(X_test)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def plotROC(model, X_test, y_test):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a multi-class classification model.

    Parameters:
        model (object): The trained classification model.
        X_test (array-like): The feature matrix of the test set.
        y_test (array-like): The true labels of the test set.

    Returns:
        fig (object): The matplotlib figure object containing the ROC curve and the confusion matrix.
    """
    fpr, tpr, roc_auc = getROC(model, X_test, y_test)
    n_classes = len(roc_auc)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    colors = cycle(['blue', 'red', 'green', 'orange', 'cyan', 'magenta'])

    # plot ROC chart
    for i, color in zip(range(n_classes), colors):
        label = 'ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i])
        ax[0].plot(fpr[i], tpr[i], color=color, lw=2, alpha=0.5, label=label)

    ax[0].plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Multi-class ROC')
    ax[0].legend(loc="lower right")

    # plot confusion matrix
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt="g", ax=ax[1], linewidths=3, linecolor='white');

    ax[1].set_title('Confusion Matrix')
    ax[1].set_xlabel('Predicted Label')
    ax[1].set_ylabel('True Label')

    plt.close(fig)
    return fig
