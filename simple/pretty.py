# -*- coding: utf-8 -*-
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
from numpy.typing import NDArray
from itertools import product, zip_longest, cycle, product
from math import ceil, sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from .utils import pmap, plist  # compatibility fix


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

def add_vlines(indices: list, props = 'border-left: 1px solid lightgray;') -> list:
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


def pp(X: pd.DataFrame, float_format: str | None = None, h_subset=None) -> Styler:
    """Pretty print pandas dataframe with ability to stroke some cells"""

    if float_format is None:
        float_format = pd.options.display.float_format
    if float_format is None:
        float_format = '{:,.2f}'
    x = X.style.format(float_format).apply(background, axis=None)
    return x if h_subset is None else x.apply(hline, axis=1, subset=h_subset)


def pg(X: pd.DataFrame, float_format: str | None = None) -> Styler:
    """Pretty print pandas dataframe with vertical lines are drawn on the groups"""
    data = [(i, c[0]) for i, c in enumerate(X.columns)]
    vlines = [0] + [index for index in range(1, len(data)) if data[index][1] != data[index - 1][1]]
    numeric_cols = X.select_dtypes(include=['float']).columns

    if float_format is None:
        float_format = pd.options.display.float_format
    if float_format is None:
        float_format = "{:.2f}"

    return X.style.format({
        col: float_format for col in numeric_cols
    }).set_table_styles(
        [{"selector": "td", "props": [("white-space", "nowrap")]}] +
        add_vlines(vlines)
    )


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


def rnd(value, prec=4):
    """Round value if possible"""
    return round(value, prec) if isinstance(value, float) else value


def plotHeatmaps(df: NDArray, x_name: str, y_name: str, value_name: str,
                 z_name: str = '', g_name: str = '', 
                 value_max: float = None,
                 fig_width=16, 
                 text_color='blue', stroke: bool = False,
                 labels: bool = True) -> plt.figure:
    """
    Create grid figure with heatmap subplots.
    
    Parameters:
        df (NDArray or DataFrame): The input data
        x_name (str): The name of the column to be used as x-axis for each subplot
        y_name (str): The name of the column to be used as y-axis for each subplot
        value_name (str): The name of the column to be used as values
        z_name (str, optional): The name of the column to be used as z-value for grid
        g_name (str, optional): The name of the column to be used as g-value for grid
        value_max (float, optional): The maximum value for the color map
        fig_width (int, optional): The width of the figure
        text_color (str, optional): The color of the text
        stroke (bool|str, optional): Flag to add a black (or specified color) stroke to the text
    
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
        # one subplot per g-value (no grid, just rectangular sequential map)
        rows, cols = get_sides(len(G))
        one = df[df[g_name] == G[0]]
        param = G
    elif z_name and not g_name:
        # one subplot per z-value (one-row grid)
        rows, cols = 1, len(Z)
        one = df[df[z_name] == Z[0]]
        param = Z

    # determine width and height of figure
    pvt = pd.DataFrame(one).pivot(columns=x_name, index=y_name, values=value_name).values
    k1 = rows / cols
    k2 = pvt.shape[0] / pvt.shape[1]
    fig_height = fig_width * k1 * k2
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex='all', sharey='all')
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    values = df[value_name]
    h = value_max if value_max is not None else values[values > 0].max()
    if stroke:
        color = stroke if isinstance(stroke, str) else 'black'
        path_effects = [pe.Stroke(linewidth=1.75, foreground=color), pe.Normal()]
    else:
        path_effects = None

    # iterate over z/g-values (one value combination for each subplot)
    for p, ax in [(param[0], axs)] if isinstance(axs, plt.Axes) else tqdm(zip_longest(param, axs.flatten()), total=len(axs.flatten())):
        # filter one slice for each subplot
        if z_name and not g_name:
            one = df[df[z_name] == p]
            text = f'{z_name}={rnd(p)}\n' if p is not None else ''
        elif g_name and not z_name:
            one = df[df[g_name] == p]
            text = f'{g_name}={rnd(p)}\n' if p is not None else ''
        elif z_name and g_name:
            one = df[(df[z_name] == p[0]) & (df[g_name] == p[1])]
            text = f'{z_name}={rnd(p[0])}\n{g_name}={rnd(p[1])}\n'
        else:
            one = df
            text = ''
        pvt = pd.DataFrame(one).pivot(columns=x_name, index=y_name, values=value_name)
    
        if len(pvt) > 0:
            ax.imshow(pvt, cmap='RdYlGn', vmin=-h, vmax=h)
            
            # place text in the center of heatmap
            y, x = [(s-1)/2 for s in pvt.shape]
            text += f'\nmax({value_name})={pvt.values.max():,.1f}\nmean({value_name})={pvt.values.mean():,.1f}'
            ax.text(x, y, text, color=text_color, ha='center', va='center', path_effects=path_effects)

            # add x and y labels
            if labels:
                xlabels = rnd(pvt.columns.values)
                ylabels = rnd(pvt.index.values)
            else:
                xlabels = []
                ylabels = []

            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels)
            ax.set_yticks(range(len(ylabels)))
            ax.set_yticklabels(ylabels)

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
