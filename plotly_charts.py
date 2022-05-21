import pandas as pd
import numpy as np
import re
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from inspect import getfullargspec
from itertools import repeat, zip_longest
from ipywidgets import interact, widgets, interactive, HBox, VBox
from plotly_resampler import FigureWidgetResampler


def addLines(fig: go.FigureWidget, **line_styles):
    """Add line (or lines) to the figure"""
    for line_name in line_styles:
        line = line_styles[line_name]
        line_class = go.scattergl.Marker if line.get('mode') == 'markers' else go.scattergl.Line

        trace_dict = {}  # parameters not belong to the Line will be moved to upper-levels
        scatter_dict = {}
        line_dict = {}
        for param in line:
            if param in getfullargspec(line_class).args:
                line_dict[param] = line[param]
            elif param in getfullargspec(go.Scattergl).args:
                scatter_dict[param] = line[param]
            else:
                trace_dict[param] = line[param]

        scatter_dict['marker' if line_class == go.scattergl.Marker else 'line'] = line_dict
        fig.add_trace(go.Scattergl(name=line_name, **scatter_dict), limit_to_view=True, **trace_dict)


def chartFigure(height=700, rows=1, template='plotly_white', line_styles=None, **layout_kwargs) -> go.FigureWidget:
    """Create default chart widget with horizontal subplots"""

    specs = [[{"secondary_y": True}] for _ in range(rows)]
    if rows > 1:
        k = (0.1 + 0.1 * rows)
        row_heights = [1 - k] + list(repeat(k / (rows - 1), rows - 1))
    else:
        row_heights = None

    fig = FigureWidgetResampler(
        go.FigureWidget(make_subplots(rows=rows, cols=1, row_heights=row_heights,
                                      vertical_spacing=0.03, shared_xaxes=True, specs=specs)))

    fig.update_layout(autosize=True, height=height, template=template,
                      legend=dict(x=0.1, y=1, orientation="h"),
                      margin=dict(l=45, r=15, b=10, t=30, pad=3),
                      **layout_kwargs)

    if line_styles is not None:
        addLines(fig, **line_styles)

    fig.update_xaxes(spikemode='across+marker', spikedash='dot', spikethickness=2, spikesnap='cursor')
    #fig.update_traces(xaxis='x2')

    return fig


def updateLines(fig: go.FigureWidget, **line_data):
    """Update lines xy-values"""

    # strip html tags and some others auxiliary marks
    names = [re.sub('<[^<]+?>|\[.*\]|~.*|\s', '', s.name) for s in fig.data]
    with fig.batch_update():
        for line_name in filter(lambda name: name in names, line_data):
            k = names.index(line_name)
            line = line_data[line_name]
            if type(line) == dict:
                fig.hf_data[k]['x'] = line['x']
                fig.hf_data[k]['y'] = line['y']
            else:
                fig.hf_data[k]['x'] = np.arange(len(line))
                fig.hf_data[k]['y'] = line

        fig.reload_data()


def updateSliders(sliders: widgets, **values: dict):
    """Update slider values"""
    for slider in sliders:
        slider.value = values[slider.description]


def interactFigure(model: callable, line_styles: dict, height: int = 700, rows: int = 1, template='plotly_white') -> widgets:
    """Interactive chart with model's internal data-series and sliders to change parameters"""

    spec = getfullargspec(model)
    x = dict(zip_longest(reversed(spec.args), [] if spec.defaults is None else reversed(spec.defaults), fillvalue=1))
    defaults = dict(reversed(x.items()))
    fig = chartFigure(height=height, rows=rows, template=template, line_styles=line_styles)

    def update(**arg):
        updateLines(fig, **model(**arg)[1])

    sliders = interactive(update, **defaults).children[:-1]

    # first run with initial values
    param = {s.description: s.value for s in sliders}
    update(**param)

    return VBox([HBox(sliders), fig])


def chartParallel(X: pd.DataFrame) -> widgets:
    """Parallel coordinates plot for optimization results"""
    fig = go.FigureWidget(data=go.Parcoords(dimensions=[{'label': c, 'values': X[c]} for c in X.columns]))
    fig.update_layout(autosize=True, height=400, template='plotly_white', margin=dict(l=45, r=45, b=20, t=50, pad=3))
    return fig


def iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False


def scatter(Y, X=None, **scatter_kw) -> FigureWidgetResampler:
    """Simple scatterplot/line chart with dynamic resampling"""
    fig = FigureWidgetResampler(go.Figure(layout={'height':400, 'template': 'plotly_white',
                              'margin': dict(l=45, r=5, b=10, t=25, pad=3), 'legend_orientation': 'h'}))

    if iterable(Y) and iterable(Y[0]):
        for y in Y:
            fig.add_trace(go.Scattergl(**scatter_kw), hf_x=X, hf_y=y)
    else:
        fig.add_trace(go.Scattergl(**scatter_kw), hf_x=X, hf_y=Y)
    return fig
