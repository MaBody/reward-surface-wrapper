import plotly
from plotly.graph_objects import Figure
from numpy.typing import ArrayLike


def plot_surface(dim1: ArrayLike, dim2: ArrayLike, loss: ArrayLike) -> Figure:
    # create and return plotly figure
    fig = plotly.graph_objects.Figure()
    return fig
