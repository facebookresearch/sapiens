# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def get_adaptive_scale(img_shape: Tuple[int, int],
                       min_scale: float = 0.3,
                       max_scale: float = 3.0) -> float:
    """Get adaptive scale according to image shape.

    The target scale depends on the the short edge length of the image. If the
    short edge length equals 224, the output is 1.0. And output linear scales
    according the short edge length.

    You can also specify the minimum scale and the maximum scale to limit the
    linear scale.

    Args:
        img_shape (Tuple[int, int]): The shape of the canvas image.
        min_size (int): The minimum scale. Defaults to 0.3.
        max_size (int): The maximum scale. Defaults to 3.0.

    Returns:
        int: The adaptive scale.
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)


def create_figure(*args, margin=False, **kwargs) -> 'Figure':
    """Create a independent figure.

    Different from the :func:`plt.figure`, the figure from this function won't
    be managed by matplotlib. And it has
    :obj:`matplotlib.backends.backend_agg.FigureCanvasAgg`, and therefore, you
    can use the ``canvas`` attribute to get access the drawn image.

    Args:
        *args: All positional arguments of :class:`matplotlib.figure.Figure`.
        margin: Whether to reserve the white edges of the figure.
            Defaults to False.
        **kwargs: All keyword arguments of :class:`matplotlib.figure.Figure`.

    Return:
        matplotlib.figure.Figure: The created figure.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(*args, **kwargs)
    FigureCanvasAgg(figure)

    if not margin:
        # remove white edges by set subplot margin
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return figure
