# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Colormap related constants and functions."""

from enum import Enum, unique
from typing import Final, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from av2.utils.typing import NDArrayByte, NDArrayFloat

RED_HEX: Final[str] = "#df0101"
GREEN_HEX: Final[str] = "#31b404"

RED_RGB: Final[Tuple[int, int, int]] = (255, 0, 0)
RED_BGR: Final[Tuple[int, int, int]] = RED_RGB[::-1]

BLUE_RGB: Final[Tuple[int, int, int]] = (0, 0, 255)
BLUE_BGR: Final[Tuple[int, int, int]] = BLUE_RGB[::-1]

HANDICAP_BLUE_RGB: Final[Tuple[int, int, int]] = (42, 130, 193)
HANDICAP_BLUE_BGR: Final[Tuple[int, int, int]] = HANDICAP_BLUE_RGB[::-1]

WHITE_RGB: Final[Tuple[int, int, int]] = (255, 255, 255)
WHITE_BGR: Final[Tuple[int, int, int]] = WHITE_RGB[::-1]

GRAY_BGR: Final[Tuple[int, int, int]] = (168, 168, 168)
DARK_GRAY_BGR: Final[Tuple[int, int, int]] = (100, 100, 100)

TRAFFIC_YELLOW1_RGB: Final[Tuple[int, int, int]] = (250, 210, 1)
TRAFFIC_YELLOW1_BGR: Final[Tuple[int, int, int]] = TRAFFIC_YELLOW1_RGB[::-1]


@unique
class ColorFormats(str, Enum):
    """Color channel formats."""

    BGR = "BGR"
    RGB = "RGB"


def create_range_map(points_xyz: NDArrayFloat) -> NDArrayByte:
    """Generate an RGB colormap as a function of the lidar range.

    Args:
        points_xyz: (N,3) Points (x,y,z).

    Returns:
        (N,3) RGB colormap.
    """
    range = points_xyz[..., 2]
    range = np.round(range).astype(int)
    color = plt.get_cmap("turbo")(np.arange(0, range.max() + 1))
    color = color[range]
    range_cmap: NDArrayByte = (color * 255.0).astype(np.uint8)
    return range_cmap


def create_colormap(color_list: Sequence[str], n_colors: int) -> NDArrayFloat:
    """Create hex colorscale to interpolate between requested colors.

    Args:
        color_list: list of requested colors, in hex format.
        n_colors: number of colors in the colormap.

    Returns:
        array of shape (n_colors, 3) representing a list of RGB colors in [0,1]
    """
    cmap = LinearSegmentedColormap.from_list(name="dummy_name", colors=color_list)
    colorscale: NDArrayFloat = np.array([cmap(k * 1 / n_colors) for k in range(n_colors)])
    # ignore the 4th alpha channel
    return colorscale[:, :3]
