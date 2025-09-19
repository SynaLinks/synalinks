# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps


def generate_distinct_colors(n):
    """Generate n distinct colors, maximizing perceptual difference.
    Args:
        n (int): The number of colors to generate.
    Returns:
        list: List of RGBA colors.
    """
    if n <= 20:
        # Use qualitative colormaps for small n
        cmap = colormaps["tab20"]
        colors = [cmap(i) for i in np.linspace(0, 1, n, endpoint=False)]
    else:
        # For larger n, sample hues in HSL space
        hues = np.linspace(0, 1, n, endpoint=False)
        colors = []
        for hue in hues:
            # Convert HSL to RGB (fixed saturation and lightness)
            rgb = mcolors.hsv_to_rgb([hue, 0.9, 0.7])
            colors.append(rgb)
    return colors
