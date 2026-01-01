"""
Colormap utilities for Ncorr visualization.

Equivalent to ncorr_util_colormap.m
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from typing import Tuple, Optional


def get_ncorr_colormap(name: str = "jet") -> LinearSegmentedColormap:
    """
    Get colormap for Ncorr visualization.

    Args:
        name: Colormap name. Options:
            - 'jet': Classic jet colormap (default)
            - 'ncorr': Custom Ncorr colormap
            - 'coolwarm': Diverging colormap
            - 'viridis': Perceptually uniform

    Returns:
        Matplotlib colormap
    """
    if name == "ncorr":
        # Custom Ncorr colormap (blue-cyan-green-yellow-red)
        colors = [
            (0.0, 0.0, 0.5),   # Dark blue
            (0.0, 0.0, 1.0),   # Blue
            (0.0, 0.5, 1.0),   # Cyan-blue
            (0.0, 1.0, 1.0),   # Cyan
            (0.0, 1.0, 0.5),   # Cyan-green
            (0.0, 1.0, 0.0),   # Green
            (0.5, 1.0, 0.0),   # Yellow-green
            (1.0, 1.0, 0.0),   # Yellow
            (1.0, 0.5, 0.0),   # Orange
            (1.0, 0.0, 0.0),   # Red
            (0.5, 0.0, 0.0),   # Dark red
        ]
        return LinearSegmentedColormap.from_list("ncorr", colors)

    elif name == "strain":
        # Strain visualization colormap (symmetric around zero)
        colors = [
            (0.0, 0.0, 0.5),   # Dark blue (compression)
            (0.0, 0.0, 1.0),   # Blue
            (0.5, 0.5, 1.0),   # Light blue
            (1.0, 1.0, 1.0),   # White (zero)
            (1.0, 0.5, 0.5),   # Light red
            (1.0, 0.0, 0.0),   # Red
            (0.5, 0.0, 0.0),   # Dark red (tension)
        ]
        return LinearSegmentedColormap.from_list("strain", colors)

    else:
        # Use matplotlib colormap
        return plt.get_cmap(name)


def apply_colormap(
    data: NDArray[np.float64],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "jet",
    nan_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
) -> NDArray[np.uint8]:
    """
    Apply colormap to data array.

    Args:
        data: 2D data array
        vmin: Minimum value for normalization (None = auto)
        vmax: Maximum value for normalization (None = auto)
        cmap_name: Colormap name
        nan_color: RGBA color for NaN values

    Returns:
        RGBA image as uint8 array (H x W x 4)
    """
    cmap = get_ncorr_colormap(cmap_name)

    # Handle NaN
    valid_mask = ~np.isnan(data)

    if not np.any(valid_mask):
        # All NaN
        result = np.zeros((*data.shape, 4), dtype=np.uint8)
        result[:, :] = [int(c * 255) for c in nan_color]
        return result

    # Auto range
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    # Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    normalized = norm(data)

    # Apply colormap
    rgba = cmap(normalized)
    rgba = (rgba * 255).astype(np.uint8)

    # Set NaN color
    rgba[~valid_mask] = [int(c * 255) for c in nan_color]

    return rgba


def create_colorbar(
    vmin: float,
    vmax: float,
    cmap_name: str = "jet",
    label: str = "",
    orientation: str = "vertical",
    num_ticks: int = 5,
) -> Tuple[NDArray, list]:
    """
    Create colorbar image and tick labels.

    Args:
        vmin: Minimum value
        vmax: Maximum value
        cmap_name: Colormap name
        label: Colorbar label
        orientation: 'vertical' or 'horizontal'
        num_ticks: Number of tick marks

    Returns:
        Tuple of (colorbar_image, tick_values)
    """
    cmap = get_ncorr_colormap(cmap_name)

    # Create gradient
    if orientation == "vertical":
        gradient = np.linspace(1, 0, 256).reshape(-1, 1)
        gradient = np.repeat(gradient, 30, axis=1)
    else:
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.repeat(gradient, 30, axis=0)

    # Apply colormap
    colorbar = (cmap(gradient) * 255).astype(np.uint8)

    # Calculate tick values
    tick_values = np.linspace(vmin, vmax, num_ticks)

    return colorbar, tick_values.tolist()


def overlay_data_on_image(
    image: NDArray[np.uint8],
    data: NDArray[np.float64],
    roi: NDArray[np.bool_],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "jet",
    alpha: float = 0.7,
) -> NDArray[np.uint8]:
    """
    Overlay data field on image.

    Args:
        image: Background image (H x W x 3)
        data: Data to overlay
        roi: Valid data mask
        vmin: Minimum value
        vmax: Maximum value
        cmap_name: Colormap name
        alpha: Overlay transparency (0-1)

    Returns:
        Blended image (H x W x 3)
    """
    # Ensure image is 3-channel
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Get colormap image
    data_rgba = apply_colormap(data, vmin, vmax, cmap_name)

    # Blend
    result = image.copy().astype(np.float64)

    for c in range(3):
        result[:, :, c] = np.where(
            roi,
            (1 - alpha) * result[:, :, c] + alpha * data_rgba[:, :, c],
            result[:, :, c]
        )

    return result.astype(np.uint8)
