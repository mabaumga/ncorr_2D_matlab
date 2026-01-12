"""
B-spline interpolation for Ncorr.

Implements biquintic B-spline interpolation for subpixel accuracy.
Equivalent to ncorr_alg_interpqbs.m and related functions.

All Numba-accelerated functions are at module level for optimal performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from scipy.fft import fft, ifft


# =============================================================================
# Module-level Numba-accelerated functions
# =============================================================================

@njit(cache=True, fastmath=True)
def _quintic_bspline(t: float) -> float:
    """
    Evaluate quintic B-spline basis function at t.

    The quintic B-spline is defined over [-3, 3] and is zero outside.

    This uses the standard centered quintic B-spline that satisfies:
    - B(0) = 11/20 = 0.55
    - B(1) = 13/60 ≈ 0.217
    - B(2) = 1/120 ≈ 0.00833
    - Sum over all integers = 1 (partition of unity)

    Derived from: B_5(x) = sum_{k=0}^{6} C(6,k) * (-1)^k * max(x-k+3, 0)^5 / 120
    """
    at = abs(t)

    if at >= 3.0:
        return 0.0
    elif at >= 2.0:
        # (3-|t|)^5 / 120
        tmp = 3.0 - at
        return tmp * tmp * tmp * tmp * tmp / 120.0
    elif at >= 1.0:
        # Correct formula for 1 <= |t| < 2:
        # B(t) = (5|t|^5 - 45|t|^4 + 150|t|^3 - 210|t|^2 + 75|t| + 51) / 120
        # Verified: B(1) = (5-45+150-210+75+51)/120 = 26/120 = 13/60 ✓
        t2 = at * at
        t3 = t2 * at
        t4 = t3 * at
        t5 = t4 * at
        return (5.0 * t5 - 45.0 * t4 + 150.0 * t3 - 210.0 * t2 + 75.0 * at + 51.0) / 120.0
    else:
        # |t| < 1: B(t) = (66 - 60t^2 + 30t^4 - 10|t|^5) / 120
        # Verified: B(0) = 66/120 = 11/20 ✓
        #           B(1-) = (66-60+30-10)/120 = 26/120 = 13/60 ✓ (continuous!)
        t2 = at * at
        t4 = t2 * t2
        t5 = t4 * at
        return (66.0 - 60.0 * t2 + 30.0 * t4 - 10.0 * t5) / 120.0


@njit(cache=True, fastmath=True)
def _quintic_bspline_derivative(t: float) -> float:
    """
    Evaluate derivative of quintic B-spline basis function at t.

    Derivatives of the corrected B-spline formulas:
    - |t| < 1: d/dt[(66 - 60t² + 30t⁴ - 10|t|⁵)/120] = sign(t)*(-120|t| + 120|t|³ - 50|t|⁴)/120
    - 1 <= |t| < 2: d/dt[(5|t|⁵ - 45|t|⁴ + 150|t|³ - 210|t|² + 75|t| + 51)/120]
                  = sign(t)*(25|t|⁴ - 180|t|³ + 450|t|² - 420|t| + 75)/120
    - 2 <= |t| < 3: d/dt[(3-|t|)⁵/120] = -sign(t)*(3-|t|)⁴/24
    """
    sign = 1.0 if t >= 0.0 else -1.0
    at = abs(t)

    if at >= 3.0:
        return 0.0
    elif at >= 2.0:
        # d/dt[(3-|t|)^5/120] = -sign(t) * 5*(3-|t|)^4 / 120 = -sign(t) * (3-|t|)^4 / 24
        tmp = 3.0 - at
        return -sign * tmp * tmp * tmp * tmp / 24.0
    elif at >= 1.0:
        # d/dt[(5|t|^5 - 45|t|^4 + 150|t|^3 - 210|t|^2 + 75|t| + 51)/120]
        # = sign(t) * (25|t|^4 - 180|t|^3 + 450|t|^2 - 420|t| + 75) / 120
        t2 = at * at
        t3 = t2 * at
        t4 = t3 * at
        return sign * (25.0 * t4 - 180.0 * t3 + 450.0 * t2 - 420.0 * at + 75.0) / 120.0
    else:
        # d/dt[(66 - 60t^2 + 30t^4 - 10|t|^5)/120]
        # = sign(t) * (-120|t| + 120|t|^3 - 50|t|^4) / 120
        t2 = at * at
        t3 = t2 * at
        t4 = t3 * at
        return sign * (-120.0 * at + 120.0 * t3 - 50.0 * t4) / 120.0


@njit(cache=True, fastmath=True)
def interpolate_point(x: float, y: float, bcoef: NDArray[np.float64]) -> float:
    """
    Interpolate value at a single point using B-spline coefficients.

    Args:
        x: X-coordinate in bcoef coordinates
        y: Y-coordinate in bcoef coordinates
        bcoef: B-spline coefficients array

    Returns:
        Interpolated value (NaN if out of bounds)
    """
    h, w = bcoef.shape

    if x < 2.0 or x >= w - 3.0 or y < 2.0 or y >= h - 3.0:
        return np.nan

    ix = int(x)
    iy = int(y)
    fx = x - ix
    fy = y - iy

    value = 0.0
    for j in range(-2, 4):
        by = _quintic_bspline(fy - j)
        for i in range(-2, 4):
            bx = _quintic_bspline(fx - i)
            value += bcoef[iy + j, ix + i] * bx * by

    return value


@njit(cache=True, fastmath=True)
def interpolate_point_with_gradient(
    x: float, y: float, bcoef: NDArray[np.float64]
) -> tuple:
    """
    Interpolate value and gradient at a single point.

    Args:
        x: X-coordinate in bcoef coordinates
        y: Y-coordinate in bcoef coordinates
        bcoef: B-spline coefficients array

    Returns:
        Tuple of (value, dx, dy)
    """
    h, w = bcoef.shape

    if x < 2.0 or x >= w - 3.0 or y < 2.0 or y >= h - 3.0:
        return np.nan, np.nan, np.nan

    ix = int(x)
    iy = int(y)
    fx = x - ix
    fy = y - iy

    value = 0.0
    dx = 0.0
    dy = 0.0

    for j in range(-2, 4):
        by = _quintic_bspline(fy - j)
        dby = _quintic_bspline_derivative(fy - j)

        for i in range(-2, 4):
            bx = _quintic_bspline(fx - i)
            dbx = _quintic_bspline_derivative(fx - i)

            coef = bcoef[iy + j, ix + i]
            value += coef * bx * by
            dx += coef * dbx * by
            dy += coef * bx * dby

    return value, dx, dy


@njit(cache=True, parallel=True, fastmath=True)
def interpolate_subset(
    bcoef: NDArray[np.float64],
    center_x: float,
    center_y: float,
    radius: int,
    border: int,
) -> NDArray[np.float64]:
    """
    Interpolate a square subset of pixels centered at (center_x, center_y).

    Vectorized version that computes all pixels in parallel.

    Args:
        bcoef: B-spline coefficients array
        center_x: X-coordinate of subset center (image coordinates)
        center_y: Y-coordinate of subset center (image coordinates)
        radius: Subset radius
        border: Border offset in bcoef

    Returns:
        2D array of interpolated values (size: 2*radius+1 x 2*radius+1)
        NaN for out-of-bounds pixels
    """
    size = 2 * radius + 1
    subset = np.empty((size, size), dtype=np.float64)
    h, w = bcoef.shape

    for j in prange(size):
        for i in range(size):
            px = center_x - radius + i + border
            py = center_y - radius + j + border

            if px < 2.0 or px >= w - 3.0 or py < 2.0 or py >= h - 3.0:
                subset[j, i] = np.nan
                continue

            ix = int(px)
            iy = int(py)
            fx = px - ix
            fy = py - iy

            value = 0.0
            for jj in range(-2, 4):
                by = _quintic_bspline(fy - jj)
                for ii in range(-2, 4):
                    bx = _quintic_bspline(fx - ii)
                    value += bcoef[iy + jj, ix + ii] * bx * by

            subset[j, i] = value

    return subset


@njit(cache=True, parallel=True, fastmath=True)
def interpolate_subset_with_gradients(
    bcoef: NDArray[np.float64],
    center_x: float,
    center_y: float,
    radius: int,
    border: int,
) -> tuple:
    """
    Interpolate a subset with gradients.

    Vectorized version that computes all pixels and gradients in parallel.

    Args:
        bcoef: B-spline coefficients array
        center_x: X-coordinate of subset center (image coordinates)
        center_y: Y-coordinate of subset center (image coordinates)
        radius: Subset radius
        border: Border offset in bcoef

    Returns:
        Tuple of (subset, dx, dy, mask) arrays
    """
    size = 2 * radius + 1
    subset = np.empty((size, size), dtype=np.float64)
    dx = np.empty((size, size), dtype=np.float64)
    dy = np.empty((size, size), dtype=np.float64)
    mask = np.ones((size, size), dtype=np.bool_)
    h, w = bcoef.shape

    for j in prange(size):
        for i in range(size):
            px = center_x - radius + i + border
            py = center_y - radius + j + border

            if px < 2.0 or px >= w - 3.0 or py < 2.0 or py >= h - 3.0:
                subset[j, i] = 0.0
                dx[j, i] = 0.0
                dy[j, i] = 0.0
                mask[j, i] = False
                continue

            ix = int(px)
            iy = int(py)
            fx = px - ix
            fy = py - iy

            value = 0.0
            grad_x = 0.0
            grad_y = 0.0

            for jj in range(-2, 4):
                by = _quintic_bspline(fy - jj)
                dby = _quintic_bspline_derivative(fy - jj)

                for ii in range(-2, 4):
                    bx = _quintic_bspline(fx - ii)
                    dbx = _quintic_bspline_derivative(fx - ii)

                    coef = bcoef[iy + jj, ix + ii]
                    value += coef * bx * by
                    grad_x += coef * dbx * by
                    grad_y += coef * bx * dby

            subset[j, i] = value
            dx[j, i] = grad_x
            dy[j, i] = grad_y

    return subset, dx, dy, mask


@njit(cache=True, parallel=True, fastmath=True)
def interpolate_values_batch(
    points: NDArray[np.float64],
    bcoef: NDArray[np.float64],
    left_offset: int,
    top_offset: int,
    border: int,
) -> NDArray[np.float64]:
    """
    Interpolate values at multiple points using B-spline coefficients.

    Args:
        points: Nx2 array of (x, y) coordinates to interpolate
        bcoef: B-spline coefficients array
        left_offset: Left bound offset of the data region
        top_offset: Top bound offset of the data region
        border: Border padding in bcoef

    Returns:
        Interpolated values at each point (NaN if out of range)
    """
    n = points.shape[0]
    result = np.empty(n, dtype=np.float64)
    h, w = bcoef.shape

    for idx in prange(n):
        x = points[idx, 0] - left_offset + border
        y = points[idx, 1] - top_offset + border

        if x < 2.0 or x >= w - 3.0 or y < 2.0 or y >= h - 3.0:
            result[idx] = np.nan
            continue

        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy

        value = 0.0
        for j in range(-2, 4):
            by = _quintic_bspline(fy - j)
            for i in range(-2, 4):
                bx = _quintic_bspline(fx - i)
                value += bcoef[iy + j, ix + i] * bx * by

        result[idx] = value

    return result


# =============================================================================
# Class wrapper for backwards compatibility
# =============================================================================

class BSplineInterpolator:
    """
    Biquintic B-spline interpolator for subpixel image values.

    This class provides a wrapper around module-level Numba functions
    for backwards compatibility.
    """

    # B-spline basis function coefficients (6th order = quintic)
    _KERNEL = np.array([1/120, 13/60, 11/20, 13/60, 1/120], dtype=np.float64)

    @staticmethod
    def compute_bcoef(data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute biquintic B-spline coefficients.

        Args:
            data: Input 2D array (must be at least 5x5 or empty)

        Returns:
            B-spline coefficients (same size as input)
        """
        if data.size == 0:
            return np.zeros_like(data)

        if data.shape[0] < 5 or data.shape[1] < 5:
            raise ValueError(
                "Array for B-spline coefficients must be >= 5x5 or empty"
            )

        kernel = BSplineInterpolator._KERNEL
        h, w = data.shape
        result = np.zeros_like(data)

        # Process rows
        kernel_x = np.zeros(w, dtype=np.complex128)
        kernel_x[:3] = kernel[2:]
        kernel_x[-2:] = kernel[:2]
        kernel_x_fft = fft(kernel_x)

        for i in range(h):
            result[i, :] = np.real(ifft(fft(data[i, :]) / kernel_x_fft))

        # Process columns
        kernel_y = np.zeros(h, dtype=np.complex128)
        kernel_y[:3] = kernel[2:]
        kernel_y[-2:] = kernel[:2]
        kernel_y_fft = fft(kernel_y)

        for j in range(w):
            result[:, j] = np.real(ifft(fft(result[:, j]) / kernel_y_fft))

        return result

    @staticmethod
    def interpolate_with_gradient(
        x: float,
        y: float,
        bcoef: NDArray[np.float64],
        border: int = 20,
    ) -> tuple:
        """
        Interpolate value and gradient at a single point.

        Wrapper around module-level function for compatibility.
        """
        return interpolate_point_with_gradient(x, y, bcoef)

    @staticmethod
    def interpolate_values(
        points: NDArray[np.float64],
        bcoef: NDArray[np.float64],
        left_offset: int,
        top_offset: int,
        border: int,
    ) -> NDArray[np.float64]:
        """
        Interpolate values at given points using B-spline coefficients.
        """
        return interpolate_values_batch(points, bcoef, left_offset, top_offset, border)

    @staticmethod
    def interpolate_displacement_field(
        coords: NDArray[np.float64],
        u_bcoef: NDArray[np.float64],
        v_bcoef: NDArray[np.float64],
        left_offset: int,
        top_offset: int,
        border: int,
    ) -> tuple:
        """
        Interpolate displacement field at given coordinates.
        """
        u_interp = interpolate_values_batch(
            coords, u_bcoef, left_offset, top_offset, border
        )
        v_interp = interpolate_values_batch(
            coords, v_bcoef, left_offset, top_offset, border
        )
        return u_interp, v_interp


# =============================================================================
# Convenience function
# =============================================================================

def interpolate_qbs(
    coords: NDArray[np.float64],
    bcoef: NDArray[np.float64],
    left_offset: int,
    top_offset: int,
    border: int,
) -> NDArray[np.float64]:
    """
    Convenience function for B-spline interpolation.

    Equivalent to ncorr_alg_interpqbs.m
    """
    return interpolate_values_batch(coords, bcoef, left_offset, top_offset, border)
