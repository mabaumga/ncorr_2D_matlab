"""
B-spline interpolation for Ncorr.

Implements biquintic B-spline interpolation for subpixel accuracy.
Equivalent to ncorr_alg_interpqbs.m and related functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import jit, prange
from scipy.fft import fft, ifft


class BSplineInterpolator:
    """
    Biquintic B-spline interpolator for subpixel image values.

    Uses quintic B-splines for smooth interpolation of image data,
    which is essential for subpixel accuracy in DIC.
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

        Raises:
            ValueError: If array is smaller than 5x5 and not empty
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
    @jit(nopython=True, cache=True)
    def _quintic_bspline(t: float) -> float:
        """
        Evaluate quintic B-spline basis function at t.

        The quintic B-spline is defined over [-3, 3] and is zero outside.

        Args:
            t: Position to evaluate (should be in [-3, 3])

        Returns:
            B-spline value at t
        """
        t = abs(t)

        if t >= 3.0:
            return 0.0
        elif t >= 2.0:
            return (3.0 - t) ** 5 / 120.0
        elif t >= 1.0:
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            t5 = t4 * t
            return (-5 * t5 + 75 * t4 - 450 * t3 + 1350 * t2 - 2025 * t + 1215) / 120.0
        else:
            t2 = t * t
            t4 = t2 * t2
            return (33 - 30 * t2 + 5 * t4) / 60.0 - t2 * (1.0 - t2) / 12.0

    @staticmethod
    @jit(nopython=True, cache=True)
    def _quintic_bspline_derivative(t: float) -> float:
        """
        Evaluate derivative of quintic B-spline basis function at t.

        Args:
            t: Position to evaluate

        Returns:
            Derivative of B-spline at t
        """
        sign = 1.0 if t >= 0 else -1.0
        t = abs(t)

        if t >= 3.0:
            return 0.0
        elif t >= 2.0:
            return -sign * (3.0 - t) ** 4 / 24.0
        elif t >= 1.0:
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            return sign * (-25 * t4 + 300 * t3 - 1350 * t2 + 2700 * t - 2025) / 120.0
        else:
            t3 = t * t * t
            return sign * (-60 * t + 20 * t3) / 60.0

    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def interpolate_values(
        points: NDArray[np.float64],
        bcoef: NDArray[np.float64],
        left_offset: int,
        top_offset: int,
        border: int,
    ) -> NDArray[np.float64]:
        """
        Interpolate values at given points using B-spline coefficients.

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

            # Check bounds
            if x < 2 or x >= w - 3 or y < 2 or y >= h - 3:
                result[idx] = np.nan
                continue

            # Integer and fractional parts
            ix = int(x)
            iy = int(y)
            fx = x - ix
            fy = y - iy

            # Evaluate B-spline
            value = 0.0
            for j in range(-2, 4):
                by = BSplineInterpolator._quintic_bspline(fy - j)
                for i in range(-2, 4):
                    bx = BSplineInterpolator._quintic_bspline(fx - i)
                    value += bcoef[iy + j, ix + i] * bx * by

            result[idx] = value

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def interpolate_with_gradient(
        x: float,
        y: float,
        bcoef: NDArray[np.float64],
        border: int = 20,
    ) -> tuple:
        """
        Interpolate value and gradient at a single point.

        Args:
            x: X-coordinate (in bcoef coordinates with border)
            y: Y-coordinate (in bcoef coordinates with border)
            bcoef: B-spline coefficients
            border: Border padding

        Returns:
            Tuple of (value, dx, dy) - interpolated value and gradients
        """
        h, w = bcoef.shape

        if x < 2 or x >= w - 3 or y < 2 or y >= h - 3:
            return np.nan, np.nan, np.nan

        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy

        value = 0.0
        dx = 0.0
        dy = 0.0

        for j in range(-2, 4):
            by = BSplineInterpolator._quintic_bspline(fy - j)
            dby = BSplineInterpolator._quintic_bspline_derivative(fy - j)

            for i in range(-2, 4):
                bx = BSplineInterpolator._quintic_bspline(fx - i)
                dbx = BSplineInterpolator._quintic_bspline_derivative(fx - i)

                coef = bcoef[iy + j, ix + i]
                value += coef * bx * by
                dx += coef * dbx * by
                dy += coef * bx * dby

        return value, dx, dy

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

        Args:
            coords: Nx2 array of (x, y) coordinates
            u_bcoef: B-spline coefficients for u displacement
            v_bcoef: B-spline coefficients for v displacement
            left_offset: Left bound offset
            top_offset: Top bound offset
            border: Border padding

        Returns:
            Tuple of (u, v) interpolated displacement arrays
        """
        u_interp = BSplineInterpolator.interpolate_values(
            coords, u_bcoef, left_offset, top_offset, border
        )
        v_interp = BSplineInterpolator.interpolate_values(
            coords, v_bcoef, left_offset, top_offset, border
        )

        return u_interp, v_interp


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

    Args:
        coords: Nx2 array of (x, y) coordinates
        bcoef: B-spline coefficient array
        left_offset: Left bound offset of data region
        top_offset: Top bound offset of data region
        border: Border padding in bcoef

    Returns:
        Interpolated values
    """
    return BSplineInterpolator.interpolate_values(
        coords, bcoef, left_offset, top_offset, border
    )
