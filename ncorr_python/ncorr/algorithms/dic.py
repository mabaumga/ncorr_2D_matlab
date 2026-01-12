"""
Digital Image Correlation (DIC) analysis algorithms.

Implements the Inverse Compositional Gauss-Newton DIC (IC-GN) algorithm
with first-order (affine) warp model.
Equivalent to ncorr_alg_rgdic.cpp and ncorr_alg_dicanalysis.m

All performance-critical functions are at module level with Numba acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from collections import deque
import heapq

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from tqdm import tqdm

from .bspline import (
    interpolate_subset,
    interpolate_subset_with_gradients,
)
from ..core.image import NcorrImage
from ..core.roi import NcorrROI
from ..core.dic_parameters import DICParameters
from ..core.status import Status


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class SeedInfo:
    """
    Information about a seed point for DIC analysis.

    Attributes:
        x: X-coordinate
        y: Y-coordinate
        u: Initial u displacement guess
        v: Initial v displacement guess
        region_idx: Index of region containing seed
        valid: Whether seed is valid
    """

    x: int
    y: int
    u: float = 0.0
    v: float = 0.0
    region_idx: int = 0
    valid: bool = True


@dataclass
class DICResult:
    """
    Results from DIC analysis.

    Attributes:
        u: U displacement field
        v: V displacement field
        corrcoef: Correlation coefficient field
        roi: ROI mask for valid points
        seed_info: Updated seed information for next image
        converged: Convergence status for each point
        iterations: Number of iterations used for each point
    """

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    corrcoef: NDArray[np.float64]
    roi: NDArray[np.bool_]
    seed_info: List[SeedInfo] = field(default_factory=list)
    converged: Optional[NDArray[np.bool_]] = None
    iterations: Optional[NDArray[np.int32]] = None

    def get_diagnostics(self) -> dict:
        """Get convergence diagnostics for the DIC result."""
        if self.converged is None:
            return {"error": "No convergence data available"}

        valid_mask = self.roi
        total_points = np.sum(valid_mask)

        if total_points == 0:
            return {"error": "No valid points"}

        converged_count = np.sum(self.converged[valid_mask])
        cc_valid = self.corrcoef[valid_mask]

        diagnostics = {
            "total_points": int(total_points),
            "converged_count": int(converged_count),
            "converged_percent": float(converged_count / total_points * 100),
            "cc_min": float(np.nanmin(cc_valid)),
            "cc_max": float(np.nanmax(cc_valid)),
            "cc_mean": float(np.nanmean(cc_valid)),
            "cc_median": float(np.nanmedian(cc_valid)),
            "cc_above_0.9": float(np.sum(cc_valid > 0.9) / len(cc_valid) * 100),
            "cc_above_0.95": float(np.sum(cc_valid > 0.95) / len(cc_valid) * 100),
        }

        if self.iterations is not None:
            iter_valid = self.iterations[valid_mask]
            diagnostics["iterations_mean"] = float(np.mean(iter_valid))
            diagnostics["iterations_max"] = int(np.max(iter_valid))

        return diagnostics

    def print_diagnostics(self) -> None:
        """Print convergence diagnostics."""
        diag = self.get_diagnostics()
        if "error" in diag:
            print(f"Diagnostics error: {diag['error']}")
            return

        print("=" * 50)
        print("DIC Convergence Diagnostics")
        print("=" * 50)
        print(f"Total points analyzed:    {diag['total_points']:,}")
        print(f"Converged:                {diag['converged_count']:,} ({diag['converged_percent']:.1f}%)")
        print(f"Correlation coefficient:")
        print(f"  Min:                    {diag['cc_min']:.4f}")
        print(f"  Max:                    {diag['cc_max']:.4f}")
        print(f"  Mean:                   {diag['cc_mean']:.4f}")
        print(f"  Median:                 {diag['cc_median']:.4f}")
        print(f"  Points with CC > 0.90:  {diag['cc_above_0.9']:.1f}%")
        print(f"  Points with CC > 0.95:  {diag['cc_above_0.95']:.1f}%")
        if "iterations_mean" in diag:
            print(f"Iterations:")
            print(f"  Mean:                   {diag['iterations_mean']:.1f}")
            print(f"  Max:                    {diag['iterations_max']}")
        print("=" * 50)

    def analyze_row_banding(self) -> dict:
        """
        Analyze horizontal banding artifacts in the v-displacement field.

        Returns:
            Dictionary with banding statistics
        """
        out_h, out_w = self.u.shape

        # Collect per-row statistics
        row_v_means = []
        row_u_means = []

        for iy in range(out_h):
            row_mask = self.roi[iy, :]
            if np.any(row_mask):
                row_v_means.append(np.mean(self.v[iy, row_mask]))
                row_u_means.append(np.mean(self.u[iy, row_mask]))

        if len(row_v_means) < 2:
            return {"error": "Not enough rows for banding analysis"}

        v_means = np.array(row_v_means)
        u_means = np.array(row_u_means)
        v_diffs = np.diff(v_means)
        u_diffs = np.diff(u_means)

        return {
            "v_row_std": float(np.std(v_means)),
            "v_row_range": float(np.max(v_means) - np.min(v_means)),
            "v_row_to_row_mean": float(np.mean(np.abs(v_diffs))),
            "v_row_to_row_max": float(np.max(np.abs(v_diffs))),
            "u_row_std": float(np.std(u_means)),
            "u_row_to_row_mean": float(np.mean(np.abs(u_diffs))),
            "banding_detected": float(np.std(v_means)) > 0.5,  # Threshold for banding
        }

    def smooth_row_banding(self, window_size: int = 5) -> 'DICResult':
        """
        Smooth horizontal banding artifacts in the v-displacement field.

        This applies a median filter along the y-direction to reduce row-to-row
        variations while preserving the overall displacement pattern.

        Args:
            window_size: Size of the median filter window (must be odd, default 5)

        Returns:
            New DICResult with smoothed v-field
        """
        from scipy.ndimage import median_filter

        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1

        # Create a copy of v
        v_smoothed = self.v.copy()

        # Apply 1D median filter along y-direction for each column
        out_h, out_w = self.v.shape

        for ix in range(out_w):
            col_mask = self.roi[:, ix]
            if np.sum(col_mask) >= window_size:
                # Get valid values in this column
                valid_indices = np.where(col_mask)[0]
                v_col = self.v[valid_indices, ix]

                # Apply median filter
                v_filtered = median_filter(v_col, size=window_size, mode='reflect')

                # Put back
                v_smoothed[valid_indices, ix] = v_filtered

        return DICResult(
            u=self.u.copy(),
            v=v_smoothed,
            corrcoef=self.corrcoef.copy(),
            roi=self.roi.copy(),
            seed_info=self.seed_info,
            converged=self.converged.copy() if self.converged is not None else None,
            iterations=self.iterations.copy() if self.iterations is not None else None,
        )

    def smooth_displacement_field(self, sigma: float = 1.0) -> 'DICResult':
        """
        Apply Gaussian smoothing to both u and v displacement fields.

        This can help reduce noise and banding artifacts while preserving
        the overall displacement pattern.

        Args:
            sigma: Standard deviation of Gaussian filter (default 1.0)

        Returns:
            New DICResult with smoothed displacement fields
        """
        from scipy.ndimage import gaussian_filter

        # Create copies
        u_smoothed = self.u.copy()
        v_smoothed = self.v.copy()

        # Replace NaN with 0 for filtering, then restore
        u_valid = np.where(self.roi, self.u, 0)
        v_valid = np.where(self.roi, self.v, 0)

        # Create weight mask for normalization
        weight = self.roi.astype(np.float64)

        # Apply Gaussian filter to values and weights
        u_filtered = gaussian_filter(u_valid, sigma=sigma)
        v_filtered = gaussian_filter(v_valid, sigma=sigma)
        weight_filtered = gaussian_filter(weight, sigma=sigma)

        # Normalize (avoid division by zero)
        mask = weight_filtered > 0.1
        u_smoothed = np.where(mask, u_filtered / weight_filtered, np.nan)
        v_smoothed = np.where(mask, v_filtered / weight_filtered, np.nan)

        # Restore original NaN positions
        u_smoothed = np.where(self.roi, u_smoothed, np.nan)
        v_smoothed = np.where(self.roi, v_smoothed, np.nan)

        return DICResult(
            u=u_smoothed,
            v=v_smoothed,
            corrcoef=self.corrcoef.copy(),
            roi=self.roi.copy(),
            seed_info=self.seed_info,
            converged=self.converged.copy() if self.converged is not None else None,
            iterations=self.iterations.copy() if self.iterations is not None else None,
        )

    def remove_rigid_body_motion(self, reference_point: Optional[Tuple[int, int]] = None) -> 'DICResult':
        """
        Remove rigid body motion (translation) from displacement fields.

        This subtracts the mean displacement (or displacement at a reference point)
        so that only relative displacements remain.

        Args:
            reference_point: Optional (x_grid, y_grid) tuple. If provided, subtract
                           displacement at this point. If None, subtract mean displacement.

        Returns:
            New DICResult with rigid body motion removed
        """
        u_corrected = self.u.copy()
        v_corrected = self.v.copy()

        if reference_point is not None:
            # Use displacement at reference point
            x_ref, y_ref = reference_point
            if self.roi[y_ref, x_ref]:
                u_offset = self.u[y_ref, x_ref]
                v_offset = self.v[y_ref, x_ref]
            else:
                # Reference point not valid, fall back to mean
                u_offset = np.nanmean(self.u[self.roi])
                v_offset = np.nanmean(self.v[self.roi])
        else:
            # Use mean displacement
            u_offset = np.nanmean(self.u[self.roi])
            v_offset = np.nanmean(self.v[self.roi])

        u_corrected = u_corrected - u_offset
        v_corrected = v_corrected - v_offset

        # Keep NaN where originally NaN
        u_corrected[~self.roi] = np.nan
        v_corrected[~self.roi] = np.nan

        return DICResult(
            u=u_corrected,
            v=v_corrected,
            corrcoef=self.corrcoef.copy(),
            roi=self.roi.copy(),
            seed_info=self.seed_info,
            converged=self.converged.copy() if self.converged is not None else None,
            iterations=self.iterations.copy() if self.iterations is not None else None,
        )

    def remove_linear_trend(self, remove_u_trend: bool = True, remove_v_trend: bool = True) -> 'DICResult':
        """
        Remove linear trend (plane fit) from displacement fields.

        This removes global bending/rotation effects by fitting and subtracting
        a linear plane: u = a*x + b*y + c

        After this, only local deviations from the linear trend remain,
        which is useful for detecting cracks and local strain concentrations.

        Args:
            remove_u_trend: Remove linear trend from u field (default True)
            remove_v_trend: Remove linear trend from v field (default True)

        Returns:
            New DICResult with linear trends removed
        """
        u_corrected = self.u.copy()
        v_corrected = self.v.copy()

        # Get valid points
        valid_y, valid_x = np.where(self.roi)

        if len(valid_x) < 10:
            # Not enough points for fitting
            return self

        # Fit and remove trend from u
        if remove_u_trend:
            u_valid = self.u[self.roi]

            # Build design matrix for plane fit: u = a*x + b*y + c
            A = np.column_stack([valid_x, valid_y, np.ones_like(valid_x)])

            # Least squares fit
            coeffs_u, _, _, _ = np.linalg.lstsq(A, u_valid, rcond=None)
            a_u, b_u, c_u = coeffs_u

            # Subtract fitted plane
            out_h, out_w = self.u.shape
            yy, xx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
            u_trend = a_u * xx + b_u * yy + c_u
            u_corrected = self.u - u_trend
            u_corrected[~self.roi] = np.nan

        # Fit and remove trend from v
        if remove_v_trend:
            v_valid = self.v[self.roi]

            A = np.column_stack([valid_x, valid_y, np.ones_like(valid_x)])
            coeffs_v, _, _, _ = np.linalg.lstsq(A, v_valid, rcond=None)
            a_v, b_v, c_v = coeffs_v

            out_h, out_w = self.v.shape
            yy, xx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
            v_trend = a_v * xx + b_v * yy + c_v
            v_corrected = self.v - v_trend
            v_corrected[~self.roi] = np.nan

        return DICResult(
            u=u_corrected,
            v=v_corrected,
            corrcoef=self.corrcoef.copy(),
            roi=self.roi.copy(),
            seed_info=self.seed_info,
            converged=self.converged.copy() if self.converged is not None else None,
            iterations=self.iterations.copy() if self.iterations is not None else None,
        )

    def get_relative_displacement(self, remove_translation: bool = True,
                                   remove_trend: bool = True) -> 'DICResult':
        """
        Get relative displacement field with rigid body motion and trends removed.

        This is a convenience method that combines:
        1. Remove rigid body translation (mean displacement)
        2. Remove linear trend (bending/rotation)

        The result shows only LOCAL deviations - useful for crack detection.

        Args:
            remove_translation: Remove mean displacement (default True)
            remove_trend: Remove linear trend (default True)

        Returns:
            New DICResult with corrections applied
        """
        result = self

        if remove_translation:
            result = result.remove_rigid_body_motion()

        if remove_trend:
            result = result.remove_linear_trend()

        return result


# =============================================================================
# Module-level Numba-accelerated functions for IC-GN optimization
# =============================================================================

@njit(cache=True, fastmath=True)
def _bspline_interp(px: float, py: float, bcoef: NDArray[np.float64]) -> float:
    """Inline B-spline interpolation for a single point (value only).

    Uses the CORRECT quintic B-spline basis functions that satisfy
    the partition of unity property (sum of basis functions = 1).
    """
    h, w = bcoef.shape
    if px < 2.0 or px >= w - 3.0 or py < 2.0 or py >= h - 3.0:
        return np.nan

    ix = int(px)
    iy = int(py)
    fx = px - ix
    fy = py - iy

    value = 0.0
    for jj in range(-2, 4):
        t_y = abs(fy - jj)
        if t_y >= 3.0:
            by = 0.0
        elif t_y >= 2.0:
            # (3-|t|)^5 / 120
            tmp = 3.0 - t_y
            by = tmp * tmp * tmp * tmp * tmp / 120.0
        elif t_y >= 1.0:
            # (5|t|^5 - 45|t|^4 + 150|t|^3 - 210|t|^2 + 75|t| + 51) / 120
            t2 = t_y * t_y
            t3 = t2 * t_y
            t4 = t3 * t_y
            t5 = t4 * t_y
            by = (5.0 * t5 - 45.0 * t4 + 150.0 * t3 - 210.0 * t2 + 75.0 * t_y + 51.0) / 120.0
        else:
            # (66 - 60*t^2 + 30*t^4 - 10*|t|^5) / 120
            t2 = t_y * t_y
            t4 = t2 * t2
            t5 = t4 * t_y
            by = (66.0 - 60.0 * t2 + 30.0 * t4 - 10.0 * t5) / 120.0

        for ii in range(-2, 4):
            t_x = abs(fx - ii)
            if t_x >= 3.0:
                bx = 0.0
            elif t_x >= 2.0:
                # (3-|t|)^5 / 120
                tmp = 3.0 - t_x
                bx = tmp * tmp * tmp * tmp * tmp / 120.0
            elif t_x >= 1.0:
                # (5|t|^5 - 45|t|^4 + 150|t|^3 - 210|t|^2 + 75|t| + 51) / 120
                t2 = t_x * t_x
                t3 = t2 * t_x
                t4 = t3 * t_x
                t5 = t4 * t_x
                bx = (5.0 * t5 - 45.0 * t4 + 150.0 * t3 - 210.0 * t2 + 75.0 * t_x + 51.0) / 120.0
            else:
                # (66 - 60*t^2 + 30*t^4 - 10*|t|^5) / 120
                t2 = t_x * t_x
                t4 = t2 * t2
                t5 = t4 * t_x
                bx = (66.0 - 60.0 * t2 + 30.0 * t4 - 10.0 * t5) / 120.0

            value += bcoef[iy + jj, ix + ii] * bx * by

    return value


@njit(cache=True, fastmath=True)
def _bspline_interp_with_grad(
    px: float, py: float, bcoef: NDArray[np.float64]
) -> Tuple[float, float, float]:
    """Inline B-spline interpolation with gradients.

    Uses the CORRECT quintic B-spline basis functions and derivatives
    that satisfy the partition of unity property.
    """
    h, w = bcoef.shape
    if px < 2.0 or px >= w - 3.0 or py < 2.0 or py >= h - 3.0:
        return np.nan, np.nan, np.nan

    ix = int(px)
    iy = int(py)
    fx = px - ix
    fy = py - iy

    value = 0.0
    grad_x = 0.0
    grad_y = 0.0

    for jj in range(-2, 4):
        t_y = abs(fy - jj)
        if t_y >= 3.0:
            by = 0.0
            dby = 0.0
        elif t_y >= 2.0:
            # B(t) = (3-|t|)^5 / 120
            # B'(t) = -sign(t) * (3-|t|)^4 / 24
            tmp = 3.0 - t_y
            by = tmp * tmp * tmp * tmp * tmp / 120.0
            dby = -tmp * tmp * tmp * tmp / 24.0
        elif t_y >= 1.0:
            # B(t) = (5|t|^5 - 45|t|^4 + 150|t|^3 - 210|t|^2 + 75|t| + 51) / 120
            # B'(t) = sign(t) * (25|t|^4 - 180|t|^3 + 450|t|^2 - 420|t| + 75) / 120
            t2 = t_y * t_y
            t3 = t2 * t_y
            t4 = t3 * t_y
            t5 = t4 * t_y
            by = (5.0 * t5 - 45.0 * t4 + 150.0 * t3 - 210.0 * t2 + 75.0 * t_y + 51.0) / 120.0
            dby = (25.0 * t4 - 180.0 * t3 + 450.0 * t2 - 420.0 * t_y + 75.0) / 120.0
        else:
            # B(t) = (66 - 60*t^2 + 30*t^4 - 10*|t|^5) / 120
            # B'(t) = (-120*|t| + 120*|t|^3 - 50*|t|^4) / 120 (before sign adjustment)
            t2 = t_y * t_y
            t3 = t2 * t_y
            t4 = t2 * t2
            t5 = t4 * t_y
            by = (66.0 - 60.0 * t2 + 30.0 * t4 - 10.0 * t5) / 120.0
            dby = (-120.0 * t_y + 120.0 * t3 - 50.0 * t4) / 120.0

        if (fy - jj) < 0:
            dby = -dby

        for ii in range(-2, 4):
            t_x = abs(fx - ii)
            if t_x >= 3.0:
                bx = 0.0
                dbx = 0.0
            elif t_x >= 2.0:
                # B(t) = (3-|t|)^5 / 120
                # B'(t) = -sign(t) * (3-|t|)^4 / 24
                tmp = 3.0 - t_x
                bx = tmp * tmp * tmp * tmp * tmp / 120.0
                dbx = -tmp * tmp * tmp * tmp / 24.0
            elif t_x >= 1.0:
                # B(t) = (5|t|^5 - 45|t|^4 + 150|t|^3 - 210|t|^2 + 75|t| + 51) / 120
                # B'(t) = sign(t) * (25|t|^4 - 180|t|^3 + 450|t|^2 - 420|t| + 75) / 120
                t2 = t_x * t_x
                t3 = t2 * t_x
                t4 = t3 * t_x
                t5 = t4 * t_x
                bx = (5.0 * t5 - 45.0 * t4 + 150.0 * t3 - 210.0 * t2 + 75.0 * t_x + 51.0) / 120.0
                dbx = (25.0 * t4 - 180.0 * t3 + 450.0 * t2 - 420.0 * t_x + 75.0) / 120.0
            else:
                # B(t) = (66 - 60*t^2 + 30*t^4 - 10*|t|^5) / 120
                # B'(t) = (-120*|t| + 120*|t|^3 - 50*|t|^4) / 120 (before sign adjustment)
                t2 = t_x * t_x
                t3 = t2 * t_x
                t4 = t2 * t2
                t5 = t4 * t_x
                bx = (66.0 - 60.0 * t2 + 30.0 * t4 - 10.0 * t5) / 120.0
                dbx = (-120.0 * t_x + 120.0 * t3 - 50.0 * t4) / 120.0

            if (fx - ii) < 0:
                dbx = -dbx

            coef = bcoef[iy + jj, ix + ii]
            value += coef * bx * by
            grad_x += coef * dbx * by
            grad_y += coef * bx * dby

    return value, grad_x, grad_y


@njit(cache=True, fastmath=True)
def _masked_mean(arr: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    """Compute mean of masked array."""
    total = 0.0
    count = 0
    h, w = mask.shape

    for j in range(h):
        for i in range(w):
            if mask[j, i]:
                total += arr[j, i]
                count += 1

    return total / count if count > 0 else 0.0


@njit(cache=True, fastmath=True)
def _masked_norm(arr: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    """Compute norm of masked array (sqrt of sum of squares)."""
    total = 0.0
    h, w = mask.shape

    for j in range(h):
        for i in range(w):
            if mask[j, i]:
                total += arr[j, i] * arr[j, i]

    return np.sqrt(total)


@njit(cache=True, fastmath=True)
def _compute_ncc(
    ref_centered: NDArray[np.float64],
    cur_centered: NDArray[np.float64],
    ref_norm: float,
    cur_norm: float,
    mask: NDArray[np.bool_],
) -> float:
    """Compute Normalized Cross-Correlation coefficient."""
    h, w = mask.shape
    dot_product = 0.0

    for j in range(h):
        for i in range(w):
            if mask[j, i]:
                dot_product += ref_centered[j, i] * cur_centered[j, i]

    return dot_product / (ref_norm * cur_norm)


@njit(cache=True, fastmath=True)
def _local_ncc_search(
    ref_bcoef: NDArray[np.float64],
    cur_bcoef: NDArray[np.float64],
    border: int,
    x: int,
    y: int,
    radius: int,
    u_init: float,
    v_init: float,
    search_radius: int = 3,
) -> Tuple[float, float, float]:
    """
    Perform local NCC search around initial guess to find best integer displacement.

    This prevents convergence to wrong local minima by ensuring we start in the
    correct "basin of attraction" for the IC-GN optimization.

    Args:
        ref_bcoef: Reference B-spline coefficients
        cur_bcoef: Current B-spline coefficients
        border: Border padding
        x, y: Subset center coordinates
        radius: Subset radius
        u_init, v_init: Initial displacement guess
        search_radius: Search range around initial guess (default ±3 pixels)

    Returns:
        Tuple of (best_u, best_v, best_ncc)
    """
    size = 2 * radius + 1
    h_ref, w_ref = ref_bcoef.shape
    h_cur, w_cur = cur_bcoef.shape

    # Extract reference subset
    ref_subset = np.empty((size, size), dtype=np.float64)
    mask = np.ones((size, size), dtype=np.bool_)
    valid_count = 0

    for j in range(size):
        for i in range(size):
            dx = float(i - radius)
            dy = float(j - radius)
            px = float(x) + dx + border
            py = float(y) + dy + border

            if px < 2.0 or px >= w_ref - 3.0 or py < 2.0 or py >= h_ref - 3.0:
                mask[j, i] = False
                ref_subset[j, i] = 0.0
                continue

            val = _bspline_interp(px, py, ref_bcoef)
            if np.isnan(val):
                mask[j, i] = False
                ref_subset[j, i] = 0.0
                continue

            ref_subset[j, i] = val
            valid_count += 1

    if valid_count < 20:
        return u_init, v_init, 0.0

    # Reference statistics
    ref_mean = _masked_mean(ref_subset, mask)
    ref_centered = ref_subset - ref_mean
    ref_norm = _masked_norm(ref_centered, mask)

    if ref_norm < 1e-10:
        return u_init, v_init, 0.0

    # Search around initial guess
    best_u = u_init
    best_v = v_init
    best_ncc = -1.0

    u_center = int(round(u_init))
    v_center = int(round(v_init))

    cur_subset = np.empty((size, size), dtype=np.float64)

    for du in range(-search_radius, search_radius + 1):
        for dv in range(-search_radius, search_radius + 1):
            u_test = float(u_center + du)
            v_test = float(v_center + dv)

            # Extract current subset at this displacement
            all_valid = True
            for j in range(size):
                for i in range(size):
                    if not mask[j, i]:
                        cur_subset[j, i] = 0.0
                        continue

                    dx = float(i - radius)
                    dy = float(j - radius)
                    wx = float(x) + dx + u_test + border
                    wy = float(y) + dy + v_test + border

                    if wx < 2.0 or wx >= w_cur - 3.0 or wy < 2.0 or wy >= h_cur - 3.0:
                        all_valid = False
                        break

                    val = _bspline_interp(wx, wy, cur_bcoef)
                    if np.isnan(val):
                        all_valid = False
                        break

                    cur_subset[j, i] = val

                if not all_valid:
                    break

            if not all_valid:
                continue

            # Compute NCC
            cur_mean = _masked_mean(cur_subset, mask)
            cur_centered = cur_subset - cur_mean
            cur_norm = _masked_norm(cur_centered, mask)

            if cur_norm < 1e-10:
                continue

            ncc = _compute_ncc(ref_centered, cur_centered, ref_norm, cur_norm, mask)

            if ncc > best_ncc:
                best_ncc = ncc
                best_u = u_test
                best_v = v_test

    return best_u, best_v, best_ncc


@njit(cache=True, fastmath=True)
def _solve_2x2(H: NDArray[np.float64], b: NDArray[np.float64]) -> Tuple[NDArray[np.float64], bool]:
    """Solve 2x2 linear system Hx = b."""
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]

    if abs(det) < 1e-12:
        return np.zeros(2, dtype=np.float64), False

    x = np.empty(2, dtype=np.float64)
    x[0] = (H[1, 1] * b[0] - H[0, 1] * b[1]) / det
    x[1] = (H[0, 0] * b[1] - H[1, 0] * b[0]) / det

    return x, True


@njit(cache=True, fastmath=True)
def _solve_6x6(H: NDArray[np.float64], b: NDArray[np.float64]) -> Tuple[NDArray[np.float64], bool]:
    """Solve 6x6 linear system Hx = b using Gaussian elimination with partial pivoting."""
    n = 6
    # Create augmented matrix
    A = np.empty((n, n + 1), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = H[i, j]
        A[i, n] = b[i]

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_val = abs(A[k, k])
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_row = i

        if max_val < 1e-12:
            return np.zeros(n, dtype=np.float64), False

        # Swap rows
        if max_row != k:
            for j in range(k, n + 1):
                tmp = A[k, j]
                A[k, j] = A[max_row, j]
                A[max_row, j] = tmp

        # Eliminate
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k, n + 1):
                A[i, j] -= factor * A[k, j]

    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = A[i, n]
        for j in range(i + 1, n):
            x[i] -= A[i, j] * x[j]
        if abs(A[i, i]) < 1e-12:
            return np.zeros(n, dtype=np.float64), False
        x[i] /= A[i, i]

    return x, True


@njit(cache=True, fastmath=True)
def _ic_gn_translation(
    ref_bcoef: NDArray[np.float64],
    cur_bcoef: NDArray[np.float64],
    border: int,
    x: int,
    y: int,
    radius: int,
    u_init: float,
    v_init: float,
    cutoff_diffnorm: float,
    cutoff_iteration: int,
) -> Tuple[float, float, float, bool, int]:
    """
    IC-GN optimization with translation-only (2-parameter) warp model.

    This is more robust for large displacements as it has fewer parameters.

    Args:
        ref_bcoef: Reference image B-spline coefficients
        cur_bcoef: Current image B-spline coefficients
        border: Border padding
        x, y: Subset center coordinates (in original image)
        radius: Subset radius
        u_init, v_init: Initial displacement guess
        cutoff_diffnorm: Convergence threshold
        cutoff_iteration: Maximum iterations

    Returns:
        Tuple of (u, v, correlation_coefficient, converged, iterations)
    """
    size = 2 * radius + 1
    h_ref, w_ref = ref_bcoef.shape
    h_cur, w_cur = cur_bcoef.shape

    # Arrays for reference subset
    ref_subset = np.empty((size, size), dtype=np.float64)
    ref_dx = np.empty((size, size), dtype=np.float64)
    ref_dy = np.empty((size, size), dtype=np.float64)
    mask = np.ones((size, size), dtype=np.bool_)

    valid_count = 0

    # Get reference subset with gradients
    for j in range(size):
        for i in range(size):
            dx = float(i - radius)
            dy = float(j - radius)
            px = float(x) + dx + border
            py = float(y) + dy + border

            if px < 2.0 or px >= w_ref - 3.0 or py < 2.0 or py >= h_ref - 3.0:
                mask[j, i] = False
                ref_subset[j, i] = 0.0
                ref_dx[j, i] = 0.0
                ref_dy[j, i] = 0.0
                continue

            val, gx, gy = _bspline_interp_with_grad(px, py, ref_bcoef)

            if np.isnan(val):
                mask[j, i] = False
                ref_subset[j, i] = 0.0
                ref_dx[j, i] = 0.0
                ref_dy[j, i] = 0.0
                continue

            ref_subset[j, i] = val
            ref_dx[j, i] = gx
            ref_dy[j, i] = gy
            valid_count += 1

    # Need minimum valid pixels
    if valid_count < 20:
        return np.nan, np.nan, np.nan, False, 0

    # Reference subset statistics
    ref_mean = _masked_mean(ref_subset, mask)
    ref_centered = ref_subset - ref_mean
    ref_norm = _masked_norm(ref_centered, mask)

    if ref_norm < 1e-10:
        return np.nan, np.nan, np.nan, False, 0

    # Normalize gradients
    grad_x_norm = ref_dx / ref_norm
    grad_y_norm = ref_dy / ref_norm

    # Precompute Hessian matrix (2x2 for translation)
    H = np.zeros((2, 2), dtype=np.float64)

    for j in range(size):
        for i in range(size):
            if not mask[j, i]:
                continue

            gx = grad_x_norm[j, i]
            gy = grad_y_norm[j, i]

            H[0, 0] += gx * gx
            H[0, 1] += gx * gy
            H[1, 0] += gy * gx
            H[1, 1] += gy * gy

    # Reference normalized
    ref_normalized = ref_centered / ref_norm

    # Initialize parameters
    u, v = u_init, v_init
    converged = False
    final_iteration = 0

    for iteration in range(cutoff_iteration):
        final_iteration = iteration + 1

        # Get current subset at warped location
        cur_subset = np.empty((size, size), dtype=np.float64)
        all_valid = True

        for j in range(size):
            for i in range(size):
                if not mask[j, i]:
                    cur_subset[j, i] = 0.0
                    continue

                dx = float(i - radius)
                dy = float(j - radius)

                # Translation-only warp
                wx = float(x) + dx + u + border
                wy = float(y) + dy + v + border

                if wx < 2.0 or wx >= w_cur - 3.0 or wy < 2.0 or wy >= h_cur - 3.0:
                    all_valid = False
                    break

                val = _bspline_interp(wx, wy, cur_bcoef)

                if np.isnan(val):
                    all_valid = False
                    break

                cur_subset[j, i] = val

            if not all_valid:
                break

        if not all_valid:
            break

        # Current subset statistics
        cur_mean = _masked_mean(cur_subset, mask)
        cur_centered = cur_subset - cur_mean
        cur_norm = _masked_norm(cur_centered, mask)

        if cur_norm < 1e-10:
            break

        cur_normalized = cur_centered / cur_norm

        # Error image
        error = ref_normalized - cur_normalized

        # Compute gradient
        b = np.zeros(2, dtype=np.float64)
        for j in range(size):
            for i in range(size):
                if not mask[j, i]:
                    continue

                b[0] += grad_x_norm[j, i] * error[j, i]
                b[1] += grad_y_norm[j, i] * error[j, i]

        # Solve for update
        dp, success = _solve_2x2(H, b)
        if not success:
            break

        # Update parameters
        u += dp[0]
        v += dp[1]

        # Check convergence
        diffnorm = np.sqrt(dp[0] * dp[0] + dp[1] * dp[1])
        if diffnorm < cutoff_diffnorm:
            converged = True
            break

    # Compute final correlation coefficient
    cur_subset = np.empty((size, size), dtype=np.float64)
    all_valid = True

    for j in range(size):
        for i in range(size):
            if not mask[j, i]:
                cur_subset[j, i] = 0.0
                continue

            dx = float(i - radius)
            dy = float(j - radius)

            wx = float(x) + dx + u + border
            wy = float(y) + dy + v + border

            if wx < 2.0 or wx >= w_cur - 3.0 or wy < 2.0 or wy >= h_cur - 3.0:
                all_valid = False
                break

            val = _bspline_interp(wx, wy, cur_bcoef)

            if np.isnan(val):
                all_valid = False
                break

            cur_subset[j, i] = val

        if not all_valid:
            break

    if not all_valid:
        return np.nan, np.nan, np.nan, False, final_iteration

    cur_mean = _masked_mean(cur_subset, mask)
    cur_centered = cur_subset - cur_mean
    cur_norm = _masked_norm(cur_centered, mask)

    if cur_norm < 1e-10:
        return np.nan, np.nan, np.nan, False, final_iteration

    ncc = _compute_ncc(ref_centered, cur_centered, ref_norm, cur_norm, mask)

    return u, v, ncc, converged, final_iteration


@njit(cache=True, fastmath=True)
def _ic_gn_affine(
    ref_bcoef: NDArray[np.float64],
    cur_bcoef: NDArray[np.float64],
    border: int,
    x: int,
    y: int,
    radius: int,
    u_init: float,
    v_init: float,
    dudx_init: float,
    dudy_init: float,
    dvdx_init: float,
    dvdy_init: float,
    cutoff_diffnorm: float,
    cutoff_iteration: int,
) -> Tuple[float, float, float, float, float, float, float, bool, int]:
    """
    IC-GN optimization with first-order affine (6-parameter) warp model.

    This model captures both displacement and strain (displacement gradients).

    Warp model:
        wx = x + dx + u + dudx*dx + dudy*dy
        wy = y + dy + v + dvdx*dx + dvdy*dy

    Parameters: p = [u, dudx, dudy, v, dvdx, dvdy]

    Args:
        ref_bcoef: Reference image B-spline coefficients
        cur_bcoef: Current image B-spline coefficients
        border: Border padding
        x, y: Subset center coordinates (in original image)
        radius: Subset radius
        u_init, v_init: Initial displacement guess
        dudx_init, dudy_init, dvdx_init, dvdy_init: Initial strain guess
        cutoff_diffnorm: Convergence threshold
        cutoff_iteration: Maximum iterations

    Returns:
        Tuple of (u, v, dudx, dudy, dvdx, dvdy, correlation_coefficient, converged, iterations)
    """
    size = 2 * radius + 1
    h_ref, w_ref = ref_bcoef.shape
    h_cur, w_cur = cur_bcoef.shape

    # Arrays for reference subset
    ref_subset = np.empty((size, size), dtype=np.float64)
    ref_dx = np.empty((size, size), dtype=np.float64)
    ref_dy = np.empty((size, size), dtype=np.float64)
    # Store local coordinates for Hessian computation
    dx_arr = np.empty((size, size), dtype=np.float64)
    dy_arr = np.empty((size, size), dtype=np.float64)
    mask = np.ones((size, size), dtype=np.bool_)

    valid_count = 0

    # Get reference subset with gradients
    for j in range(size):
        for i in range(size):
            dx = float(i - radius)
            dy = float(j - radius)
            dx_arr[j, i] = dx
            dy_arr[j, i] = dy

            px = float(x) + dx + border
            py = float(y) + dy + border

            if px < 2.0 or px >= w_ref - 3.0 or py < 2.0 or py >= h_ref - 3.0:
                mask[j, i] = False
                ref_subset[j, i] = 0.0
                ref_dx[j, i] = 0.0
                ref_dy[j, i] = 0.0
                continue

            val, gx, gy = _bspline_interp_with_grad(px, py, ref_bcoef)

            if np.isnan(val):
                mask[j, i] = False
                ref_subset[j, i] = 0.0
                ref_dx[j, i] = 0.0
                ref_dy[j, i] = 0.0
                continue

            ref_subset[j, i] = val
            ref_dx[j, i] = gx
            ref_dy[j, i] = gy
            valid_count += 1

    # Need minimum valid pixels
    if valid_count < 20:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, 0

    # Reference subset statistics
    ref_mean = _masked_mean(ref_subset, mask)
    ref_centered = ref_subset - ref_mean
    ref_norm = _masked_norm(ref_centered, mask)

    if ref_norm < 1e-10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, 0

    # Normalize gradients
    grad_x_norm = ref_dx / ref_norm
    grad_y_norm = ref_dy / ref_norm

    # Precompute Hessian matrix (6x6 for affine)
    # Parameters: [u, dudx, dudy, v, dvdx, dvdy]
    # Steepest descent images:
    # SD[0] = grad_x * 1 = grad_x (for u)
    # SD[1] = grad_x * dx (for dudx)
    # SD[2] = grad_x * dy (for dudy)
    # SD[3] = grad_y * 1 = grad_y (for v)
    # SD[4] = grad_y * dx (for dvdx)
    # SD[5] = grad_y * dy (for dvdy)

    H = np.zeros((6, 6), dtype=np.float64)

    for j in range(size):
        for i in range(size):
            if not mask[j, i]:
                continue

            gx = grad_x_norm[j, i]
            gy = grad_y_norm[j, i]
            dx = dx_arr[j, i]
            dy = dy_arr[j, i]

            # Steepest descent images for this pixel
            sd = np.empty(6, dtype=np.float64)
            sd[0] = gx          # u
            sd[1] = gx * dx     # dudx
            sd[2] = gx * dy     # dudy
            sd[3] = gy          # v
            sd[4] = gy * dx     # dvdx
            sd[5] = gy * dy     # dvdy

            # Add outer product to Hessian
            for k in range(6):
                for l in range(6):
                    H[k, l] += sd[k] * sd[l]

    # Reference normalized
    ref_normalized = ref_centered / ref_norm

    # Initialize parameters
    u = u_init
    dudx = dudx_init
    dudy = dudy_init
    v = v_init
    dvdx = dvdx_init
    dvdy = dvdy_init

    converged = False
    final_iteration = 0

    for iteration in range(cutoff_iteration):
        final_iteration = iteration + 1

        # Get current subset at warped location
        cur_subset = np.empty((size, size), dtype=np.float64)
        all_valid = True

        for j in range(size):
            for i in range(size):
                if not mask[j, i]:
                    cur_subset[j, i] = 0.0
                    continue

                dx = dx_arr[j, i]
                dy = dy_arr[j, i]

                # First-order affine warp
                wx = float(x) + dx + u + dudx * dx + dudy * dy + border
                wy = float(y) + dy + v + dvdx * dx + dvdy * dy + border

                if wx < 2.0 or wx >= w_cur - 3.0 or wy < 2.0 or wy >= h_cur - 3.0:
                    all_valid = False
                    break

                val = _bspline_interp(wx, wy, cur_bcoef)

                if np.isnan(val):
                    all_valid = False
                    break

                cur_subset[j, i] = val

            if not all_valid:
                break

        if not all_valid:
            break

        # Current subset statistics
        cur_mean = _masked_mean(cur_subset, mask)
        cur_centered = cur_subset - cur_mean
        cur_norm = _masked_norm(cur_centered, mask)

        if cur_norm < 1e-10:
            break

        cur_normalized = cur_centered / cur_norm

        # Error image
        error = ref_normalized - cur_normalized

        # Compute gradient vector
        b = np.zeros(6, dtype=np.float64)
        for j in range(size):
            for i in range(size):
                if not mask[j, i]:
                    continue

                gx = grad_x_norm[j, i]
                gy = grad_y_norm[j, i]
                dx = dx_arr[j, i]
                dy = dy_arr[j, i]
                err = error[j, i]

                b[0] += gx * err           # u
                b[1] += gx * dx * err      # dudx
                b[2] += gx * dy * err      # dudy
                b[3] += gy * err           # v
                b[4] += gy * dx * err      # dvdx
                b[5] += gy * dy * err      # dvdy

        # Solve for update
        dp, success = _solve_6x6(H, b)
        if not success:
            break

        # Inverse compositional update
        # For IC, we need to invert the warp update and compose
        # For small updates, this simplifies to additive update
        u += dp[0]
        dudx += dp[1]
        dudy += dp[2]
        v += dp[3]
        dvdx += dp[4]
        dvdy += dp[5]

        # Check convergence (use displacement components primarily)
        diffnorm = np.sqrt(dp[0] * dp[0] + dp[3] * dp[3])
        if diffnorm < cutoff_diffnorm:
            converged = True
            break

    # Compute final correlation coefficient
    cur_subset = np.empty((size, size), dtype=np.float64)
    all_valid = True

    for j in range(size):
        for i in range(size):
            if not mask[j, i]:
                cur_subset[j, i] = 0.0
                continue

            dx = dx_arr[j, i]
            dy = dy_arr[j, i]

            wx = float(x) + dx + u + dudx * dx + dudy * dy + border
            wy = float(y) + dy + v + dvdx * dx + dvdy * dy + border

            if wx < 2.0 or wx >= w_cur - 3.0 or wy < 2.0 or wy >= h_cur - 3.0:
                all_valid = False
                break

            val = _bspline_interp(wx, wy, cur_bcoef)

            if np.isnan(val):
                all_valid = False
                break

            cur_subset[j, i] = val

        if not all_valid:
            break

    if not all_valid:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, final_iteration

    cur_mean = _masked_mean(cur_subset, mask)
    cur_centered = cur_subset - cur_mean
    cur_norm = _masked_norm(cur_centered, mask)

    if cur_norm < 1e-10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, final_iteration

    ncc = _compute_ncc(ref_centered, cur_centered, ref_norm, cur_norm, mask)

    return u, v, dudx, dudy, dvdx, dvdy, ncc, converged, final_iteration


# Alias for backwards compatibility
_ic_gn_first_order = _ic_gn_affine
_ic_gn_single_point = _ic_gn_translation


@njit(cache=True, fastmath=True)
def _check_point_in_region(
    x: int,
    y: int,
    leftbound: int,
    noderange: NDArray[np.int32],
    nodelist: NDArray[np.int32],
) -> bool:
    """Check if a point is inside a region defined by nodelist/noderange."""
    idx = x - leftbound

    if idx < 0 or idx >= len(noderange):
        return False

    n_ranges = noderange[idx]
    for k in range(0, n_ranges, 2):
        if nodelist[idx, k] <= y <= nodelist[idx, k + 1]:
            return True

    return False


@njit(cache=True, parallel=True)
def _refine_initial_guesses_parallel(
    ref_bcoef: NDArray[np.float64],
    cur_bcoef: NDArray[np.float64],
    border: int,
    points_x: NDArray[np.int32],
    points_y: NDArray[np.int32],
    u_init: NDArray[np.float64],
    v_init: NDArray[np.float64],
    radius: int,
    search_radius: int = 3,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Refine initial displacement guesses using local NCC search.

    This ensures each point starts IC-GN from the correct integer displacement,
    preventing convergence to wrong local minima and eliminating banding artifacts.
    """
    n_points = len(points_x)
    u_refined = np.empty(n_points, dtype=np.float64)
    v_refined = np.empty(n_points, dtype=np.float64)

    for idx in prange(n_points):
        u, v, _ = _local_ncc_search(
            ref_bcoef, cur_bcoef, border,
            points_x[idx], points_y[idx], radius,
            u_init[idx], v_init[idx],
            search_radius,
        )
        u_refined[idx] = u
        v_refined[idx] = v

    return u_refined, v_refined


@njit(cache=True, parallel=True)
def _process_points_parallel_translation(
    ref_bcoef: NDArray[np.float64],
    cur_bcoef: NDArray[np.float64],
    border: int,
    points_x: NDArray[np.int32],
    points_y: NDArray[np.int32],
    u_init: NDArray[np.float64],
    v_init: NDArray[np.float64],
    radius: int,
    cutoff_diffnorm: float,
    cutoff_iteration: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.bool_], NDArray[np.int32]]:
    """
    Process multiple points in parallel using translation-only warp model.

    This is more robust for large displacements with small strain.

    Args:
        ref_bcoef: Reference B-spline coefficients
        cur_bcoef: Current B-spline coefficients
        border: Border size
        points_x, points_y: Point coordinates
        u_init, v_init: Initial displacement guesses
        radius: Subset radius
        cutoff_diffnorm: Convergence threshold
        cutoff_iteration: Maximum iterations

    Returns:
        Tuple of (u, v, corrcoef, converged, iterations) arrays
    """
    n_points = len(points_x)

    u_out = np.empty(n_points, dtype=np.float64)
    v_out = np.empty(n_points, dtype=np.float64)
    cc_out = np.empty(n_points, dtype=np.float64)
    conv_out = np.empty(n_points, dtype=np.bool_)
    iter_out = np.empty(n_points, dtype=np.int32)

    for idx in prange(n_points):
        u, v, cc, conv, n_iter = _ic_gn_translation(
            ref_bcoef, cur_bcoef, border,
            points_x[idx], points_y[idx], radius,
            u_init[idx], v_init[idx],
            cutoff_diffnorm, cutoff_iteration,
        )

        u_out[idx] = u
        v_out[idx] = v
        cc_out[idx] = cc
        conv_out[idx] = conv
        iter_out[idx] = n_iter

    return u_out, v_out, cc_out, conv_out, iter_out


@njit(cache=True, parallel=True)
def _process_points_parallel(
    ref_bcoef: NDArray[np.float64],
    cur_bcoef: NDArray[np.float64],
    border: int,
    points_x: NDArray[np.int32],
    points_y: NDArray[np.int32],
    u_init: NDArray[np.float64],
    v_init: NDArray[np.float64],
    dudx_init: NDArray[np.float64],
    dudy_init: NDArray[np.float64],
    dvdx_init: NDArray[np.float64],
    dvdy_init: NDArray[np.float64],
    radius: int,
    cutoff_diffnorm: float,
    cutoff_iteration: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.bool_], NDArray[np.int32]]:
    """
    Process multiple points in parallel using affine warp model.

    Args:
        ref_bcoef: Reference B-spline coefficients
        cur_bcoef: Current B-spline coefficients
        border: Border size
        points_x, points_y: Point coordinates
        u_init, v_init: Initial displacement guesses
        dudx_init, dudy_init, dvdx_init, dvdy_init: Initial strain guesses
        radius: Subset radius
        cutoff_diffnorm: Convergence threshold
        cutoff_iteration: Maximum iterations

    Returns:
        Tuple of (u, v, dudx, dudy, dvdx, dvdy, corrcoef, converged, iterations) arrays
    """
    n_points = len(points_x)

    u_out = np.empty(n_points, dtype=np.float64)
    v_out = np.empty(n_points, dtype=np.float64)
    dudx_out = np.empty(n_points, dtype=np.float64)
    dudy_out = np.empty(n_points, dtype=np.float64)
    dvdx_out = np.empty(n_points, dtype=np.float64)
    dvdy_out = np.empty(n_points, dtype=np.float64)
    cc_out = np.empty(n_points, dtype=np.float64)
    conv_out = np.empty(n_points, dtype=np.bool_)
    iter_out = np.empty(n_points, dtype=np.int32)

    for idx in prange(n_points):
        u, v, dudx, dudy, dvdx, dvdy, cc, conv, n_iter = _ic_gn_affine(
            ref_bcoef, cur_bcoef, border,
            points_x[idx], points_y[idx], radius,
            u_init[idx], v_init[idx],
            dudx_init[idx], dudy_init[idx],
            dvdx_init[idx], dvdy_init[idx],
            cutoff_diffnorm, cutoff_iteration,
        )

        u_out[idx] = u
        v_out[idx] = v
        dudx_out[idx] = dudx
        dudy_out[idx] = dudy
        dvdx_out[idx] = dvdx
        dvdy_out[idx] = dvdy
        cc_out[idx] = cc
        conv_out[idx] = conv
        iter_out[idx] = n_iter

    return u_out, v_out, dudx_out, dudy_out, dvdx_out, dvdy_out, cc_out, conv_out, iter_out


@njit(cache=True, parallel=True)
def _process_batch_with_ncc_search(
    ref_bcoef: NDArray[np.float64],
    cur_bcoef: NDArray[np.float64],
    border: int,
    points_x: NDArray[np.int32],
    points_y: NDArray[np.int32],
    u_init: NDArray[np.float64],
    v_init: NDArray[np.float64],
    dudx_init: NDArray[np.float64],
    dudy_init: NDArray[np.float64],
    dvdx_init: NDArray[np.float64],
    dvdy_init: NDArray[np.float64],
    radius: int,
    cutoff_diffnorm: float,
    cutoff_iteration: int,
    ncc_search_radius: int = 2,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.bool_], NDArray[np.int32]]:
    """
    Process a batch of points in parallel: NCC search + IC-GN.

    For each point:
    1. Refine initial guess with local NCC search
    2. Run IC-GN optimization with affine warp

    Args:
        ref_bcoef, cur_bcoef: B-spline coefficients
        border: Border size
        points_x, points_y: Point coordinates
        u_init, v_init: Initial displacement guesses
        dudx_init, dudy_init, dvdx_init, dvdy_init: Initial strain guesses
        radius: Subset radius
        cutoff_diffnorm, cutoff_iteration: IC-GN convergence parameters
        ncc_search_radius: Local NCC search radius (default 2)

    Returns:
        Tuple of (u, v, dudx, dudy, dvdx, dvdy, corrcoef, converged, iterations)
    """
    n_points = len(points_x)

    u_out = np.empty(n_points, dtype=np.float64)
    v_out = np.empty(n_points, dtype=np.float64)
    dudx_out = np.empty(n_points, dtype=np.float64)
    dudy_out = np.empty(n_points, dtype=np.float64)
    dvdx_out = np.empty(n_points, dtype=np.float64)
    dvdy_out = np.empty(n_points, dtype=np.float64)
    cc_out = np.empty(n_points, dtype=np.float64)
    conv_out = np.empty(n_points, dtype=np.bool_)
    iter_out = np.empty(n_points, dtype=np.int32)

    for idx in prange(n_points):
        # Step 1: Refine initial guess with NCC search
        u_ncc, v_ncc, ncc_val = _local_ncc_search(
            ref_bcoef, cur_bcoef, border,
            points_x[idx], points_y[idx], radius,
            u_init[idx], v_init[idx],
            ncc_search_radius,
        )

        # Use refined guess if NCC search succeeded
        if ncc_val > 0.5:
            u_start = u_ncc
            v_start = v_ncc
        else:
            u_start = u_init[idx]
            v_start = v_init[idx]

        # Step 2: Run IC-GN with affine warp
        u, v, dudx, dudy, dvdx, dvdy, cc, conv, n_iter = _ic_gn_affine(
            ref_bcoef, cur_bcoef, border,
            points_x[idx], points_y[idx], radius,
            u_start, v_start,
            dudx_init[idx], dudy_init[idx],
            dvdx_init[idx], dvdy_init[idx],
            cutoff_diffnorm, cutoff_iteration,
        )

        u_out[idx] = u
        v_out[idx] = v
        dudx_out[idx] = dudx
        dudy_out[idx] = dudy
        dvdx_out[idx] = dvdx
        dvdy_out[idx] = dvdy
        cc_out[idx] = cc
        conv_out[idx] = conv
        iter_out[idx] = n_iter

    return u_out, v_out, dudx_out, dudy_out, dvdx_out, dvdy_out, cc_out, conv_out, iter_out


# =============================================================================
# Main DICAnalysis class
# =============================================================================

class DICAnalysis:
    """
    Digital Image Correlation analysis using IC-GN algorithm.

    The IC-GN algorithm uses an inverse compositional Gauss-Newton
    optimization with translation warp to find subpixel
    displacements between a reference and current image.
    """

    def __init__(self, params: DICParameters):
        """
        Initialize DIC analysis.

        Args:
            params: DIC parameters
        """
        self.params = params
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str = "") -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(progress, message)

    def analyze(
        self,
        ref_img: NcorrImage,
        cur_imgs: List[NcorrImage],
        roi: NcorrROI,
        seeds: List[SeedInfo],
    ) -> List[DICResult]:
        """
        Perform DIC analysis on image sequence.

        Args:
            ref_img: Reference image
            cur_imgs: List of current images
            roi: Region of interest
            seeds: Initial seed points

        Returns:
            List of DICResult for each current image
        """
        results = []
        current_seeds = seeds
        prev_result = None  # Previous result for initialization

        for i, cur_img in enumerate(cur_imgs):
            self._report_progress(i / len(cur_imgs), f"Analyzing image {i+1}/{len(cur_imgs)}")

            result = self._analyze_single(ref_img, cur_img, roi, current_seeds, prev_result)
            results.append(result)

            # Use this result as initial guess for next image
            prev_result = result

            # Update seeds for next image if step analysis enabled
            if self.params.step_analysis.enabled:
                current_seeds = result.seed_info

        return results

    def _analyze_single(
        self,
        ref_img: NcorrImage,
        cur_img: NcorrImage,
        roi: NcorrROI,
        seeds: List[SeedInfo],
        prev_result: Optional[DICResult] = None,
    ) -> DICResult:
        """
        Analyze single image pair.

        Args:
            ref_img: Reference image
            cur_img: Current image
            roi: Region of interest
            seeds: Seed points
            prev_result: Previous result to use as initial guess (for sequences)
        """
        # Get image data
        ref_bcoef = ref_img.get_bcoef()
        cur_bcoef = cur_img.get_bcoef()
        border = ref_img.border_bcoef

        # Initialize output arrays
        h, w = ref_img.height, ref_img.width
        step = self.params.spacing + 1
        out_h = (h + step - 1) // step
        out_w = (w + step - 1) // step

        u_plot = np.full((out_h, out_w), np.nan)
        v_plot = np.full((out_h, out_w), np.nan)
        corrcoef_plot = np.full((out_h, out_w), np.nan)
        roi_plot = np.zeros((out_h, out_w), dtype=np.bool_)
        converged = np.zeros((out_h, out_w), dtype=np.bool_)
        iterations = np.zeros((out_h, out_w), dtype=np.int32)

        # Get previous displacement fields for initialization
        prev_u = prev_result.u if prev_result is not None else None
        prev_v = prev_result.v if prev_result is not None else None
        prev_roi = prev_result.roi if prev_result is not None else None

        # Process each seed/region
        updated_seeds = []
        total_points = 0

        for seed in seeds:
            if not seed.valid:
                updated_seeds.append(seed)
                continue

            region_idx = seed.region_idx
            if region_idx >= len(roi.regions) or roi.regions[region_idx].is_empty():
                updated_seeds.append(SeedInfo(
                    x=seed.x, y=seed.y, u=0, v=0,
                    region_idx=region_idx, valid=False
                ))
                continue

            region = roi.regions[region_idx]

            # Estimate number of points in this region
            estimated_points = region.totalpoints // (step * step) if region.totalpoints > 0 else out_h * out_w

            # Perform IC-GN for this region using optimized flood-fill
            points_processed = self._process_region_optimized(
                ref_bcoef, cur_bcoef, border,
                region, seed,
                u_plot, v_plot, corrcoef_plot, roi_plot, converged, iterations,
                step,
                estimated_points,
                prev_u, prev_v, prev_roi
            )
            total_points += points_processed

            # Update seed for next image
            sx, sy = seed.x // step, seed.y // step
            if 0 <= sy < out_h and 0 <= sx < out_w and roi_plot[sy, sx]:
                updated_seeds.append(SeedInfo(
                    x=seed.x, y=seed.y,
                    u=u_plot[sy, sx], v=v_plot[sy, sx],
                    region_idx=region_idx,
                    valid=corrcoef_plot[sy, sx] > 0.5
                ))
            else:
                updated_seeds.append(SeedInfo(
                    x=seed.x, y=seed.y, u=seed.u, v=seed.v,
                    region_idx=region_idx, valid=False
                ))

        return DICResult(
            u=u_plot,
            v=v_plot,
            corrcoef=corrcoef_plot,
            roi=roi_plot,
            seed_info=updated_seeds,
            converged=converged,
            iterations=iterations,
        )

    def _process_region_optimized(
        self,
        ref_bcoef: NDArray[np.float64],
        cur_bcoef: NDArray[np.float64],
        border: int,
        region,
        seed: SeedInfo,
        u_plot: NDArray[np.float64],
        v_plot: NDArray[np.float64],
        corrcoef_plot: NDArray[np.float64],
        roi_plot: NDArray[np.bool_],
        converged: NDArray[np.bool_],
        iterations: NDArray[np.int32],
        step: int,
        estimated_points: int = 0,
        prev_u: Optional[NDArray[np.float64]] = None,
        prev_v: Optional[NDArray[np.float64]] = None,
        prev_roi: Optional[NDArray[np.bool_]] = None,
    ) -> int:
        """
        Process a single region using parallel wavefront propagation.

        Parallel batch approach:
        1. Process seed point first
        2. Collect all frontier points (unprocessed neighbors of processed points)
        3. Process entire frontier in parallel using Numba prange
        4. Filter results and add valid points
        5. Repeat until no more frontier points
        6. Display progress with tqdm
        """
        radius = self.params.radius
        cutoff_diffnorm = self.params.cutoff_diffnorm
        cutoff_iteration = self.params.cutoff_iteration

        # MATLAB cutoffs
        cutoff_disp = float(step)  # Displacement jump cutoff (spacing+1 in MATLAB)

        # Get region data for in-region checking
        leftbound = region.leftbound
        noderange = np.asarray(region.noderange, dtype=np.int32)
        nodelist = np.asarray(region.nodelist, dtype=np.int32)

        # Debug counters for rejection reasons
        reject_outside_roi = 0
        reject_nan_cc = 0
        reject_low_cc = 0
        reject_disp_jump = 0

        out_h, out_w = u_plot.shape

        # Store strain fields for propagation
        dudx_field = np.zeros((out_h, out_w), dtype=np.float64)
        dudy_field = np.zeros((out_h, out_w), dtype=np.float64)
        dvdx_field = np.zeros((out_h, out_w), dtype=np.float64)
        dvdy_field = np.zeros((out_h, out_w), dtype=np.float64)

        # Initialize seed using AFFINE model
        seed_u, seed_v, seed_dudx, seed_dudy, seed_dvdx, seed_dvdy, seed_cc, seed_conv, seed_iter = _ic_gn_affine(
            ref_bcoef, cur_bcoef, border,
            seed.x, seed.y, radius,
            seed.u, seed.v,
            0.0, 0.0, 0.0, 0.0,  # Initial strain guess = 0
            cutoff_diffnorm, cutoff_iteration,
        )

        if np.isnan(seed_cc) or seed_cc < 0.5:
            print(f"WARNING: Seed failed! CC={seed_cc}")
            return 0

        # Track calculated points
        calc_points = np.zeros((out_h, out_w), dtype=np.bool_)
        ox_seed, oy_seed = seed.x // step, seed.y // step
        calc_points[oy_seed, ox_seed] = True

        # Store seed result
        u_plot[oy_seed, ox_seed] = seed_u
        v_plot[oy_seed, ox_seed] = seed_v
        corrcoef_plot[oy_seed, ox_seed] = seed_cc
        roi_plot[oy_seed, ox_seed] = True
        converged[oy_seed, ox_seed] = True
        dudx_field[oy_seed, ox_seed] = seed_dudx
        dudy_field[oy_seed, ox_seed] = seed_dudy
        dvdx_field[oy_seed, ox_seed] = seed_dvdx
        dvdy_field[oy_seed, ox_seed] = seed_dvdy

        points_processed = 1

        if estimated_points <= 0:
            estimated_points = region.totalpoints // (step * step) if region.totalpoints > 0 else 10000

        # Queue of points to propagate from: (x, y, u, v, dudx, dudy, dvdx, dvdy)
        active_points = [(seed.x, seed.y, seed_u, seed_v, seed_dudx, seed_dudy, seed_dvdx, seed_dvdy)]

        # Progress bar for displacement calculation
        pbar = tqdm(total=estimated_points, desc="Displacement", unit="pts",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        pbar.update(1)  # Seed point

        while active_points:
            # Collect all frontier points from active points
            frontier = []  # List of (nx, ny, u_init, v_init, dudx_init, dudy_init, dvdx_init, dvdy_init, parent_idx)

            for parent_idx, (x, y, u, v, dudx, dudy, dvdx, dvdy) in enumerate(active_points):
                # Check neighbors - MATLAB order: top, right, bottom, left
                for dx_step, dy_step in [(0, -step), (step, 0), (0, step), (-step, 0)]:
                    nx, ny = x + dx_step, y + dy_step
                    nx_out, ny_out = nx // step, ny // step

                    if not (0 <= ny_out < out_h and 0 <= nx_out < out_w):
                        continue

                    if calc_points[ny_out, nx_out]:
                        continue

                    # Mark as being processed (to avoid duplicates in frontier)
                    calc_points[ny_out, nx_out] = True

                    if not _check_point_in_region(nx, ny, leftbound, noderange, nodelist):
                        reject_outside_roi += 1
                        continue

                    # Initial guess: propagate displacement using strain
                    u_init = u + dudx * dx_step + dudy * dy_step
                    v_init = v + dvdx * dx_step + dvdy * dy_step

                    frontier.append((nx, ny, u_init, v_init, dudx, dudy, dvdx, dvdy))

            if not frontier:
                break

            # Convert frontier to arrays for parallel processing
            n_frontier = len(frontier)
            points_x = np.array([f[0] for f in frontier], dtype=np.int32)
            points_y = np.array([f[1] for f in frontier], dtype=np.int32)
            u_init_arr = np.array([f[2] for f in frontier], dtype=np.float64)
            v_init_arr = np.array([f[3] for f in frontier], dtype=np.float64)
            dudx_init_arr = np.array([f[4] for f in frontier], dtype=np.float64)
            dudy_init_arr = np.array([f[5] for f in frontier], dtype=np.float64)
            dvdx_init_arr = np.array([f[6] for f in frontier], dtype=np.float64)
            dvdy_init_arr = np.array([f[7] for f in frontier], dtype=np.float64)

            # Process frontier in parallel (NCC search + IC-GN)
            u_out, v_out, dudx_out, dudy_out, dvdx_out, dvdy_out, cc_out, conv_out, iter_out = \
                _process_batch_with_ncc_search(
                    ref_bcoef, cur_bcoef, border,
                    points_x, points_y,
                    u_init_arr, v_init_arr,
                    dudx_init_arr, dudy_init_arr, dvdx_init_arr, dvdy_init_arr,
                    radius, cutoff_diffnorm, cutoff_iteration,
                    ncc_search_radius=2,
                )

            # Process results and collect new active points
            new_active = []
            for i in range(n_frontier):
                nx, ny = points_x[i], points_y[i]
                nx_out, ny_out = nx // step, ny // step
                new_u, new_v = u_out[i], v_out[i]
                new_cc = cc_out[i]
                u_init = u_init_arr[i]
                v_init = v_init_arr[i]

                if np.isnan(new_cc):
                    reject_nan_cc += 1
                    continue

                if new_cc < 0.0:
                    reject_low_cc += 1
                    continue

                # Displacement jump cutoff
                if abs(u_init - new_u) >= cutoff_disp or abs(v_init - new_v) >= cutoff_disp:
                    reject_disp_jump += 1
                    continue

                # Accept this point
                u_plot[ny_out, nx_out] = new_u
                v_plot[ny_out, nx_out] = new_v
                corrcoef_plot[ny_out, nx_out] = new_cc
                roi_plot[ny_out, nx_out] = True
                converged[ny_out, nx_out] = conv_out[i]
                iterations[ny_out, nx_out] = iter_out[i]

                dudx_field[ny_out, nx_out] = dudx_out[i]
                dudy_field[ny_out, nx_out] = dudy_out[i]
                dvdx_field[ny_out, nx_out] = dvdx_out[i]
                dvdy_field[ny_out, nx_out] = dvdy_out[i]

                points_processed += 1
                new_active.append((nx, ny, new_u, new_v, dudx_out[i], dudy_out[i], dvdx_out[i], dvdy_out[i]))

            # Update progress bar
            pbar.update(len(new_active))

            # Sort new_active by correlation (highest first) for better propagation order
            # This maintains some of the priority queue behavior
            active_points = sorted(new_active, key=lambda p: -corrcoef_plot[p[1] // step, p[0] // step])

        pbar.close()

        # Print summary
        print(f"\nDisplacement calculation complete:")
        print(f"  Points processed: {points_processed:,}")
        print(f"  Rejections: ROI={reject_outside_roi}, NaN={reject_nan_cc}, "
              f"LowCC={reject_low_cc}, Jump={reject_disp_jump}")

        # Print displacement statistics
        if points_processed > 0:
            valid_mask = roi_plot
            if np.any(valid_mask):
                u_valid = u_plot[valid_mask]
                v_valid = v_plot[valid_mask]
                cc_valid = corrcoef_plot[valid_mask]
                print(f"  u: [{np.min(u_valid):.3f}, {np.max(u_valid):.3f}], mean={np.mean(u_valid):.3f}")
                print(f"  v: [{np.min(v_valid):.3f}, {np.max(v_valid):.3f}], mean={np.mean(v_valid):.3f}")
                print(f"  CC: [{np.min(cc_valid):.4f}, {np.max(cc_valid):.4f}], mean={np.mean(cc_valid):.4f}")

        return points_processed


# =============================================================================
# Convenience function
# =============================================================================

def run_dic_analysis(
    ref_img: NcorrImage,
    cur_imgs: List[NcorrImage],
    roi: NcorrROI,
    seeds: List[SeedInfo],
    params: DICParameters,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[DICResult]:
    """
    Convenience function to run DIC analysis.

    Args:
        ref_img: Reference image
        cur_imgs: List of current images
        roi: Region of interest
        seeds: Initial seed points
        params: DIC parameters
        progress_callback: Optional progress callback

    Returns:
        List of DIC results
    """
    analysis = DICAnalysis(params)
    if progress_callback:
        analysis.set_progress_callback(progress_callback)

    return analysis.analyze(ref_img, cur_imgs, roi, seeds)
