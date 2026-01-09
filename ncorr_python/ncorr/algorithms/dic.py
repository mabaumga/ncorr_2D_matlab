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


# =============================================================================
# Module-level Numba-accelerated functions for IC-GN optimization
# =============================================================================

@njit(cache=True, fastmath=True)
def _bspline_interp(px: float, py: float, bcoef: NDArray[np.float64]) -> float:
    """Inline B-spline interpolation for a single point (value only)."""
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
            tmp = 3.0 - t_y
            by = tmp * tmp * tmp * tmp * tmp / 120.0
        elif t_y >= 1.0:
            t2 = t_y * t_y
            t3 = t2 * t_y
            t4 = t3 * t_y
            t5 = t4 * t_y
            by = (-5.0 * t5 + 75.0 * t4 - 450.0 * t3 + 1350.0 * t2 - 2025.0 * t_y + 1215.0) / 120.0
        else:
            t2 = t_y * t_y
            t4 = t2 * t2
            by = (33.0 - 30.0 * t2 + 5.0 * t4) / 60.0 - t2 * (1.0 - t2) / 12.0

        for ii in range(-2, 4):
            t_x = abs(fx - ii)
            if t_x >= 3.0:
                bx = 0.0
            elif t_x >= 2.0:
                tmp = 3.0 - t_x
                bx = tmp * tmp * tmp * tmp * tmp / 120.0
            elif t_x >= 1.0:
                t2 = t_x * t_x
                t3 = t2 * t_x
                t4 = t3 * t_x
                t5 = t4 * t_x
                bx = (-5.0 * t5 + 75.0 * t4 - 450.0 * t3 + 1350.0 * t2 - 2025.0 * t_x + 1215.0) / 120.0
            else:
                t2 = t_x * t_x
                t4 = t2 * t2
                bx = (33.0 - 30.0 * t2 + 5.0 * t4) / 60.0 - t2 * (1.0 - t2) / 12.0

            value += bcoef[iy + jj, ix + ii] * bx * by

    return value


@njit(cache=True, fastmath=True)
def _bspline_interp_with_grad(
    px: float, py: float, bcoef: NDArray[np.float64]
) -> Tuple[float, float, float]:
    """Inline B-spline interpolation with gradients."""
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
            tmp = 3.0 - t_y
            by = tmp * tmp * tmp * tmp * tmp / 120.0
            dby = -tmp * tmp * tmp * tmp / 24.0
        elif t_y >= 1.0:
            t2 = t_y * t_y
            t3 = t2 * t_y
            t4 = t3 * t_y
            t5 = t4 * t_y
            by = (-5.0 * t5 + 75.0 * t4 - 450.0 * t3 + 1350.0 * t2 - 2025.0 * t_y + 1215.0) / 120.0
            dby = (-25.0 * t4 + 300.0 * t3 - 1350.0 * t2 + 2700.0 * t_y - 2025.0) / 120.0
        else:
            t2 = t_y * t_y
            t4 = t2 * t2
            by = (33.0 - 30.0 * t2 + 5.0 * t4) / 60.0 - t2 * (1.0 - t2) / 12.0
            t3 = t2 * t_y
            dby = (-60.0 * t_y + 20.0 * t3) / 60.0

        if (fy - jj) < 0:
            dby = -dby

        for ii in range(-2, 4):
            t_x = abs(fx - ii)
            if t_x >= 3.0:
                bx = 0.0
                dbx = 0.0
            elif t_x >= 2.0:
                tmp = 3.0 - t_x
                bx = tmp * tmp * tmp * tmp * tmp / 120.0
                dbx = -tmp * tmp * tmp * tmp / 24.0
            elif t_x >= 1.0:
                t2 = t_x * t_x
                t3 = t2 * t_x
                t4 = t3 * t_x
                t5 = t4 * t_x
                bx = (-5.0 * t5 + 75.0 * t4 - 450.0 * t3 + 1350.0 * t2 - 2025.0 * t_x + 1215.0) / 120.0
                dbx = (-25.0 * t4 + 300.0 * t3 - 1350.0 * t2 + 2700.0 * t_x - 2025.0) / 120.0
            else:
                t2 = t_x * t_x
                t4 = t2 * t2
                bx = (33.0 - 30.0 * t2 + 5.0 * t4) / 60.0 - t2 * (1.0 - t2) / 12.0
                t3 = t2 * t_x
                dbx = (-60.0 * t_x + 20.0 * t3) / 60.0

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
        Process a single region using flood-fill from seed.

        Hybrid approach:
        1. Priority queue (heapq) - highest correlation processed first (like MATLAB)
        2. Affine model (6-parameter) to capture strain/deformation
        3. Displacement jump cutoff (spacing+1 pixels)
        4. MATLAB-identical neighbor order: top, right, bottom, left
        """
        radius = self.params.radius
        cutoff_diffnorm = self.params.cutoff_diffnorm
        cutoff_iteration = self.params.cutoff_iteration

        # Check if we have previous results to use as initial guesses
        use_prev_result = (prev_u is not None and prev_v is not None and prev_roi is not None)

        # MATLAB cutoffs
        cutoff_disp = float(step)  # Displacement jump cutoff (spacing+1 in MATLAB)

        # Get region data for in-region checking
        leftbound = region.leftbound
        noderange = np.asarray(region.noderange, dtype=np.int32)
        nodelist = np.asarray(region.nodelist, dtype=np.int32)

        # DEBUG
        print(f"\n{'='*60}")
        print(f"DEBUG: Using AFFINE Model (6 parameters) for strain measurement")
        print(f"  Seed: ({seed.x}, {seed.y}), u={seed.u:.4f}, v={seed.v:.4f}")
        print(f"  Step: {step}, Cutoff: {cutoff_disp}")
        print(f"  Image size: {ref_bcoef.shape}, Border: {border}")
        print(f"  ROI leftbound: {leftbound}, noderange len: {len(noderange)}")
        print(f"{'='*60}")

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

        print(f"  Seed result: u={seed_u:.4f}, v={seed_v:.4f}, CC={seed_cc:.4f}")
        print(f"  Seed strain: dudx={seed_dudx:.6f}, dudy={seed_dudy:.6f}, dvdx={seed_dvdx:.6f}, dvdy={seed_dvdy:.6f}")

        # Priority queue: (neg_cc, insertion_order, x, y, u, v, dudx, dudy, dvdx, dvdy)
        # Using negative CC because heapq is a min-heap
        insertion_counter = 0
        heap = []
        heapq.heappush(heap, (-seed_cc, insertion_counter, seed.x, seed.y,
                              seed_u, seed_v, seed_dudx, seed_dudy, seed_dvdx, seed_dvdy))
        insertion_counter += 1

        # Track calculated points
        calc_points = np.zeros((out_h, out_w), dtype=np.bool_)
        ox_seed, oy_seed = seed.x // step, seed.y // step
        calc_points[oy_seed, ox_seed] = True

        points_processed = 0
        last_progress_percent = -1

        if estimated_points <= 0:
            estimated_points = region.totalpoints // (step * step) if region.totalpoints > 0 else 10000

        while heap:
            # Pop point with highest correlation
            neg_cc, _, x, y, u, v, dudx, dudy, dvdx, dvdy = heapq.heappop(heap)
            cc = -neg_cc

            ox, oy = x // step, y // step

            # Store result
            u_plot[oy, ox] = u
            v_plot[oy, ox] = v
            corrcoef_plot[oy, ox] = cc
            roi_plot[oy, ox] = True
            converged[oy, ox] = True
            iterations[oy, ox] = 0

            # Store strain for propagation
            dudx_field[oy, ox] = dudx
            dudy_field[oy, ox] = dudy
            dvdx_field[oy, ox] = dvdx
            dvdy_field[oy, ox] = dvdy

            points_processed += 1

            # Progress
            current_percent = int(points_processed * 100 / max(1, estimated_points))
            if current_percent > last_progress_percent and self._progress_callback:
                last_progress_percent = current_percent
                self._report_progress(min(0.99, points_processed / max(1, estimated_points)),
                                      f"Processing... {points_processed:,} points")

            # Analyze neighbors - MATLAB order: top, right, bottom, left
            for dx_step, dy_step in [(0, -step), (step, 0), (0, step), (-step, 0)]:
                nx, ny = x + dx_step, y + dy_step
                nx_out, ny_out = nx // step, ny // step

                if not (0 <= ny_out < out_h and 0 <= nx_out < out_w):
                    continue

                if calc_points[ny_out, nx_out]:
                    continue

                if not _check_point_in_region(nx, ny, leftbound, noderange, nodelist):
                    calc_points[ny_out, nx_out] = True
                    reject_outside_roi += 1
                    continue

                # Initial guess: propagate displacement using strain
                # u_new = u + dudx * dx + dudy * dy
                # v_new = v + dvdx * dx + dvdy * dy
                u_init = u + dudx * dx_step + dudy * dy_step
                v_init = v + dvdx * dx_step + dvdy * dy_step

                # Initial strain guess: use parent's strain
                dudx_init = dudx
                dudy_init = dudy
                dvdx_init = dvdx
                dvdy_init = dvdy

                # Run IC-GN with AFFINE model
                new_u, new_v, new_dudx, new_dudy, new_dvdx, new_dvdy, new_cc, new_conv, new_iter = _ic_gn_affine(
                    ref_bcoef, cur_bcoef, border,
                    nx, ny, radius,
                    u_init, v_init,
                    dudx_init, dudy_init, dvdx_init, dvdy_init,
                    cutoff_diffnorm, cutoff_iteration,
                )

                calc_points[ny_out, nx_out] = True

                if np.isnan(new_cc):
                    reject_nan_cc += 1
                    if reject_nan_cc <= 5:  # Print first 5 NaN rejections
                        print(f"  NaN at ({nx}, {ny}): init=({u_init:.2f}, {v_init:.2f})")
                    continue

                if new_cc < 0.0:
                    reject_low_cc += 1
                    if reject_low_cc <= 10:  # Print first 10 low CC rejections
                        print(f"  Low CC at ({nx}, {ny}): init=({u_init:.2f}, {v_init:.2f}), "
                              f"result=({new_u:.2f}, {new_v:.2f}), CC={new_cc:.4f}")
                    continue

                # Displacement jump cutoff
                if abs(u_init - new_u) >= cutoff_disp or abs(v_init - new_v) >= cutoff_disp:
                    reject_disp_jump += 1
                    if reject_disp_jump <= 5:  # Print first 5 displacement jump rejections
                        print(f"  Disp jump at ({nx}, {ny}): init=({u_init:.2f}, {v_init:.2f}), "
                              f"result=({new_u:.2f}, {new_v:.2f}), CC={new_cc:.4f}")
                    continue

                heapq.heappush(heap, (-new_cc, insertion_counter, nx, ny,
                                      new_u, new_v, new_dudx, new_dudy, new_dvdx, new_dvdy))
                insertion_counter += 1

        # Debug summary
        print(f"\n{'='*60}")
        print(f"DEBUG: Propagation Summary")
        print(f"  Points processed: {points_processed}")
        print(f"  Rejections:")
        print(f"    Outside ROI:       {reject_outside_roi}")
        print(f"    NaN correlation:   {reject_nan_cc}")
        print(f"    Low correlation:   {reject_low_cc}")
        print(f"    Displacement jump: {reject_disp_jump}")
        print(f"{'='*60}")

        # Extended debug: Show results summary and grid
        if points_processed > 0:
            valid_mask = roi_plot
            if np.any(valid_mask):
                u_valid = u_plot[valid_mask]
                v_valid = v_plot[valid_mask]
                cc_valid = corrcoef_plot[valid_mask]
                dudx_valid = dudx_field[valid_mask]
                dvdy_valid = dvdy_field[valid_mask]

                print(f"\n{'='*60}")
                print(f"DEBUG: Results Summary")
                print(f"{'='*60}")
                print(f"  Displacement u: min={np.min(u_valid):.4f}, max={np.max(u_valid):.4f}, "
                      f"mean={np.mean(u_valid):.4f}, std={np.std(u_valid):.4f}")
                print(f"  Displacement v: min={np.min(v_valid):.4f}, max={np.max(v_valid):.4f}, "
                      f"mean={np.mean(v_valid):.4f}, std={np.std(v_valid):.4f}")
                print(f"  Correlation CC: min={np.min(cc_valid):.4f}, max={np.max(cc_valid):.4f}, "
                      f"mean={np.mean(cc_valid):.4f}, std={np.std(cc_valid):.4f}")
                print(f"  Strain dudx:    min={np.min(dudx_valid):.6f}, max={np.max(dudx_valid):.6f}, "
                      f"mean={np.mean(dudx_valid):.6f}")
                print(f"  Strain dvdy:    min={np.min(dvdy_valid):.6f}, max={np.max(dvdy_valid):.6f}, "
                      f"mean={np.mean(dvdy_valid):.6f}")

                # Show coarse grid of u values (every 10th point)
                print(f"\n  Coarse grid of u-displacement (every 10th point):")
                grid_step = 10
                for iy in range(0, out_h, grid_step):
                    row_str = f"    y={iy:3d}: "
                    for ix in range(0, out_w, grid_step):
                        if roi_plot[iy, ix]:
                            row_str += f"{u_plot[iy, ix]:7.2f}"
                        else:
                            row_str += "      -"
                    print(row_str)

                # Show coarse grid of CC values
                print(f"\n  Coarse grid of correlation coefficient (every 10th point):")
                for iy in range(0, out_h, grid_step):
                    row_str = f"    y={iy:3d}: "
                    for ix in range(0, out_w, grid_step):
                        if roi_plot[iy, ix]:
                            row_str += f"{corrcoef_plot[iy, ix]:7.3f}"
                        else:
                            row_str += "      -"
                    print(row_str)

                print(f"{'='*60}")

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
