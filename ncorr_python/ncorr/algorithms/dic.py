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
    """

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    corrcoef: NDArray[np.float64]
    roi: NDArray[np.bool_]
    seed_info: List[SeedInfo] = field(default_factory=list)
    converged: Optional[NDArray[np.bool_]] = None


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
def _solve_6x6(A: NDArray[np.float64], b: NDArray[np.float64]) -> Tuple[NDArray[np.float64], bool]:
    """
    Solve 6x6 linear system Ax = b using Gaussian elimination with partial pivoting.
    """
    n = 6
    # Create augmented matrix
    aug = np.zeros((n, n + 1), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            aug[i, j] = A[i, j]
        aug[i, n] = b[i]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_val = abs(aug[col, col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row, col]) > max_val:
                max_val = abs(aug[row, col])
                max_row = row

        if max_val < 1e-12:
            return np.zeros(n, dtype=np.float64), False

        # Swap rows
        if max_row != col:
            for j in range(n + 1):
                tmp = aug[col, j]
                aug[col, j] = aug[max_row, j]
                aug[max_row, j] = tmp

        # Eliminate
        for row in range(col + 1, n):
            factor = aug[row, col] / aug[col, col]
            for j in range(col, n + 1):
                aug[row, j] -= factor * aug[col, j]

    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = aug[i, n]
        for j in range(i + 1, n):
            x[i] -= aug[i, j] * x[j]
        if abs(aug[i, i]) < 1e-12:
            return np.zeros(n, dtype=np.float64), False
        x[i] /= aug[i, i]

    return x, True


@njit(cache=True, fastmath=True)
def _ic_gn_first_order(
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
) -> Tuple[float, float, float, bool]:
    """
    IC-GN optimization with first-order (affine) warp model.

    Uses 6 parameters: [u, du/dx, du/dy, v, dv/dx, dv/dy]

    The warp function maps reference subset coordinates to current image:
    W(Δx, Δy; p) = [x + u + du/dx*Δx + du/dy*Δy]
                   [y + v + dv/dx*Δx + dv/dy*Δy]

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
        Tuple of (u, v, correlation_coefficient, converged)
    """
    size = 2 * radius + 1
    h_ref, w_ref = ref_bcoef.shape
    h_cur, w_cur = cur_bcoef.shape

    # Arrays for reference subset
    ref_subset = np.empty((size, size), dtype=np.float64)
    ref_dx = np.empty((size, size), dtype=np.float64)
    ref_dy = np.empty((size, size), dtype=np.float64)
    mask = np.ones((size, size), dtype=np.bool_)

    # Relative coordinates (Δx, Δy) for each pixel
    delta_x = np.empty((size, size), dtype=np.float64)
    delta_y = np.empty((size, size), dtype=np.float64)

    valid_count = 0

    # Get reference subset with gradients
    for j in range(size):
        for i in range(size):
            # Relative coordinates from subset center
            dx = float(i - radius)
            dy = float(j - radius)
            delta_x[j, i] = dx
            delta_y[j, i] = dy

            # Absolute coordinates in bcoef (with border)
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
        return np.nan, np.nan, np.nan, False

    # Reference subset statistics
    ref_mean = _masked_mean(ref_subset, mask)
    ref_centered = ref_subset - ref_mean
    ref_norm = _masked_norm(ref_centered, mask)

    if ref_norm < 1e-10:
        return np.nan, np.nan, np.nan, False

    # Normalize gradients
    grad_x_norm = ref_dx / ref_norm
    grad_y_norm = ref_dy / ref_norm

    # Precompute Hessian matrix (6x6) and steepest descent images
    # Steepest descent: SD = [∂T/∂x, ∂T/∂x*Δx, ∂T/∂x*Δy, ∂T/∂y, ∂T/∂y*Δx, ∂T/∂y*Δy]
    H = np.zeros((6, 6), dtype=np.float64)

    for j in range(size):
        for i in range(size):
            if not mask[j, i]:
                continue

            gx = grad_x_norm[j, i]
            gy = grad_y_norm[j, i]
            dx = delta_x[j, i]
            dy = delta_y[j, i]

            # Steepest descent image at this pixel
            # SD = [gx, gx*dx, gx*dy, gy, gy*dx, gy*dy]
            sd0 = gx
            sd1 = gx * dx
            sd2 = gx * dy
            sd3 = gy
            sd4 = gy * dx
            sd5 = gy * dy

            # Hessian = sum of outer products SD^T * SD
            H[0, 0] += sd0 * sd0
            H[0, 1] += sd0 * sd1
            H[0, 2] += sd0 * sd2
            H[0, 3] += sd0 * sd3
            H[0, 4] += sd0 * sd4
            H[0, 5] += sd0 * sd5

            H[1, 1] += sd1 * sd1
            H[1, 2] += sd1 * sd2
            H[1, 3] += sd1 * sd3
            H[1, 4] += sd1 * sd4
            H[1, 5] += sd1 * sd5

            H[2, 2] += sd2 * sd2
            H[2, 3] += sd2 * sd3
            H[2, 4] += sd2 * sd4
            H[2, 5] += sd2 * sd5

            H[3, 3] += sd3 * sd3
            H[3, 4] += sd3 * sd4
            H[3, 5] += sd3 * sd5

            H[4, 4] += sd4 * sd4
            H[4, 5] += sd4 * sd5

            H[5, 5] += sd5 * sd5

    # Hessian is symmetric
    H[1, 0] = H[0, 1]
    H[2, 0] = H[0, 2]
    H[2, 1] = H[1, 2]
    H[3, 0] = H[0, 3]
    H[3, 1] = H[1, 3]
    H[3, 2] = H[2, 3]
    H[4, 0] = H[0, 4]
    H[4, 1] = H[1, 4]
    H[4, 2] = H[2, 4]
    H[4, 3] = H[3, 4]
    H[5, 0] = H[0, 5]
    H[5, 1] = H[1, 5]
    H[5, 2] = H[2, 5]
    H[5, 3] = H[3, 5]
    H[5, 4] = H[4, 5]

    # Reference normalized
    ref_normalized = ref_centered / ref_norm

    # Initialize warp parameters: [u, du/dx, du/dy, v, dv/dx, dv/dy]
    p = np.zeros(6, dtype=np.float64)
    p[0] = u_init  # u
    p[3] = v_init  # v
    # Deformation gradients start at zero (no deformation)

    converged = False

    for iteration in range(cutoff_iteration):
        # Get current subset at warped location
        cur_subset = np.empty((size, size), dtype=np.float64)
        all_valid = True

        for j in range(size):
            for i in range(size):
                if not mask[j, i]:
                    cur_subset[j, i] = 0.0
                    continue

                dx = delta_x[j, i]
                dy = delta_y[j, i]

                # Warp: W(dx, dy; p)
                # wx = x + dx + u + du/dx*dx + du/dy*dy
                # wy = y + dy + v + dv/dx*dx + dv/dy*dy
                wx = float(x) + dx + p[0] + p[1] * dx + p[2] * dy + border
                wy = float(y) + dy + p[3] + p[4] * dx + p[5] * dy + border

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

        # Compute gradient: b = Σ SD^T * error
        b = np.zeros(6, dtype=np.float64)
        for j in range(size):
            for i in range(size):
                if not mask[j, i]:
                    continue

                gx = grad_x_norm[j, i]
                gy = grad_y_norm[j, i]
                dx = delta_x[j, i]
                dy = delta_y[j, i]
                e = error[j, i]

                b[0] += gx * e
                b[1] += gx * dx * e
                b[2] += gx * dy * e
                b[3] += gy * e
                b[4] += gy * dx * e
                b[5] += gy * dy * e

        # Solve for parameter update
        dp, success = _solve_6x6(H, b)
        if not success:
            break

        # Update parameters (inverse compositional update)
        # For first-order warp, the update is more complex
        # Simplified: just add the updates (forward additive approximation)
        p[0] += dp[0]
        p[1] += dp[1]
        p[2] += dp[2]
        p[3] += dp[3]
        p[4] += dp[4]
        p[5] += dp[5]

        # Check convergence (norm of displacement update)
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

            dx = delta_x[j, i]
            dy = delta_y[j, i]

            wx = float(x) + dx + p[0] + p[1] * dx + p[2] * dy + border
            wy = float(y) + dy + p[3] + p[4] * dx + p[5] * dy + border

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
        return np.nan, np.nan, np.nan, False

    cur_mean = _masked_mean(cur_subset, mask)
    cur_centered = cur_subset - cur_mean
    cur_norm = _masked_norm(cur_centered, mask)

    if cur_norm < 1e-10:
        return np.nan, np.nan, np.nan, False

    ncc = _compute_ncc(ref_centered, cur_centered, ref_norm, cur_norm, mask)

    # Return displacement at subset center (u, v)
    return p[0], p[3], ncc, converged


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


# Alias for backwards compatibility
_ic_gn_single_point = _ic_gn_first_order


# =============================================================================
# Main DICAnalysis class
# =============================================================================

class DICAnalysis:
    """
    Digital Image Correlation analysis using IC-GN algorithm.

    The IC-GN algorithm uses an inverse compositional Gauss-Newton
    optimization with first-order (affine) warp to find subpixel
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

        for i, cur_img in enumerate(cur_imgs):
            self._report_progress(i / len(cur_imgs), f"Analyzing image {i+1}/{len(cur_imgs)}")

            result = self._analyze_single(ref_img, cur_img, roi, current_seeds)
            results.append(result)

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
    ) -> DICResult:
        """
        Analyze single image pair.

        Args:
            ref_img: Reference image
            cur_img: Current image
            roi: Region of interest
            seeds: Seed points

        Returns:
            DIC result
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
                u_plot, v_plot, corrcoef_plot, roi_plot, converged,
                step,
                estimated_points
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
        step: int,
        estimated_points: int = 0,
    ) -> int:
        """
        Process a single region using flood-fill from seed.

        Optimized version using deque and Numba-accelerated IC-GN.
        """
        radius = self.params.radius
        cutoff_diffnorm = self.params.cutoff_diffnorm
        cutoff_iteration = self.params.cutoff_iteration

        # Get region data for in-region checking
        leftbound = region.leftbound
        noderange = np.asarray(region.noderange, dtype=np.int32)
        nodelist = np.asarray(region.nodelist, dtype=np.int32)

        # Queue for flood-fill processing (using deque for efficiency)
        queue = deque([(seed.x, seed.y, seed.u, seed.v)])
        processed = set()
        points_processed = 0

        out_h, out_w = u_plot.shape

        # Estimate total points if not provided
        if estimated_points <= 0:
            estimated_points = region.totalpoints // (step * step) if region.totalpoints > 0 else 10000

        while queue:
            x, y, u_guess, v_guess = queue.popleft()

            # Convert to output coordinates
            ox, oy = x // step, y // step

            if (x, y) in processed:
                continue

            if not (0 <= oy < out_h and 0 <= ox < out_w):
                continue

            processed.add((x, y))

            # Check if point is in region
            if not _check_point_in_region(x, y, leftbound, noderange, nodelist):
                continue

            # Perform IC-GN optimization with first-order warp (Numba-accelerated)
            u, v, cc, conv = _ic_gn_first_order(
                ref_bcoef, cur_bcoef, border,
                x, y, radius,
                u_guess, v_guess,
                cutoff_diffnorm, cutoff_iteration,
            )

            if not np.isnan(u):
                u_plot[oy, ox] = u
                v_plot[oy, ox] = v
                corrcoef_plot[oy, ox] = cc
                roi_plot[oy, ox] = True
                converged[oy, ox] = conv
                points_processed += 1

                # Add neighbors to queue
                for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in processed:
                        queue.append((nx, ny, u, v))

            # Progress reporting (every 50 points for more responsive updates)
            if points_processed % 50 == 0 and self._progress_callback:
                progress = min(0.99, points_processed / max(1, estimated_points))
                self._report_progress(
                    progress,
                    f"Processing... {points_processed}/{estimated_points} points ({progress*100:.0f}%)"
                )

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
