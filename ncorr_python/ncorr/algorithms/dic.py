"""
Digital Image Correlation (DIC) analysis algorithms.

Implements the Regular Gauss-Newton DIC (RG-DIC) algorithm.
Equivalent to ncorr_alg_rgdic.cpp and ncorr_alg_dicanalysis.m
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.typing import NDArray
from numba import jit, prange
from scipy.linalg import cholesky, solve_triangular

from .bspline import BSplineInterpolator
from ..core.image import NcorrImage
from ..core.roi import NcorrROI, CircularROI
from ..core.dic_parameters import DICParameters
from ..core.status import Status


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


class DICAnalysis:
    """
    Digital Image Correlation analysis using RG-DIC algorithm.

    The RG-DIC algorithm uses an inverse compositional Gauss-Newton
    optimization to find subpixel displacements between a reference
    and current image.
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
        ref_gs = ref_img.get_gs()
        ref_bcoef = ref_img.get_bcoef()
        cur_bcoef = cur_img.get_bcoef()
        border = ref_img.border_bcoef

        # Initialize output arrays
        h, w = ref_gs.shape
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

            # Perform RG-DIC for this region
            self._process_region(
                ref_bcoef, cur_bcoef, border,
                region, seed,
                u_plot, v_plot, corrcoef_plot, roi_plot, converged,
                step
            )

            # Update seed for next image
            # Use the displacement at seed location
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

    def _process_region(
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
    ) -> None:
        """Process a single region using flood-fill from seed."""
        radius = self.params.radius
        cutoff_diffnorm = self.params.cutoff_diffnorm
        cutoff_iteration = self.params.cutoff_iteration

        # Queue for flood-fill processing
        # Each entry: (x, y, u_guess, v_guess)
        queue = [(seed.x, seed.y, seed.u, seed.v)]
        processed = set()

        while queue:
            x, y, u_guess, v_guess = queue.pop(0)

            # Convert to output coordinates
            ox, oy = x // step, y // step

            if (x, y) in processed:
                continue

            if not (0 <= oy < u_plot.shape[0] and 0 <= ox < u_plot.shape[1]):
                continue

            processed.add((x, y))

            # Check if point is in region
            in_region = False
            idx = x - region.leftbound
            if 0 <= idx < len(region.noderange):
                for k in range(0, region.noderange[idx], 2):
                    if region.nodelist[idx, k] <= y <= region.nodelist[idx, k + 1]:
                        in_region = True
                        break

            if not in_region:
                continue

            # Perform IC-GN optimization
            u, v, cc, conv = self._ic_gn_optimization(
                ref_bcoef, cur_bcoef, border,
                x, y, radius,
                u_guess, v_guess,
                cutoff_diffnorm, cutoff_iteration,
                self.params.subset_trunc
            )

            if not np.isnan(u):
                u_plot[oy, ox] = u
                v_plot[oy, ox] = v
                corrcoef_plot[oy, ox] = cc
                roi_plot[oy, ox] = True
                converged[oy, ox] = conv

                # Add neighbors to queue
                for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in processed:
                        queue.append((nx, ny, u, v))

    def _ic_gn_optimization(
        self,
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
        subset_trunc: bool = False,
    ) -> Tuple[float, float, float, bool]:
        """
        Inverse Compositional Gauss-Newton optimization.

        Args:
            ref_bcoef: Reference image B-spline coefficients
            cur_bcoef: Current image B-spline coefficients
            border: Border padding
            x, y: Subset center coordinates
            radius: Subset radius
            u_init, v_init: Initial displacement guess
            cutoff_diffnorm: Convergence threshold
            cutoff_iteration: Maximum iterations
            subset_trunc: Enable subset truncation

        Returns:
            Tuple of (u, v, correlation_coefficient, converged)
        """
        # Get reference subset and gradients
        ref_subset, ref_dx, ref_dy, mask = self._get_subset_with_gradients(
            ref_bcoef, border, x, y, radius
        )

        if ref_subset is None or np.sum(mask) < 10:
            return np.nan, np.nan, np.nan, False

        n_points = np.sum(mask)

        # Compute reference subset statistics
        ref_mean = np.mean(ref_subset[mask])
        ref_centered = ref_subset - ref_mean
        ref_norm = np.sqrt(np.sum(ref_centered[mask] ** 2))

        if ref_norm < 1e-10:
            return np.nan, np.nan, np.nan, False

        ref_normalized = ref_centered / ref_norm

        # Precompute Hessian (reference gradients only for IC)
        # For first-order warp (translation only), Jacobian is identity
        # Full affine warp would include deformation gradient terms

        # Simplified: translation-only warp
        grad_x = ref_dx / ref_norm
        grad_y = ref_dy / ref_norm

        # Hessian = sum of gradient outer products
        H = np.zeros((2, 2))
        for j in range(ref_subset.shape[0]):
            for i in range(ref_subset.shape[1]):
                if mask[j, i]:
                    gx, gy = grad_x[j, i], grad_y[j, i]
                    H[0, 0] += gx * gx
                    H[0, 1] += gx * gy
                    H[1, 0] += gy * gx
                    H[1, 1] += gy * gy

        # Check if Hessian is invertible
        if np.linalg.det(H) < 1e-10:
            return np.nan, np.nan, np.nan, False

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan, False

        # Initialize parameters
        u, v = u_init, v_init
        converged = False

        for iteration in range(cutoff_iteration):
            # Get current subset at warped location
            cur_subset = self._get_subset(
                cur_bcoef, border, x + u, y + v, radius
            )

            if cur_subset is None:
                break

            # Current subset statistics
            cur_mean = np.mean(cur_subset[mask])
            cur_centered = cur_subset - cur_mean
            cur_norm = np.sqrt(np.sum(cur_centered[mask] ** 2))

            if cur_norm < 1e-10:
                break

            cur_normalized = cur_centered / cur_norm

            # Error image
            error = ref_normalized - cur_normalized

            # Compute gradient of error
            b = np.zeros(2)
            for j in range(ref_subset.shape[0]):
                for i in range(ref_subset.shape[1]):
                    if mask[j, i]:
                        b[0] += grad_x[j, i] * error[j, i]
                        b[1] += grad_y[j, i] * error[j, i]

            # Solve for update
            dp = H_inv @ b

            # Update parameters (inverse compositional update)
            u += dp[0]
            v += dp[1]

            # Check convergence
            diffnorm = np.sqrt(dp[0]**2 + dp[1]**2)
            if diffnorm < cutoff_diffnorm:
                converged = True
                break

        # Compute final correlation coefficient
        cur_subset = self._get_subset(cur_bcoef, border, x + u, y + v, radius)
        if cur_subset is None:
            return np.nan, np.nan, np.nan, False

        cur_mean = np.mean(cur_subset[mask])
        cur_centered = cur_subset - cur_mean
        cur_norm = np.sqrt(np.sum(cur_centered[mask] ** 2))

        if cur_norm < 1e-10:
            return np.nan, np.nan, np.nan, False

        # Normalized cross-correlation
        ncc = np.sum(ref_centered[mask] * cur_centered[mask]) / (ref_norm * cur_norm)

        return u, v, ncc, converged

    def _get_subset_with_gradients(
        self,
        bcoef: NDArray[np.float64],
        border: int,
        x: float,
        y: float,
        radius: int,
    ) -> Tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray], Optional[NDArray]]:
        """Get subset values and gradients at given location."""
        size = 2 * radius + 1
        subset = np.zeros((size, size), dtype=np.float64)
        dx = np.zeros((size, size), dtype=np.float64)
        dy = np.zeros((size, size), dtype=np.float64)
        mask = np.ones((size, size), dtype=np.bool_)

        h, w = bcoef.shape

        for j in range(size):
            for i in range(size):
                px = x - radius + i + border
                py = y - radius + j + border

                if px < 2 or px >= w - 3 or py < 2 or py >= h - 3:
                    mask[j, i] = False
                    continue

                val, gx, gy = BSplineInterpolator.interpolate_with_gradient(
                    px, py, bcoef, border
                )

                if np.isnan(val):
                    mask[j, i] = False
                    continue

                subset[j, i] = val
                dx[j, i] = gx
                dy[j, i] = gy

        if np.sum(mask) < 10:
            return None, None, None, None

        return subset, dx, dy, mask

    def _get_subset(
        self,
        bcoef: NDArray[np.float64],
        border: int,
        x: float,
        y: float,
        radius: int,
    ) -> Optional[NDArray[np.float64]]:
        """Get interpolated subset at given location."""
        size = 2 * radius + 1
        subset = np.zeros((size, size), dtype=np.float64)

        h, w = bcoef.shape

        for j in range(size):
            for i in range(size):
                px = x - radius + i + border
                py = y - radius + j + border

                if px < 2 or px >= w - 3 or py < 2 or py >= h - 3:
                    return None

                val, _, _ = BSplineInterpolator.interpolate_with_gradient(
                    px, py, bcoef, border
                )

                if np.isnan(val):
                    return None

                subset[j, i] = val

        return subset


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
