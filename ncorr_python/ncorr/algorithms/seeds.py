"""
Seed calculation and placement algorithms.

Equivalent to ncorr_alg_calcseeds.cpp and ncorr_gui_seedanalysis.m

Implements FFT-accelerated NCC search for robust initial displacement estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from .dic import SeedInfo
from .bspline import BSplineInterpolator
from ..core.image import NcorrImage
from ..core.roi import NcorrROI
from ..core.dic_parameters import DICParameters


@dataclass
class SeedSearchResult:
    """Result of seed search."""

    x: int
    y: int
    u: float
    v: float
    correlation: float
    converged: bool


class SeedCalculator:
    """
    Calculate initial seed positions and displacements.

    Uses FFT-accelerated normalized cross-correlation (NCC) for initial search
    followed by subpixel refinement using IC-GN optimization.
    """

    def __init__(self, params: DICParameters):
        """
        Initialize seed calculator.

        Args:
            params: DIC parameters
        """
        self.params = params
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set progress callback."""
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str = "") -> None:
        """Report progress."""
        if self._progress_callback:
            self._progress_callback(progress, message)

    def calculate_seeds(
        self,
        ref_img: NcorrImage,
        cur_img: NcorrImage,
        roi: NcorrROI,
        search_radius: int = 150,
    ) -> List[SeedInfo]:
        """
        Calculate seed points for each region.

        Uses direct FFT-NCC search for robust displacement estimation.

        Args:
            ref_img: Reference image
            cur_img: Current (first) image
            roi: Region of interest
            search_radius: Maximum search radius for NCC (default 150 pixels)

        Returns:
            List of SeedInfo for each region
        """
        seeds = []

        # Get grayscale images
        ref_gs = ref_img.get_gs()
        cur_gs = cur_img.get_gs()

        print(f"\n{'='*60}")
        print("Seed Calculation Debug")
        print(f"{'='*60}")
        print(f"Reference image shape: {ref_gs.shape}")
        print(f"Current image shape: {cur_gs.shape}")
        print(f"Search radius: {search_radius}")
        print(f"Subset radius: {self.params.radius}")

        for region_idx, region in enumerate(roi.regions):
            self._report_progress(
                region_idx / len(roi.regions),
                f"Calculating seed for region {region_idx + 1}/{len(roi.regions)}"
            )

            if region.is_empty():
                seeds.append(SeedInfo(
                    x=0, y=0, u=0, v=0,
                    region_idx=region_idx, valid=False
                ))
                continue

            # Find center of region
            center_x, center_y = self._find_region_center(region)
            print(f"\nRegion {region_idx + 1}:")
            print(f"  Center: ({center_x}, {center_y})")

            # Perform direct FFT-NCC search (no coarse-to-fine)
            result = self._fft_ncc_search(
                ref_gs, cur_gs, center_x, center_y,
                self.params.radius, search_radius
            )

            if result is None:
                print(f"  FFT-NCC search FAILED")
                seeds.append(SeedInfo(
                    x=center_x, y=center_y, u=0, v=0,
                    region_idx=region_idx, valid=False
                ))
                continue

            print(f"  FFT-NCC result: u={result.u:.2f}, v={result.v:.2f}, CC={result.correlation:.4f}")

            # Subpixel refinement
            refined = self._subpixel_refine(
                ref_img, cur_img,
                center_x, center_y,
                result.u, result.v
            )

            if refined:
                final_u = refined.u
                final_v = refined.v
                final_cc = refined.correlation
                print(f"  Subpixel refined: u={final_u:.2f}, v={final_v:.2f}, CC={final_cc:.4f}")
            else:
                final_u = result.u
                final_v = result.v
                final_cc = result.correlation
                print(f"  Subpixel refinement failed, using NCC result")

            seeds.append(SeedInfo(
                x=center_x,
                y=center_y,
                u=final_u,
                v=final_v,
                region_idx=region_idx,
                valid=final_cc > 0.5
            ))

            print(f"  Final seed: u={final_u:.2f}, v={final_v:.2f}, CC={final_cc:.4f}, valid={final_cc > 0.5}")

        print(f"{'='*60}\n")
        return seeds

    def _find_region_center(self, region) -> Tuple[int, int]:
        """Find approximate center of a region."""
        # Use centroid of bounds
        center_x = (region.leftbound + region.rightbound) // 2
        center_y = (region.upperbound + region.lowerbound) // 2

        # Verify point is in region, if not find nearest valid point
        idx = center_x - region.leftbound
        if 0 <= idx < len(region.noderange):
            for k in range(0, region.noderange[idx], 2):
                if region.nodelist[idx, k] <= center_y <= region.nodelist[idx, k + 1]:
                    return center_x, center_y

            # Find nearest y in this column
            if region.noderange[idx] > 0:
                y_mid = (region.nodelist[idx, 0] + region.nodelist[idx, 1]) // 2
                return center_x, y_mid

        return center_x, center_y

    def _fft_ncc_search(
        self,
        ref_gs: NDArray[np.float64],
        cur_gs: NDArray[np.float64],
        x: int,
        y: int,
        radius: int,
        search_radius: int,
    ) -> Optional[SeedSearchResult]:
        """
        Perform FFT-accelerated normalized cross-correlation search.

        Args:
            ref_gs: Reference grayscale image
            cur_gs: Current grayscale image
            x, y: Center point in reference
            radius: Subset radius
            search_radius: Search radius

        Returns:
            SeedSearchResult or None if failed
        """
        h, w = ref_gs.shape

        # Extract reference template
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)

        template = ref_gs[y1:y2, x1:x2].astype(np.float64)

        if template.size == 0 or template.shape[0] < 5 or template.shape[1] < 5:
            print(f"    Template too small: {template.shape}")
            return None

        # Check template has enough contrast
        template_std = np.std(template)
        if template_std < 1e-10:
            print(f"    Template has no contrast: std={template_std}")
            return None

        # Define search area in current image
        # Search area should be large enough to cover expected displacement
        sy1 = max(0, y - radius - search_radius)
        sy2 = min(h, y + radius + search_radius + 1)
        sx1 = max(0, x - radius - search_radius)
        sx2 = min(w, x + radius + search_radius + 1)

        search_area = cur_gs[sy1:sy2, sx1:sx2].astype(np.float64)

        if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
            print(f"    Search area too small: {search_area.shape} < {template.shape}")
            return None

        # Compute NCC using FFT
        ncc_map = self._compute_ncc_fft(template, search_area)

        if ncc_map is None or ncc_map.size == 0:
            print(f"    NCC computation failed")
            return None

        # Find best match
        best_idx = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
        best_ncc = ncc_map[best_idx]
        best_dy, best_dx = best_idx

        print(f"    NCC map shape: {ncc_map.shape}, max NCC: {best_ncc:.4f} at ({best_dx}, {best_dy})")

        if best_ncc < 0.3:
            print(f"    NCC too low: {best_ncc:.4f} < 0.3")
            return None

        # Calculate displacement
        # Template was extracted from ref at [y1:y2, x1:x2]
        # Search area was extracted from cur at [sy1:sy2, sx1:sx2]
        # NCC best match at (best_dy, best_dx) in NCC map
        # With 'valid' mode, this means template placed at (best_dx, best_dy) in search area

        # Template center in ref image: (x, y)
        # Template center offset from top-left: (x - x1, y - y1)
        # Template center in cur image: (sx1 + best_dx + (x - x1), sy1 + best_dy + (y - y1))
        # Displacement: u = cur_x - ref_x, v = cur_y - ref_y

        # Simplified formula (the (x - x1) and -x terms cancel to -x1):
        u = sx1 + best_dx - x1
        v = sy1 + best_dy - y1

        print(f"    Displacement calculation:")
        print(f"      Template center in ref: ({x}, {y})")
        print(f"      Template bounds: x1={x1}, y1={y1}")
        print(f"      Search area start: sx1={sx1}, sy1={sy1}")
        print(f"      Best match in NCC: dx={best_dx}, dy={best_dy}")
        print(f"      => u = {sx1} + {best_dx} - {x1} = {u}")
        print(f"      => v = {sy1} + {best_dy} - {y1} = {v}")

        return SeedSearchResult(
            x=x, y=y, u=float(u), v=float(v),
            correlation=float(best_ncc), converged=True
        )

    def _compute_ncc_fft(
        self,
        template: NDArray[np.float64],
        search_area: NDArray[np.float64],
    ) -> Optional[NDArray[np.float64]]:
        """
        Compute NCC map using FFT convolution.

        Args:
            template: Template image
            search_area: Search area in current image

        Returns:
            NCC map or None if failed
        """
        th, tw = template.shape
        sh, sw = search_area.shape

        if sh < th or sw < tw:
            return None

        n_pixels = th * tw

        # Template statistics
        template_mean = np.mean(template)
        template_centered = template - template_mean
        template_std = np.std(template)

        if template_std < 1e-10:
            return None

        # Flip template for correlation (to get convolution = correlation)
        template_flipped = template_centered[::-1, ::-1]

        # Compute mean and std of sliding windows using convolution
        ones = np.ones(template.shape)

        # Local sums
        local_sum = fftconvolve(search_area, ones, mode='valid')
        local_sum_sq = fftconvolve(search_area ** 2, ones, mode='valid')

        # Local mean and variance
        local_mean = local_sum / n_pixels
        local_var = local_sum_sq / n_pixels - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 1e-10))

        # Cross-correlation
        cross_corr = fftconvolve(search_area, template_flipped, mode='valid')

        # NCC formula: sum((t - tm)(w - wm)) / (n * std_t * std_w)
        # Since template is centered: sum(t_centered * (w - wm)) / (n * std_t * std_w)
        # = (sum(t_centered * w) - wm * sum(t_centered)) / (n * std_t * std_w)
        # = cross_corr / (n * std_t * std_w)  [since sum(t_centered) = 0]
        ncc = cross_corr / (template_std * local_std * n_pixels)

        return ncc

    def _subpixel_refine(
        self,
        ref_img: NcorrImage,
        cur_img: NcorrImage,
        x: int,
        y: int,
        u_init: float,
        v_init: float,
    ) -> Optional[SeedSearchResult]:
        """
        Refine displacement to subpixel accuracy using IC-GN.

        Args:
            ref_img: Reference image
            cur_img: Current image
            x, y: Center point
            u_init, v_init: Initial displacement from NCC

        Returns:
            Refined SeedSearchResult or None
        """
        ref_bcoef = ref_img.get_bcoef()
        cur_bcoef = cur_img.get_bcoef()
        border = ref_img.border_bcoef
        radius = self.params.radius

        u, v = u_init, v_init

        # Get reference subset
        size = 2 * radius + 1
        ref_subset = np.zeros((size, size))
        ref_dx = np.zeros((size, size))
        ref_dy = np.zeros((size, size))
        mask = np.ones((size, size), dtype=np.bool_)

        h, w = ref_bcoef.shape

        for j in range(size):
            for i in range(size):
                px = x - radius + i + border
                py = y - radius + j + border

                if px < 2 or px >= w - 3 or py < 2 or py >= h - 3:
                    mask[j, i] = False
                    continue

                val, gx, gy = BSplineInterpolator.interpolate_with_gradient(
                    float(px), float(py), ref_bcoef, border
                )
                ref_subset[j, i] = val
                ref_dx[j, i] = gx
                ref_dy[j, i] = gy

        if np.sum(mask) < 10:
            return None

        # Normalize reference
        ref_mean = np.mean(ref_subset[mask])
        ref_centered = ref_subset - ref_mean
        ref_norm = np.sqrt(np.sum(ref_centered[mask]**2))

        if ref_norm < 1e-10:
            return None

        # Build Hessian
        grad_x = ref_dx / ref_norm
        grad_y = ref_dy / ref_norm

        H = np.zeros((2, 2))
        for j in range(size):
            for i in range(size):
                if mask[j, i]:
                    gx, gy = grad_x[j, i], grad_y[j, i]
                    H[0, 0] += gx * gx
                    H[0, 1] += gx * gy
                    H[1, 0] += gy * gx
                    H[1, 1] += gy * gy

        if np.linalg.det(H) < 1e-10:
            return None

        try:
            H_inv = np.linalg.inv(H)
        except:
            return None

        # IC-GN iterations
        error = np.zeros((size, size))
        for iteration in range(self.params.cutoff_iteration):
            # Get current subset
            cur_subset = np.zeros((size, size))
            all_valid = True

            for j in range(size):
                for i in range(size):
                    if not mask[j, i]:
                        continue

                    # Correct warp: reference position + displacement
                    dx = i - radius
                    dy = j - radius
                    px = x + dx + u + border
                    py = y + dy + v + border

                    if px < 2 or px >= w - 3 or py < 2 or py >= h - 3:
                        all_valid = False
                        cur_subset[j, i] = 0
                        continue

                    val, _, _ = BSplineInterpolator.interpolate_with_gradient(
                        px, py, cur_bcoef, border
                    )
                    cur_subset[j, i] = val

            if not all_valid:
                break

            # Normalize current
            cur_mean = np.mean(cur_subset[mask])
            cur_centered = cur_subset - cur_mean
            cur_norm = np.sqrt(np.sum(cur_centered[mask]**2))

            if cur_norm < 1e-10:
                break

            # Error
            error = (ref_centered / ref_norm) - (cur_centered / cur_norm)

            # Compute update
            b = np.zeros(2)
            for j in range(size):
                for i in range(size):
                    if mask[j, i]:
                        b[0] += grad_x[j, i] * error[j, i]
                        b[1] += grad_y[j, i] * error[j, i]

            dp = H_inv @ b
            u += dp[0]
            v += dp[1]

            if np.sqrt(dp[0]**2 + dp[1]**2) < self.params.cutoff_diffnorm:
                break

        # Final correlation
        ncc = 1.0 - 0.5 * np.sum(error[mask]**2)

        return SeedSearchResult(
            x=x, y=y, u=u, v=v,
            correlation=ncc, converged=True
        )


def calculate_seeds(
    ref_img: NcorrImage,
    cur_img: NcorrImage,
    roi: NcorrROI,
    params: DICParameters,
    search_radius: int = 150,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[SeedInfo]:
    """
    Convenience function to calculate seeds.

    Args:
        ref_img: Reference image
        cur_img: Current image
        roi: Region of interest
        params: DIC parameters
        search_radius: NCC search radius (default 150 for large displacements)
        progress_callback: Progress callback

    Returns:
        List of seed info
    """
    calculator = SeedCalculator(params)
    if progress_callback:
        calculator.set_progress_callback(progress_callback)

    return calculator.calculate_seeds(ref_img, cur_img, roi, search_radius)
