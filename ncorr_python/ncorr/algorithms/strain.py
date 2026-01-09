"""
Strain calculation algorithms.

Computes Green-Lagrange and Eulerian-Almansi strains from displacement fields.
Equivalent to ncorr_alg_dispgrad.cpp

All performance-critical functions are at module level with Numba acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from ..core.roi import NcorrROI


@dataclass
class StrainResult:
    """
    Strain calculation results.

    Attributes:
        exx: Strain component xx
        exy: Strain component xy (shear)
        eyy: Strain component yy
        roi: Valid data mask
        dudx: Displacement gradient du/dx
        dudy: Displacement gradient du/dy
        dvdx: Displacement gradient dv/dx
        dvdy: Displacement gradient dv/dy
    """

    exx: NDArray[np.float64]
    exy: NDArray[np.float64]
    eyy: NDArray[np.float64]
    roi: NDArray[np.bool_]
    dudx: Optional[NDArray[np.float64]] = None
    dudy: Optional[NDArray[np.float64]] = None
    dvdx: Optional[NDArray[np.float64]] = None
    dvdy: Optional[NDArray[np.float64]] = None


# =============================================================================
# Module-level Numba-accelerated functions
# =============================================================================

@njit(cache=True, parallel=True, fastmath=True)
def _calculate_gradients_numba(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    mask: NDArray[np.bool_],
    r: int,
    spacing: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """
    Calculate displacement gradients using central differences.

    Numba-accelerated version with parallel processing.

    Args:
        u: U displacement field
        v: V displacement field
        mask: Valid data mask
        r: Strain radius
        spacing: Pixel spacing

    Returns:
        Tuple of (dudx, dudy, dvdx, dvdy, valid_mask)
    """
    h, w = u.shape

    dudx = np.full((h, w), np.nan, dtype=np.float64)
    dudy = np.full((h, w), np.nan, dtype=np.float64)
    dvdx = np.full((h, w), np.nan, dtype=np.float64)
    dvdy = np.full((h, w), np.nan, dtype=np.float64)
    valid = np.zeros((h, w), dtype=np.bool_)

    scale = 1.0 / (2.0 * r * spacing)

    for j in prange(r, h - r):
        for i in range(r, w - r):
            if not mask[j, i]:
                continue

            # Check center point
            if np.isnan(u[j, i]) or np.isnan(v[j, i]):
                continue

            # Check boundary points only (much faster than checking all points)
            u_left = u[j, i - r]
            u_right = u[j, i + r]
            u_top = u[j - r, i]
            u_bottom = u[j + r, i]
            v_left = v[j, i - r]
            v_right = v[j, i + r]
            v_top = v[j - r, i]
            v_bottom = v[j + r, i]

            if (np.isnan(u_left) or np.isnan(u_right) or
                np.isnan(u_top) or np.isnan(u_bottom) or
                np.isnan(v_left) or np.isnan(v_right) or
                np.isnan(v_top) or np.isnan(v_bottom)):
                continue

            # Central differences
            dudx[j, i] = (u_right - u_left) * scale
            dudy[j, i] = (u_bottom - u_top) * scale
            dvdx[j, i] = (v_right - v_left) * scale
            dvdy[j, i] = (v_bottom - v_top) * scale

            valid[j, i] = True

    return dudx, dudy, dvdx, dvdy, valid


@njit(cache=True, parallel=True, fastmath=True)
def _compute_green_lagrange_strains(
    dudx: NDArray[np.float64],
    dudy: NDArray[np.float64],
    dvdx: NDArray[np.float64],
    dvdy: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Green-Lagrange strains from displacement gradients.

    E = 1/2 * (F^T * F - I)

    Args:
        dudx, dudy, dvdx, dvdy: Displacement gradients
        valid: Valid data mask

    Returns:
        Tuple of (exx, exy, eyy)
    """
    h, w = valid.shape
    exx = np.full((h, w), np.nan, dtype=np.float64)
    exy = np.full((h, w), np.nan, dtype=np.float64)
    eyy = np.full((h, w), np.nan, dtype=np.float64)

    for j in prange(h):
        for i in range(w):
            if not valid[j, i]:
                continue

            ux = dudx[j, i]
            uy = dudy[j, i]
            vx = dvdx[j, i]
            vy = dvdy[j, i]

            # Exx = du/dx + 0.5 * ((du/dx)^2 + (dv/dx)^2)
            exx[j, i] = ux + 0.5 * (ux * ux + vx * vx)

            # Eyy = dv/dy + 0.5 * ((du/dy)^2 + (dv/dy)^2)
            eyy[j, i] = vy + 0.5 * (uy * uy + vy * vy)

            # Exy = 0.5 * (du/dy + dv/dx + du/dx*du/dy + dv/dx*dv/dy)
            exy[j, i] = 0.5 * (uy + vx + ux * uy + vx * vy)

    return exx, exy, eyy


@njit(cache=True, parallel=True, fastmath=True)
def _compute_eulerian_almansi_strains(
    dudx: NDArray[np.float64],
    dudy: NDArray[np.float64],
    dvdx: NDArray[np.float64],
    dvdy: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Eulerian-Almansi strains from displacement gradients.

    e = 1/2 * (I - F^(-T) * F^(-1))

    Args:
        dudx, dudy, dvdx, dvdy: Displacement gradients
        valid: Valid data mask

    Returns:
        Tuple of (exx, exy, eyy)
    """
    h, w = valid.shape
    exx = np.full((h, w), np.nan, dtype=np.float64)
    exy = np.full((h, w), np.nan, dtype=np.float64)
    eyy = np.full((h, w), np.nan, dtype=np.float64)

    for j in prange(h):
        for i in range(w):
            if not valid[j, i]:
                continue

            ux = dudx[j, i]
            uy = dudy[j, i]
            vx = dvdx[j, i]
            vy = dvdy[j, i]

            # exx = du/dx - 0.5 * ((du/dx)^2 + (dv/dx)^2)
            exx[j, i] = ux - 0.5 * (ux * ux + vx * vx)

            # eyy = dv/dy - 0.5 * ((du/dy)^2 + (dv/dy)^2)
            eyy[j, i] = vy - 0.5 * (uy * uy + vy * vy)

            # exy = 0.5 * (du/dy + dv/dx - du/dx*du/dy - dv/dx*dv/dy)
            exy[j, i] = 0.5 * (uy + vx - ux * uy - vx * vy)

    return exx, exy, eyy


# =============================================================================
# StrainCalculator class
# =============================================================================

class StrainCalculator:
    """
    Calculate strain fields from displacement data.

    Computes displacement gradients using finite differences and
    calculates both Green-Lagrange (reference configuration) and
    Eulerian-Almansi (current configuration) strains.
    """

    def __init__(self, strain_radius: int = 5):
        """
        Initialize strain calculator.

        Args:
            strain_radius: Radius for gradient calculation (pixels)
        """
        self.strain_radius = strain_radius

    def calculate_green_lagrange(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        roi: NcorrROI,
        spacing: int = 1,
    ) -> StrainResult:
        """
        Calculate Green-Lagrange strain (reference configuration).

        E = 1/2 * (F^T * F - I)

        Args:
            u: U displacement field
            v: V displacement field
            roi: Region of interest
            spacing: Pixel spacing for gradient calculation

        Returns:
            StrainResult with Green-Lagrange strains
        """
        # Calculate displacement gradients using Numba
        dudx, dudy, dvdx, dvdy, grad_roi = _calculate_gradients_numba(
            u, v, roi.mask, self.strain_radius, float(spacing)
        )

        # Calculate strains using Numba
        exx, exy, eyy = _compute_green_lagrange_strains(
            dudx, dudy, dvdx, dvdy, grad_roi
        )

        return StrainResult(
            exx=exx,
            exy=exy,
            eyy=eyy,
            roi=grad_roi,
            dudx=dudx,
            dudy=dudy,
            dvdx=dvdx,
            dvdy=dvdy,
        )

    def calculate_eulerian_almansi(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        roi: NcorrROI,
        spacing: int = 1,
    ) -> StrainResult:
        """
        Calculate Eulerian-Almansi strain (current configuration).

        e = 1/2 * (I - F^(-T) * F^(-1))

        Args:
            u: U displacement field
            v: V displacement field
            roi: Region of interest
            spacing: Pixel spacing

        Returns:
            StrainResult with Eulerian-Almansi strains
        """
        # Calculate displacement gradients using Numba
        dudx, dudy, dvdx, dvdy, grad_roi = _calculate_gradients_numba(
            u, v, roi.mask, self.strain_radius, float(spacing)
        )

        # Calculate strains using Numba
        exx, exy, eyy = _compute_eulerian_almansi_strains(
            dudx, dudy, dvdx, dvdy, grad_roi
        )

        return StrainResult(
            exx=exx,
            exy=exy,
            eyy=eyy,
            roi=grad_roi,
            dudx=dudx,
            dudy=dudy,
            dvdx=dvdx,
            dvdy=dvdy,
        )

    @staticmethod
    def calculate_principal_strains(
        exx: NDArray[np.float64],
        exy: NDArray[np.float64],
        eyy: NDArray[np.float64],
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Calculate principal strains and angle.

        Args:
            exx: Strain component xx
            exy: Strain component xy
            eyy: Strain component yy

        Returns:
            Tuple of (e1, e2, theta) - principal strains and angle
        """
        avg = 0.5 * (exx + eyy)
        diff = 0.5 * (exx - eyy)
        r = np.sqrt(diff**2 + exy**2)

        e1 = avg + r  # Maximum principal strain
        e2 = avg - r  # Minimum principal strain

        # Principal angle
        theta = 0.5 * np.arctan2(2 * exy, exx - eyy)

        return e1, e2, theta

    @staticmethod
    def calculate_von_mises(
        exx: NDArray[np.float64],
        exy: NDArray[np.float64],
        eyy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Calculate von Mises equivalent strain.

        Args:
            exx: Strain component xx
            exy: Strain component xy
            eyy: Strain component yy

        Returns:
            Von Mises equivalent strain
        """
        return np.sqrt(2.0 / 3.0) * np.sqrt(
            exx**2 + eyy**2 - exx * eyy + 3.0 * exy**2
        )


def calculate_strains(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    roi: NcorrROI,
    strain_radius: int = 5,
    spacing: int = 1,
    strain_type: str = "green_lagrange",
) -> StrainResult:
    """
    Convenience function to calculate strains.

    Args:
        u: U displacement field
        v: V displacement field
        roi: Region of interest
        strain_radius: Radius for gradient calculation
        spacing: Pixel spacing
        strain_type: 'green_lagrange' or 'eulerian_almansi'

    Returns:
        StrainResult
    """
    calculator = StrainCalculator(strain_radius)

    if strain_type == "green_lagrange":
        return calculator.calculate_green_lagrange(u, v, roi, spacing)
    elif strain_type == "eulerian_almansi":
        return calculator.calculate_eulerian_almansi(u, v, roi, spacing)
    else:
        raise ValueError(f"Unknown strain type: {strain_type}")
