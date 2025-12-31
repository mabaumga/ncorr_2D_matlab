"""
Strain calculation algorithms.

Computes Green-Lagrange and Eulerian-Almansi strains from displacement fields.
Equivalent to ncorr_alg_dispgrad.cpp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import jit, prange

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

        Where F = I + grad(u) is the deformation gradient.

        Strain components:
            Exx = du/dx + 0.5 * ((du/dx)^2 + (dv/dx)^2)
            Eyy = dv/dy + 0.5 * ((du/dy)^2 + (dv/dy)^2)
            Exy = 0.5 * (du/dy + dv/dx) + du/dx*du/dy + dv/dx*dv/dy

        Args:
            u: U displacement field
            v: V displacement field
            roi: Region of interest
            spacing: Pixel spacing for gradient calculation

        Returns:
            StrainResult with Green-Lagrange strains
        """
        # Calculate displacement gradients
        dudx, dudy, dvdx, dvdy, grad_roi = self._calculate_gradients(
            u, v, roi.mask, spacing
        )

        # Initialize strain arrays
        exx = np.full_like(u, np.nan)
        exy = np.full_like(u, np.nan)
        eyy = np.full_like(u, np.nan)

        # Calculate Green-Lagrange strains where gradients are valid
        valid = grad_roi

        # Exx = du/dx + 0.5 * ((du/dx)^2 + (dv/dx)^2)
        exx[valid] = dudx[valid] + 0.5 * (dudx[valid]**2 + dvdx[valid]**2)

        # Eyy = dv/dy + 0.5 * ((du/dy)^2 + (dv/dy)^2)
        eyy[valid] = dvdy[valid] + 0.5 * (dudy[valid]**2 + dvdy[valid]**2)

        # Exy = 0.5 * (du/dy + dv/dx + du/dx*du/dy + dv/dx*dv/dy)
        exy[valid] = 0.5 * (dudy[valid] + dvdx[valid] +
                           dudx[valid] * dudy[valid] +
                           dvdx[valid] * dvdy[valid])

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

        Strain components:
            exx = du/dx - 0.5 * ((du/dx)^2 + (dv/dx)^2)
            eyy = dv/dy - 0.5 * ((du/dy)^2 + (dv/dy)^2)
            exy = 0.5 * (du/dy + dv/dx) - du/dx*du/dy - dv/dx*dv/dy

        Args:
            u: U displacement field
            v: V displacement field
            roi: Region of interest
            spacing: Pixel spacing

        Returns:
            StrainResult with Eulerian-Almansi strains
        """
        # Calculate displacement gradients
        dudx, dudy, dvdx, dvdy, grad_roi = self._calculate_gradients(
            u, v, roi.mask, spacing
        )

        # Initialize strain arrays
        exx = np.full_like(u, np.nan)
        exy = np.full_like(u, np.nan)
        eyy = np.full_like(u, np.nan)

        valid = grad_roi

        # exx = du/dx - 0.5 * ((du/dx)^2 + (dv/dx)^2)
        exx[valid] = dudx[valid] - 0.5 * (dudx[valid]**2 + dvdx[valid]**2)

        # eyy = dv/dy - 0.5 * ((du/dy)^2 + (dv/dy)^2)
        eyy[valid] = dvdy[valid] - 0.5 * (dudy[valid]**2 + dvdy[valid]**2)

        # exy = 0.5 * (du/dy + dv/dx - du/dx*du/dy - dv/dx*dv/dy)
        exy[valid] = 0.5 * (dudy[valid] + dvdx[valid] -
                           dudx[valid] * dudy[valid] -
                           dvdx[valid] * dvdy[valid])

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

    def _calculate_gradients(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        mask: NDArray[np.bool_],
        spacing: int,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Calculate displacement gradients using finite differences.

        Uses central differences with configurable radius.

        Args:
            u: U displacement field
            v: V displacement field
            mask: Valid data mask
            spacing: Pixel spacing

        Returns:
            Tuple of (dudx, dudy, dvdx, dvdy, valid_mask)
        """
        h, w = u.shape
        r = self.strain_radius

        dudx = np.full((h, w), np.nan)
        dudy = np.full((h, w), np.nan)
        dvdx = np.full((h, w), np.nan)
        dvdy = np.full((h, w), np.nan)
        valid = np.zeros((h, w), dtype=np.bool_)

        # Reduce mask by strain radius
        reduced_mask = mask.copy()

        # Erode mask by radius to ensure all gradient calculations are valid
        from scipy.ndimage import binary_erosion
        struct = np.ones((2 * r + 1, 2 * r + 1))
        reduced_mask = binary_erosion(mask, struct)

        # Calculate gradients using central differences
        for j in range(r, h - r):
            for i in range(r, w - r):
                if not reduced_mask[j, i]:
                    continue

                if np.isnan(u[j, i]) or np.isnan(v[j, i]):
                    continue

                # Check if all points in radius are valid
                all_valid = True
                for dj in range(-r, r + 1):
                    for di in range(-r, r + 1):
                        if np.isnan(u[j + dj, i + di]) or np.isnan(v[j + dj, i + di]):
                            all_valid = False
                            break
                    if not all_valid:
                        break

                if not all_valid:
                    continue

                # Central differences
                # du/dx = (u(x+r) - u(x-r)) / (2*r*spacing)
                dudx[j, i] = (u[j, i + r] - u[j, i - r]) / (2.0 * r * spacing)
                dudy[j, i] = (u[j + r, i] - u[j - r, i]) / (2.0 * r * spacing)
                dvdx[j, i] = (v[j, i + r] - v[j, i - r]) / (2.0 * r * spacing)
                dvdy[j, i] = (v[j + r, i] - v[j - r, i]) / (2.0 * r * spacing)

                valid[j, i] = True

        return dudx, dudy, dvdx, dvdy, valid

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
        # Principal strains from eigenvalue decomposition
        # E = [[exx, exy], [exy, eyy]]
        # Eigenvalues: e1, e2 = (exx + eyy)/2 +/- sqrt(((exx-eyy)/2)^2 + exy^2)

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
        # von Mises for plane strain:
        # e_vm = sqrt(2/3) * sqrt(exx^2 + eyy^2 - exx*eyy + 3*exy^2)

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
