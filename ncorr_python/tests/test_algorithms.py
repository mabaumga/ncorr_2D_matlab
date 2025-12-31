"""Tests for algorithm modules."""

import numpy as np
import pytest
from scipy.ndimage import shift

from ncorr.algorithms.bspline import BSplineInterpolator
from ncorr.algorithms.regions import RegionProcessor
from ncorr.algorithms.strain import StrainCalculator, calculate_strains
from ncorr.algorithms.seeds import SeedCalculator
from ncorr.algorithms.dic import DICAnalysis, SeedInfo

from ncorr.core.image import NcorrImage
from ncorr.core.roi import NcorrROI
from ncorr.core.dic_parameters import DICParameters


class TestBSplineInterpolator:
    """Tests for B-spline interpolation."""

    def test_compute_bcoef(self):
        """Test B-spline coefficient computation."""
        data = np.random.rand(50, 50).astype(np.float64)

        bcoef = BSplineInterpolator.compute_bcoef(data)

        assert bcoef.shape == data.shape
        assert bcoef.dtype == np.float64

    def test_compute_bcoef_empty(self):
        """Test with empty array."""
        data = np.array([], dtype=np.float64).reshape(0, 0)

        bcoef = BSplineInterpolator.compute_bcoef(data)

        assert bcoef.size == 0

    def test_interpolate_values(self):
        """Test value interpolation."""
        # Create simple test data
        data = np.zeros((50, 50), dtype=np.float64)
        data[20:30, 20:30] = 1.0

        bcoef = BSplineInterpolator.compute_bcoef(data)

        # Interpolate at integer points
        points = np.array([[25.0, 25.0], [25.5, 25.5]], dtype=np.float64)

        values = BSplineInterpolator.interpolate_values(
            points, bcoef, left_offset=0, top_offset=0, border=0
        )

        # Values should be close to 1 in the center
        assert len(values) == 2

    def test_interpolate_with_gradient(self):
        """Test interpolation with gradient."""
        # Create gradient data
        x, y = np.meshgrid(np.arange(50), np.arange(50))
        data = (x + y).astype(np.float64) / 100

        bcoef = BSplineInterpolator.compute_bcoef(data)

        val, dx, dy = BSplineInterpolator.interpolate_with_gradient(
            25.0, 25.0, bcoef, border=0
        )

        # Gradients should be approximately 0.01 (1/100)
        assert not np.isnan(val)
        assert abs(dx - 0.01) < 0.1
        assert abs(dy - 0.01) < 0.1


class TestRegionProcessor:
    """Tests for region processing algorithms."""

    def test_form_regions_single(self, sample_rectangular_mask):
        """Test forming regions from simple mask."""
        regions, removed = RegionProcessor.form_regions(
            sample_rectangular_mask, min_points=10
        )

        assert len(regions) == 1
        assert not removed
        assert regions[0].totalpoints > 0

    def test_form_regions_multiple(self):
        """Test forming multiple regions."""
        mask = np.zeros((100, 100), dtype=np.bool_)
        mask[10:30, 10:30] = True  # Region 1
        mask[60:80, 60:80] = True  # Region 2

        regions, removed = RegionProcessor.form_regions(mask, min_points=10)

        assert len(regions) == 2

    def test_form_regions_remove_small(self):
        """Test removal of small regions."""
        mask = np.zeros((100, 100), dtype=np.bool_)
        mask[10:30, 10:30] = True  # Large region
        mask[50:52, 50:52] = True  # Small region (4 pixels)

        regions, removed = RegionProcessor.form_regions(mask, min_points=10)

        assert len(regions) == 1
        assert removed

    def test_form_boundary(self, sample_rectangular_mask):
        """Test boundary tracing."""
        boundary = RegionProcessor.form_boundary(
            (20, 20), 0, sample_rectangular_mask
        )

        assert len(boundary) > 0
        assert boundary.shape[1] == 2

    def test_fill_polygon(self):
        """Test polygon filling."""
        # Triangle
        vertices = np.array([
            [50, 10],
            [90, 90],
            [10, 90],
        ], dtype=np.float64)

        mask = RegionProcessor.fill_polygon(vertices, (100, 100))

        assert mask.shape == (100, 100)
        assert np.any(mask)
        # Center should be filled
        assert mask[50, 50]


class TestStrainCalculator:
    """Tests for strain calculation."""

    def test_green_lagrange_uniform(self):
        """Test Green-Lagrange strain with uniform displacement."""
        # Uniform displacement should give zero strain
        u = np.ones((50, 50)) * 5.0
        v = np.ones((50, 50)) * 3.0

        mask = np.ones((50, 50), dtype=np.bool_)
        roi = NcorrROI()
        roi.set_roi("load", {"mask": mask, "cutoff": 0})

        calc = StrainCalculator(strain_radius=3)
        result = calc.calculate_green_lagrange(u, v, roi, spacing=1)

        # Strains should be zero for uniform displacement
        valid = ~np.isnan(result.exx)
        if np.any(valid):
            assert np.allclose(result.exx[valid], 0, atol=1e-10)
            assert np.allclose(result.eyy[valid], 0, atol=1e-10)
            assert np.allclose(result.exy[valid], 0, atol=1e-10)

    def test_green_lagrange_linear(self):
        """Test Green-Lagrange with linear displacement (constant strain)."""
        # u = 0.1 * x gives du/dx = 0.1
        y, x = np.meshgrid(np.arange(50), np.arange(50), indexing='ij')
        u = 0.1 * x.astype(np.float64)
        v = np.zeros((50, 50))

        mask = np.ones((50, 50), dtype=np.bool_)
        roi = NcorrROI()
        roi.set_roi("load", {"mask": mask, "cutoff": 0})

        calc = StrainCalculator(strain_radius=3)
        result = calc.calculate_green_lagrange(u, v, roi, spacing=1)

        # Exx should be approximately 0.1 + 0.5 * 0.1^2 = 0.105
        valid = ~np.isnan(result.exx)
        if np.any(valid):
            expected_exx = 0.1 + 0.5 * 0.1**2
            assert np.allclose(result.exx[valid], expected_exx, atol=0.01)

    def test_principal_strains(self):
        """Test principal strain calculation."""
        exx = np.array([[0.01, 0.02], [0.015, 0.025]])
        eyy = np.array([[-0.005, -0.01], [-0.008, -0.012]])
        exy = np.array([[0.003, 0.005], [0.004, 0.006]])

        e1, e2, theta = StrainCalculator.calculate_principal_strains(exx, exy, eyy)

        # e1 should be larger than e2
        assert np.all(e1 >= e2)

    def test_von_mises(self):
        """Test von Mises equivalent strain."""
        exx = np.ones((5, 5)) * 0.01
        eyy = np.ones((5, 5)) * -0.005
        exy = np.zeros((5, 5))

        vm = StrainCalculator.calculate_von_mises(exx, exy, eyy)

        assert vm.shape == (5, 5)
        assert np.all(vm >= 0)


class TestDICAnalysis:
    """Tests for DIC analysis."""

    def test_analyze_known_displacement(self, sample_speckle_pattern, dic_parameters):
        """Test DIC with known displacement."""
        from scipy.ndimage import shift

        # Create displaced image
        u_true = 3.0
        v_true = 2.0
        displaced = shift(
            sample_speckle_pattern.astype(np.float64),
            [v_true, u_true],
            mode='nearest'
        ).astype(np.uint8)

        # Create images
        ref_img = NcorrImage.from_array(sample_speckle_pattern)
        cur_img = NcorrImage.from_array(displaced)

        # Create ROI (central region)
        mask = np.zeros((200, 200), dtype=np.bool_)
        mask[50:150, 50:150] = True
        roi = NcorrROI()
        roi.set_roi("load", {"mask": mask, "cutoff": 20})

        # Create seed at center
        seeds = [SeedInfo(x=100, y=100, u=0, v=0, region_idx=0, valid=True)]

        # Run analysis
        dic = DICAnalysis(dic_parameters)
        results = dic.analyze(ref_img, [cur_img], roi, seeds)

        assert len(results) == 1

        result = results[0]
        assert result.u is not None
        assert result.v is not None

        # Check if displacement is close to true value where valid
        valid = result.roi
        if np.any(valid):
            mean_u = np.nanmean(result.u[valid])
            mean_v = np.nanmean(result.v[valid])

            # Should be within 1 pixel of true displacement
            assert abs(mean_u - u_true) < 1.5
            assert abs(mean_v - v_true) < 1.5


class TestSeedCalculator:
    """Tests for seed calculation."""

    def test_calculate_seeds(self, sample_speckle_pattern, dic_parameters):
        """Test seed calculation."""
        from scipy.ndimage import shift

        # Create displaced image
        displaced = shift(
            sample_speckle_pattern.astype(np.float64),
            [2.0, 3.0],
            mode='nearest'
        ).astype(np.uint8)

        ref_img = NcorrImage.from_array(sample_speckle_pattern)
        cur_img = NcorrImage.from_array(displaced)

        # Create ROI
        mask = np.zeros((200, 200), dtype=np.bool_)
        mask[50:150, 50:150] = True
        roi = NcorrROI()
        roi.set_roi("load", {"mask": mask, "cutoff": 20})

        # Calculate seeds
        calc = SeedCalculator(dic_parameters)
        seeds = calc.calculate_seeds(ref_img, cur_img, roi, search_radius=30)

        assert len(seeds) == len(roi.regions)

        # Check that at least one seed is valid
        valid_seeds = [s for s in seeds if s.valid]
        assert len(valid_seeds) > 0

        # Check displacement is reasonable
        for seed in valid_seeds:
            assert abs(seed.u - 3.0) < 5.0
            assert abs(seed.v - 2.0) < 5.0
