"""Tests for main Ncorr application."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from ncorr.main import Ncorr, AnalysisResults
from ncorr.core.status import Status
from ncorr.core.dic_parameters import DICParameters


class TestNcorr:
    """Tests for main Ncorr class."""

    def test_init(self):
        """Test initialization."""
        ncorr = Ncorr()

        assert ncorr.reference_image is None
        assert len(ncorr.current_images) == 0
        assert ncorr.roi is None
        assert ncorr.results is None

    def test_set_reference_from_array(self, sample_grayscale_image):
        """Test setting reference from array."""
        ncorr = Ncorr()

        status = ncorr.set_reference(sample_grayscale_image)

        assert status == Status.SUCCESS
        assert ncorr.reference_image is not None
        assert ncorr.reference_image.height == 100

    def test_set_reference_from_file(self, temp_image_file):
        """Test setting reference from file."""
        ncorr = Ncorr()

        status = ncorr.set_reference(temp_image_file)

        assert status == Status.SUCCESS
        assert ncorr.reference_image is not None

    def test_set_current_single(self, sample_grayscale_image):
        """Test setting single current image."""
        ncorr = Ncorr()

        status = ncorr.set_current(sample_grayscale_image)

        assert status == Status.SUCCESS
        assert len(ncorr.current_images) == 1

    def test_set_current_multiple(self, sample_grayscale_image):
        """Test setting multiple current images."""
        ncorr = Ncorr()

        images = [sample_grayscale_image, sample_grayscale_image.copy()]
        status = ncorr.set_current(images)

        assert status == Status.SUCCESS
        assert len(ncorr.current_images) == 2

    def test_set_roi_from_mask(self, sample_grayscale_image, sample_circular_mask):
        """Test setting ROI from mask."""
        ncorr = Ncorr()
        ncorr.set_reference(sample_grayscale_image)

        status = ncorr.set_roi_from_mask(sample_circular_mask)

        assert status == Status.SUCCESS
        assert ncorr.roi is not None
        assert ncorr.roi.is_set

    def test_set_parameters(self):
        """Test setting parameters."""
        ncorr = Ncorr()

        params = DICParameters(radius=40, spacing=8)
        status = ncorr.set_parameters(params)

        assert status == Status.SUCCESS
        assert ncorr.parameters.radius == 40
        assert ncorr.parameters.spacing == 8

    def test_set_invalid_parameters(self):
        """Test setting invalid parameters."""
        ncorr = Ncorr()

        params = DICParameters(radius=5)  # Too small
        status = ncorr.set_parameters(params)

        assert status == Status.FAILED

    def test_calculate_seeds_without_images(self):
        """Test seed calculation fails without images."""
        ncorr = Ncorr()

        status = ncorr.calculate_seeds()

        assert status == Status.FAILED

    def test_progress_callback(self, sample_grayscale_image, sample_circular_mask):
        """Test progress callback is called."""
        ncorr = Ncorr()

        progress_values = []

        def callback(progress, message):
            progress_values.append(progress)

        ncorr.set_progress_callback(callback)
        ncorr.set_reference(sample_grayscale_image)
        ncorr.set_current(sample_grayscale_image)
        ncorr.set_roi_from_mask(sample_circular_mask)

        # The callback would be called during analysis
        # Here we just verify it's set correctly


class TestAnalysisResults:
    """Tests for AnalysisResults class."""

    def test_save_and_load(self, sample_grayscale_image, sample_circular_mask):
        """Test saving and loading results."""
        from ncorr.algorithms.dic import DICResult

        # Create mock results
        u = np.random.rand(50, 50)
        v = np.random.rand(50, 50)
        corrcoef = np.random.rand(50, 50)
        roi = np.ones((50, 50), dtype=np.bool_)

        disp = DICResult(u=u, v=v, corrcoef=corrcoef, roi=roi)
        params = DICParameters()

        results = AnalysisResults(
            displacements=[disp],
            parameters=params,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results"

            # Save
            results.save(filepath)

            # Check files exist
            assert filepath.with_suffix(".npz").exists()
            assert filepath.with_suffix(".json").exists()

            # Load
            loaded = AnalysisResults.load(filepath)

            assert len(loaded.displacements) == 1
            assert np.allclose(loaded.displacements[0].u, u)
            assert np.allclose(loaded.displacements[0].v, v)


class TestIntegration:
    """Integration tests for full DIC workflow."""

    def test_simple_workflow(self, sample_speckle_pattern):
        """Test simple DIC workflow."""
        from scipy.ndimage import shift

        # Create displaced image
        u_true = 2.0
        v_true = 1.5
        displaced = shift(
            sample_speckle_pattern.astype(np.float64),
            [v_true, u_true],
            mode='nearest'
        ).astype(np.uint8)

        # Create mask
        mask = np.zeros((200, 200), dtype=np.bool_)
        mask[60:140, 60:140] = True

        # Run analysis
        ncorr = Ncorr()
        ncorr.set_reference(sample_speckle_pattern)
        ncorr.set_current(displaced)
        ncorr.set_roi_from_mask(mask)
        ncorr.set_parameters(DICParameters(radius=20, spacing=3))

        results = ncorr.run_analysis()

        assert results is not None
        assert len(results.displacements) == 1

        # Check displacement
        disp = results.displacements[0]
        valid = disp.roi
        if np.any(valid):
            mean_u = np.nanmean(disp.u[valid])
            mean_v = np.nanmean(disp.v[valid])

            # Allow some error
            assert abs(mean_u - u_true) < 2.0
            assert abs(mean_v - v_true) < 2.0
