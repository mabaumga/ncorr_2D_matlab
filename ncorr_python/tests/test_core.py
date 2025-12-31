"""Tests for core data classes."""

import numpy as np
import pytest

from ncorr.core.status import Status
from ncorr.core.image import NcorrImage, ImageType
from ncorr.core.roi import NcorrROI, Region, Boundary
from ncorr.core.dic_parameters import DICParameters, StepAnalysis


class TestStatus:
    """Tests for Status enumeration."""

    def test_status_values(self):
        """Test status enum values."""
        assert Status.SUCCESS == 1
        assert Status.FAILED == 0
        assert Status.CANCELLED == -1

    def test_status_methods(self):
        """Test status class methods."""
        assert Status.success() == Status.SUCCESS
        assert Status.failed() == Status.FAILED
        assert Status.cancelled() == Status.CANCELLED

    def test_status_comparison(self):
        """Test status comparison."""
        assert Status.SUCCESS != Status.FAILED
        assert Status.SUCCESS == Status.success()


class TestNcorrImage:
    """Tests for NcorrImage class."""

    def test_init_empty(self):
        """Test empty initialization."""
        img = NcorrImage()
        assert not img.is_set
        assert img.height == 0
        assert img.width == 0

    def test_set_grayscale_image(self, sample_grayscale_image):
        """Test setting grayscale image."""
        img = NcorrImage()
        img.set_image(
            ImageType.LOAD,
            {"img": sample_grayscale_image, "name": "test", "path": ""}
        )

        assert img.is_set
        assert img.height == 100
        assert img.width == 100
        assert img.img_type == ImageType.LOAD

    def test_set_rgb_image(self, sample_rgb_image):
        """Test setting RGB image."""
        img = NcorrImage()
        img.set_image(
            ImageType.LOAD,
            {"img": sample_rgb_image, "name": "test", "path": ""}
        )

        assert img.is_set
        assert img.height == 100
        assert img.width == 100

        # Check grayscale conversion
        gs = img.get_gs()
        assert gs.shape == (100, 100)
        assert gs.dtype == np.float64
        assert 0 <= gs.min() <= gs.max() <= 1

    def test_set_16bit_image(self, sample_16bit_image):
        """Test setting 16-bit image."""
        img = NcorrImage()
        img.set_image(
            ImageType.LOAD,
            {"img": sample_16bit_image, "name": "test", "path": ""}
        )

        assert img.is_set
        gs = img.get_gs()
        assert 0 <= gs.min() <= gs.max() <= 1

    def test_get_img(self, sample_grayscale_image):
        """Test getting image data."""
        img = NcorrImage()
        img.set_image(
            ImageType.LOAD,
            {"img": sample_grayscale_image, "name": "test", "path": ""}
        )

        result = img.get_img()
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_get_gs(self, sample_grayscale_image):
        """Test getting grayscale data."""
        img = NcorrImage()
        img.set_image(
            ImageType.LOAD,
            {"img": sample_grayscale_image, "name": "test", "path": ""}
        )

        gs = img.get_gs()
        assert gs.shape == (100, 100)
        assert gs.dtype == np.float64

    def test_get_bcoef(self, sample_grayscale_image):
        """Test getting B-spline coefficients."""
        img = NcorrImage()
        img.set_image(
            ImageType.LOAD,
            {"img": sample_grayscale_image, "name": "test", "path": ""}
        )

        bcoef = img.get_bcoef()
        expected_shape = (100 + 2 * 20, 100 + 2 * 20)  # border_bcoef = 20
        assert bcoef.shape == expected_shape
        assert bcoef.dtype == np.float64

    def test_reduce(self, sample_grayscale_image):
        """Test image reduction."""
        img = NcorrImage()
        img.set_image(
            ImageType.LOAD,
            {"img": sample_grayscale_image, "name": "test", "path": ""}
        )

        # Reduce by factor of 2 (spacing=1)
        reduced = img.reduce(spacing=1)
        assert reduced.is_set
        assert reduced.height == 50
        assert reduced.width == 50

    def test_from_file(self, temp_image_file):
        """Test loading from file."""
        img = NcorrImage.from_file(temp_image_file)

        assert img.is_set
        assert img.height == 100
        assert img.width == 100
        assert img.name == temp_image_file.name

    def test_from_array(self, sample_grayscale_image):
        """Test creating from array."""
        img = NcorrImage.from_array(sample_grayscale_image, name="test_array")

        assert img.is_set
        assert img.height == 100
        assert img.width == 100
        assert img.name == "test_array"

    def test_form_bcoef(self):
        """Test B-spline coefficient calculation."""
        # Create test data
        data = np.random.rand(20, 20).astype(np.float64)

        bcoef = NcorrImage.form_bcoef(data)

        assert bcoef.shape == data.shape
        assert bcoef.dtype == np.float64

    def test_form_bcoef_small_array(self):
        """Test B-spline with too small array."""
        data = np.random.rand(3, 3).astype(np.float64)

        with pytest.raises(ValueError):
            NcorrImage.form_bcoef(data)

    def test_formatted(self, sample_grayscale_image):
        """Test formatted output."""
        img = NcorrImage()
        img.set_image(
            ImageType.LOAD,
            {"img": sample_grayscale_image, "name": "test", "path": ""}
        )

        formatted = img.formatted()

        assert isinstance(formatted, dict)
        assert "img" in formatted
        assert "gs" in formatted
        assert "bcoef" in formatted
        assert formatted["height"] == 100
        assert formatted["width"] == 100


class TestNcorrROI:
    """Tests for NcorrROI class."""

    def test_init_empty(self):
        """Test empty initialization."""
        roi = NcorrROI()
        assert not roi.is_set
        assert len(roi.regions) == 0

    def test_set_from_mask(self, sample_circular_mask):
        """Test setting ROI from mask."""
        roi = NcorrROI()
        roi.set_roi("load", {"mask": sample_circular_mask, "cutoff": 10})

        assert roi.is_set
        assert len(roi.regions) >= 1
        assert np.any(roi.mask)

    def test_set_from_rectangular_mask(self, sample_rectangular_mask):
        """Test setting ROI from rectangular mask."""
        roi = NcorrROI()
        roi.set_roi("load", {"mask": sample_rectangular_mask, "cutoff": 10})

        assert roi.is_set
        assert len(roi.regions) == 1

        # Check region bounds
        region = roi.regions[0]
        assert region.leftbound == 20
        assert region.rightbound == 79
        assert region.upperbound == 20
        assert region.lowerbound == 79

    def test_reduce(self, sample_rectangular_mask):
        """Test ROI reduction."""
        roi = NcorrROI()
        roi.set_roi("load", {"mask": sample_rectangular_mask, "cutoff": 10})

        reduced = roi.reduce(spacing=1)

        assert reduced.is_set
        assert reduced.mask.shape[0] == 50
        assert reduced.mask.shape[1] == 50

    def test_get_num_region(self, sample_rectangular_mask):
        """Test finding region containing point."""
        roi = NcorrROI()
        roi.set_roi("load", {"mask": sample_rectangular_mask, "cutoff": 10})

        # Point inside ROI
        region_idx, node_idx = roi.get_num_region(50, 50)
        assert region_idx == 0

        # Point outside ROI
        region_idx, node_idx = roi.get_num_region(5, 5)
        assert region_idx == -1

    def test_get_region_mask(self, sample_rectangular_mask):
        """Test getting mask for specific region."""
        roi = NcorrROI()
        roi.set_roi("load", {"mask": sample_rectangular_mask, "cutoff": 10})

        region_mask = roi.get_region_mask(0)

        assert region_mask.shape == sample_rectangular_mask.shape
        assert np.array_equal(region_mask, sample_rectangular_mask)

    def test_get_full_regions_count(self, sample_rectangular_mask):
        """Test counting non-empty regions."""
        roi = NcorrROI()
        roi.set_roi("load", {"mask": sample_rectangular_mask, "cutoff": 10})

        count = roi.get_full_regions_count()
        assert count == 1


class TestRegion:
    """Tests for Region class."""

    def test_empty_placeholder(self):
        """Test creating empty placeholder."""
        region = Region.empty_placeholder()

        assert region.is_empty()
        assert region.totalpoints == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        region = Region(
            nodelist=np.array([[0, 10], [5, 15]], dtype=np.int32),
            noderange=np.array([2, 2], dtype=np.int32),
            leftbound=0,
            rightbound=1,
            upperbound=0,
            lowerbound=15,
            totalpoints=22,
        )

        d = region.to_dict()

        assert "nodelist" in d
        assert "noderange" in d
        assert d["totalpoints"] == 22

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "nodelist": np.array([[0, 10]], dtype=np.int32),
            "noderange": np.array([2], dtype=np.int32),
            "leftbound": 0,
            "rightbound": 0,
            "upperbound": 0,
            "lowerbound": 10,
            "totalpoints": 11,
        }

        region = Region.from_dict(d)

        assert region.totalpoints == 11
        assert not region.is_empty()


class TestDICParameters:
    """Tests for DICParameters class."""

    def test_default_values(self):
        """Test default parameter values."""
        params = DICParameters()

        assert params.radius == 30
        assert params.spacing == 5
        assert params.cutoff_diffnorm == 1e-2
        assert params.cutoff_iteration == 20
        assert params.total_threads == 1

    def test_validate_valid(self):
        """Test validation with valid parameters."""
        params = DICParameters(
            radius=50,
            spacing=10,
            cutoff_diffnorm=1e-3,
            cutoff_iteration=30,
        )

        assert params.validate()

    def test_validate_invalid_radius(self):
        """Test validation with invalid radius."""
        params = DICParameters(radius=5)  # Too small

        with pytest.raises(ValueError):
            params.validate()

    def test_validate_invalid_spacing(self):
        """Test validation with invalid spacing."""
        params = DICParameters(spacing=100)  # Too large

        with pytest.raises(ValueError):
            params.validate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = DICParameters()
        d = params.to_dict()

        assert d["radius"] == 30
        assert d["spacing"] == 5
        assert "step_analysis" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "radius": 40,
            "spacing": 8,
            "cutoff_diffnorm": 1e-4,
            "cutoff_iteration": 25,
        }

        params = DICParameters.from_dict(d)

        assert params.radius == 40
        assert params.spacing == 8
