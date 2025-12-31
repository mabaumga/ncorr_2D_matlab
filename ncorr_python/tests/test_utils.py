"""Tests for utility functions."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from ncorr.utils.validation import is_real_bounded, is_int_bounded, validate_dic_parameters
from ncorr.utils.image_loader import (
    load_images, validate_image_format, get_image_info, images_compatible
)
from ncorr.utils.colormaps import get_ncorr_colormap, apply_colormap, overlay_data_on_image


class TestValidation:
    """Tests for validation utilities."""

    def test_is_real_bounded_valid(self):
        """Test valid real number."""
        valid, value, msg = is_real_bounded(0.5, 0, 1)

        assert valid
        assert value == 0.5
        assert msg == ""

    def test_is_real_bounded_string(self):
        """Test parsing string."""
        valid, value, msg = is_real_bounded("0.75", 0, 1)

        assert valid
        assert value == 0.75

    def test_is_real_bounded_invalid_string(self):
        """Test invalid string."""
        valid, value, msg = is_real_bounded("abc", 0, 1)

        assert not valid
        assert value is None

    def test_is_real_bounded_out_of_range(self):
        """Test out of range value."""
        valid, value, msg = is_real_bounded(1.5, 0, 1)

        assert not valid
        assert "1.5 > 1" in msg

    def test_is_real_bounded_exclusive(self):
        """Test exclusive bounds."""
        # Exactly at lower bound (inclusive)
        valid, _, _ = is_real_bounded(0, 0, 1, include_lower=True)
        assert valid

        # Exactly at lower bound (exclusive)
        valid, _, _ = is_real_bounded(0, 0, 1, include_lower=False)
        assert not valid

    def test_is_int_bounded_valid(self):
        """Test valid integer."""
        valid, value, msg = is_int_bounded(50, 0, 100)

        assert valid
        assert value == 50

    def test_is_int_bounded_float_to_int(self):
        """Test float conversion to int."""
        valid, value, msg = is_int_bounded(50.0, 0, 100)

        assert valid
        assert value == 50
        assert isinstance(value, int)

    def test_is_int_bounded_non_integer_float(self):
        """Test non-integer float fails."""
        valid, value, msg = is_int_bounded(50.5, 0, 100)

        assert not valid

    def test_validate_dic_parameters_valid(self):
        """Test valid DIC parameters."""
        params = {
            "radius": 30,
            "spacing": 5,
            "cutoff_diffnorm": 1e-3,
            "cutoff_iteration": 20,
            "total_threads": 4,
        }

        valid, msg = validate_dic_parameters(params)

        assert valid
        assert msg == ""

    def test_validate_dic_parameters_invalid(self):
        """Test invalid DIC parameters."""
        params = {"radius": 5}  # Too small

        valid, msg = validate_dic_parameters(params)

        assert not valid
        assert "radius" in msg.lower()


class TestImageLoader:
    """Tests for image loading utilities."""

    def test_validate_image_format_grayscale(self, sample_grayscale_image):
        """Test validating grayscale image."""
        valid, msg = validate_image_format(sample_grayscale_image)

        assert valid
        assert msg == ""

    def test_validate_image_format_rgb(self, sample_rgb_image):
        """Test validating RGB image."""
        valid, msg = validate_image_format(sample_rgb_image)

        assert valid

    def test_validate_image_format_4channel(self):
        """Test rejecting 4-channel image."""
        img = np.random.rand(100, 100, 4).astype(np.float32)

        valid, msg = validate_image_format(img)

        assert not valid
        assert "4 channels" in msg

    def test_validate_image_format_small(self):
        """Test rejecting small image."""
        img = np.random.rand(5, 5).astype(np.uint8)

        valid, msg = validate_image_format(img)

        assert not valid
        assert "too small" in msg.lower()

    def test_load_images_single(self, temp_image_file):
        """Test loading single image."""
        images = load_images(temp_image_file)

        assert len(images) == 1
        assert images[0].is_set

    def test_load_images_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_images("/nonexistent/image.png")

    def test_get_image_info(self, temp_image_file):
        """Test getting image info."""
        info = get_image_info(temp_image_file)

        assert "width" in info
        assert "height" in info
        assert info["width"] == 100
        assert info["height"] == 100

    def test_images_compatible(self, sample_grayscale_image):
        """Test checking image compatibility."""
        from ncorr.core.image import NcorrImage

        img1 = NcorrImage.from_array(sample_grayscale_image)
        img2 = NcorrImage.from_array(sample_grayscale_image.copy())

        compatible, msg = images_compatible(img1, img2)

        assert compatible

    def test_images_incompatible_size(self, sample_grayscale_image):
        """Test detecting incompatible sizes."""
        from ncorr.core.image import NcorrImage

        img1 = NcorrImage.from_array(sample_grayscale_image)
        img2 = NcorrImage.from_array(sample_grayscale_image[::2, ::2])  # Half size

        compatible, msg = images_compatible(img1, img2)

        assert not compatible
        assert "mismatch" in msg.lower()


class TestColormaps:
    """Tests for colormap utilities."""

    def test_get_ncorr_colormap_default(self):
        """Test getting default colormap."""
        cmap = get_ncorr_colormap("jet")

        assert cmap is not None

    def test_get_ncorr_colormap_custom(self):
        """Test getting custom ncorr colormap."""
        cmap = get_ncorr_colormap("ncorr")

        assert cmap is not None
        assert cmap.name == "ncorr"

    def test_get_ncorr_colormap_strain(self):
        """Test getting strain colormap."""
        cmap = get_ncorr_colormap("strain")

        assert cmap is not None

    def test_apply_colormap(self):
        """Test applying colormap to data."""
        data = np.random.rand(50, 50)

        rgba = apply_colormap(data, vmin=0, vmax=1)

        assert rgba.shape == (50, 50, 4)
        assert rgba.dtype == np.uint8

    def test_apply_colormap_nan(self):
        """Test colormap with NaN values."""
        data = np.random.rand(50, 50)
        data[20:30, 20:30] = np.nan

        rgba = apply_colormap(data, nan_color=(0.5, 0.5, 0.5, 1.0))

        # Check NaN region is gray
        assert rgba[25, 25, 0] == 127  # ~0.5 * 255

    def test_overlay_data_on_image(self, sample_grayscale_image):
        """Test overlaying data on image."""
        data = np.random.rand(100, 100)
        roi = np.ones((100, 100), dtype=np.bool_)
        roi[0:10, :] = False  # Exclude top rows

        result = overlay_data_on_image(
            sample_grayscale_image,
            data,
            roi,
            alpha=0.5
        )

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
