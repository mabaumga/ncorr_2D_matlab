"""
Image class for Ncorr.

Equivalent to MATLAB's ncorr_class_img.m
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage
from scipy.fft import fft, ifft


class ImageType(str, Enum):
    """Types of image loading."""

    FILE = "file"       # Full image loaded through image file
    LOAD = "load"       # Full image loaded manually through workspace
    LAZY = "lazy"       # Full image that loads on demand
    REDUCED = "reduced"  # Reduced image for display purposes


@dataclass
class NcorrImage:
    """
    Image class for Ncorr DIC analysis.

    This class represents an image with metadata and supports various loading
    types including lazy loading for memory efficiency.

    Attributes:
        img: RGB image data (uint8, shape: H x W x 3)
        gs: Grayscale values for computation (float64, shape: H x W)
        bcoef: B-spline coefficients for interpolation
        img_type: How the image was loaded
        height: Image height in pixels
        width: Image width in pixels
        max_gs: Maximum grayscale value
        min_gs: Minimum grayscale value
        border_bcoef: Border padding for B-spline coefficients
        name: Image filename
        path: Image file path
    """

    # Private properties (lazily computed)
    _img: Optional[NDArray[np.uint8]] = field(default=None, repr=False)
    _gs: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _bcoef: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    # Public properties
    img_type: ImageType = field(default=ImageType.FILE)
    height: int = 0
    width: int = 0
    max_gs: float = 1.0
    min_gs: float = 0.0
    border_bcoef: int = 20  # Must be >= 2
    name: str = ""
    path: str = ""

    def __post_init__(self):
        """Initialize the image after dataclass creation."""
        self._is_set = False

    @property
    def is_set(self) -> bool:
        """Check if the image has been set."""
        return self._is_set

    def set_image(
        self,
        img_type: Union[str, ImageType],
        data: dict,
    ) -> None:
        """
        Set the image and other useful properties.

        Args:
            img_type: How the image was loaded. Supported types:
                - 'load': Full image loaded manually through workspace
                - 'file': Full image loaded through image file
                - 'lazy': Full image that loads on demand
                - 'reduced': Reduced image for display purposes
            data: Dictionary containing image data:
                - For 'load', 'file', 'reduced': {'img': ndarray, 'name': str, 'path': str}
                - For 'lazy': {'name': str, 'path': str}

        Raises:
            ValueError: If img_type is not supported
        """
        if isinstance(img_type, str):
            try:
                img_type = ImageType(img_type)
            except ValueError:
                raise ValueError(f"Incorrect type provided: {img_type}")

        self.img_type = img_type
        self.name = data.get("name", "")
        self.path = data.get("path", "")

        # Set gs and image for 'file', 'load', and 'reduced'
        if img_type in (ImageType.FILE, ImageType.LOAD, ImageType.REDUCED):
            img_data = data["img"]

            if img_data.ndim == 3 and img_data.shape[2] == 3:
                # Image is RGB-color
                self._img = self._to_uint8(img_data)
                # Convert to grayscale for computation
                self._gs = self._rgb_to_grayscale(img_data)
            else:
                # Image is monochrome
                img_mono = self._to_uint8(img_data)
                # Convert to RGB for display
                self._img = np.stack([img_mono, img_mono, img_mono], axis=-1)
                # Grayscale
                self._gs = self._to_double(img_data)

            self.height, self.width = self._gs.shape

        # Calculate B-spline coefficients for 'file' and 'load' only
        if img_type in (ImageType.FILE, ImageType.LOAD):
            self._compute_bcoef()

        # For lazy loading, get dimensions from file info
        if img_type == ImageType.LAZY:
            filepath = Path(self.path) / self.name
            with Image.open(filepath) as pil_img:
                self.width, self.height = pil_img.size

        self.max_gs = 1.0
        self.min_gs = 0.0
        self._is_set = True

    def _compute_bcoef(self) -> None:
        """Compute B-spline coefficients with border padding."""
        border = self.border_bcoef
        h, w = self._gs.shape

        # Create padded array
        bcoef = np.zeros((h + 2 * border, w + 2 * border), dtype=np.float64)

        # Place grayscale data in center
        bcoef[border:border + h, border:border + w] = self._gs

        # Fill corners
        bcoef[:border, :border] = bcoef[border, border]  # top-left
        bcoef[:border, -border:] = bcoef[border, -border - 1]  # top-right
        bcoef[-border:, -border:] = bcoef[-border - 1, -border - 1]  # bottom-right
        bcoef[-border:, :border] = bcoef[-border - 1, border]  # bottom-left

        # Fill sides
        bcoef[:border, border:-border] = bcoef[border, border:-border]  # top
        bcoef[border:-border, -border:] = bcoef[border:-border, -border - 1:- border]  # right
        bcoef[-border:, border:-border] = bcoef[-border - 1, border:-border]  # bottom
        bcoef[border:-border, :border] = bcoef[border:-border, border:border + 1]  # left

        # Compute B-spline coefficients
        self._bcoef = self.form_bcoef(bcoef)

    @staticmethod
    def form_bcoef(data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute biquintic B-spline coefficients of input array.

        This function returns the B-spline coefficients of the input array data.
        The size returned will be the same as the input array.

        Args:
            data: Input array (must be at least 5x5 or empty)

        Returns:
            Biquintic B-spline coefficients

        Raises:
            ValueError: If array is smaller than 5x5 and not empty
        """
        if data.size == 0:
            return np.zeros_like(data)

        if data.shape[0] < 5 or data.shape[1] < 5:
            raise ValueError(
                "Array for obtaining B-spline coefficients must be >= 5x5 or empty"
            )

        # B-spline kernel for biquintic splines
        kernel_b = np.array([1/120, 13/60, 11/20, 13/60, 1/120])

        h, w = data.shape
        result = np.zeros_like(data)

        # FFT across rows (x-direction)
        kernel_b_x = np.zeros(w, dtype=np.complex128)
        kernel_b_x[:3] = kernel_b[2:]  # [11/20, 13/60, 1/120]
        kernel_b_x[-2:] = kernel_b[:2]  # [1/120, 13/60]
        kernel_b_x = fft(kernel_b_x)

        for i in range(h):
            result[i, :] = np.real(ifft(fft(data[i, :]) / kernel_b_x))

        # FFT across columns (y-direction)
        kernel_b_y = np.zeros(h, dtype=np.complex128)
        kernel_b_y[:3] = kernel_b[2:]
        kernel_b_y[-2:] = kernel_b[:2]
        kernel_b_y = fft(kernel_b_y)

        for j in range(w):
            result[:, j] = np.real(ifft(fft(result[:, j]) / kernel_b_y))

        return result

    @staticmethod
    def _to_uint8(img: NDArray) -> NDArray[np.uint8]:
        """Convert image to uint8."""
        if img.dtype == np.uint8:
            return img
        elif img.dtype == np.uint16:
            return (img / 256).astype(np.uint8)
        elif img.dtype in (np.float32, np.float64):
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            return img.astype(np.uint8)

    @staticmethod
    def _to_double(img: NDArray) -> NDArray[np.float64]:
        """Convert image to float64 in range [0, 1]."""
        if img.dtype == np.float64:
            return img
        elif img.dtype == np.float32:
            return img.astype(np.float64)
        elif img.dtype == np.uint8:
            return img.astype(np.float64) / 255.0
        elif img.dtype == np.uint16:
            return img.astype(np.float64) / 65535.0
        else:
            return img.astype(np.float64)

    @staticmethod
    def _rgb_to_grayscale(img: NDArray) -> NDArray[np.float64]:
        """Convert RGB image to grayscale using ITU-R 601-2 luma transform."""
        img_double = NcorrImage._to_double(img)
        if img_double.ndim == 2:
            return img_double
        # Standard luminance weights
        return (
            0.299 * img_double[:, :, 0] +
            0.587 * img_double[:, :, 1] +
            0.114 * img_double[:, :, 2]
        )

    def get_img(self) -> NDArray[np.uint8]:
        """
        Get the RGB image array (H x W x 3).

        For lazy loading, reads the image from file on demand.

        Returns:
            RGB image as uint8 array

        Raises:
            RuntimeError: If image has not been set
        """
        if not self._is_set:
            raise RuntimeError("Image has not been set yet")

        if self.img_type == ImageType.LAZY:
            filepath = Path(self.path) / self.name
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Image could not be located: {filepath}"
                )

            with Image.open(filepath) as pil_img:
                img_array = np.array(pil_img)

            if img_array.ndim == 3 and img_array.shape[2] == 3:
                return self._to_uint8(img_array)
            else:
                img_mono = self._to_uint8(img_array)
                return np.stack([img_mono, img_mono, img_mono], axis=-1)

        return self._img

    def get_gs(self) -> NDArray[np.float64]:
        """
        Get the grayscale array (H x W).

        For lazy loading, reads the image from file on demand.

        Returns:
            Grayscale image as float64 array in range [0, 1]

        Raises:
            RuntimeError: If image has not been set
        """
        if not self._is_set:
            raise RuntimeError("Image has not been set yet")

        if self.img_type == ImageType.LAZY:
            filepath = Path(self.path) / self.name
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Image could not be located: {filepath}"
                )

            with Image.open(filepath) as pil_img:
                img_array = np.array(pil_img)

            return self._rgb_to_grayscale(img_array)

        return self._gs

    def get_bcoef(self) -> NDArray[np.float64]:
        """
        Get the B-spline coefficient array.

        For lazy loading, computes coefficients on demand.

        Returns:
            B-spline coefficients

        Raises:
            RuntimeError: If image has not been set or is reduced type
        """
        if not self._is_set:
            raise RuntimeError("Image has not been set yet")

        if self.img_type == ImageType.REDUCED:
            raise RuntimeError(
                "B-coefficients should never be obtained for reduced images"
            )

        if self.img_type == ImageType.LAZY:
            # Compute on demand
            gs = self.get_gs()
            border = self.border_bcoef
            h, w = gs.shape

            bcoef = np.zeros((h + 2 * border, w + 2 * border), dtype=np.float64)
            bcoef[border:border + h, border:border + w] = gs

            # Fill corners
            bcoef[:border, :border] = bcoef[border, border]
            bcoef[:border, -border:] = bcoef[border, -border - 1]
            bcoef[-border:, -border:] = bcoef[-border - 1, -border - 1]
            bcoef[-border:, :border] = bcoef[-border - 1, border]

            # Fill sides
            bcoef[:border, border:-border] = bcoef[border, border:-border]
            bcoef[border:-border, -border:] = bcoef[border:-border, -border - 1:-border]
            bcoef[-border:, border:-border] = bcoef[-border - 1, border:-border]
            bcoef[border:-border, :border] = bcoef[border:-border, border:border + 1]

            return self.form_bcoef(bcoef)

        return self._bcoef

    def reduce(self, spacing: int) -> "NcorrImage":
        """
        Create a reduced image for display purposes.

        The image is filtered before downsizing to prevent aliasing.

        Args:
            spacing: Spacing parameter for reduction (0 = no reduction)

        Returns:
            Reduced NcorrImage instance

        Raises:
            RuntimeError: If image has not been set
        """
        if not self._is_set:
            raise RuntimeError("Image has not been set yet")

        img_reduced = NcorrImage()
        img_data = self.get_img()

        if spacing > 0:
            # Low-pass filter with Gaussian
            sigma = (spacing + 1) / 2
            filtered = np.zeros_like(img_data)
            for c in range(3):
                filtered[:, :, c] = ndimage.gaussian_filter(
                    img_data[:, :, c].astype(np.float64),
                    sigma=sigma
                ).astype(np.uint8)

            # Resample
            step = spacing + 1
            img_data = filtered[::step, ::step, :]

        img_reduced.set_image(
            ImageType.REDUCED,
            {"img": img_data, "name": self.name, "path": self.path}
        )

        return img_reduced

    def formatted(self) -> dict:
        """
        Get formatted image data as dictionary.

        This is used for passing to computation functions that expect
        a dictionary-like structure.

        Returns:
            Dictionary with all image properties
        """
        if not self._is_set:
            raise RuntimeError("Image has not been set yet")

        result = {
            "img": self.get_img(),
            "gs": self.get_gs(),
            "bcoef": None if self.img_type == ImageType.REDUCED else self.get_bcoef(),
            "type": self.img_type.value,
            "height": self.height,
            "width": self.width,
            "max_gs": self.max_gs,
            "min_gs": self.min_gs,
            "border_bcoef": self.border_bcoef,
            "name": self.name,
            "path": self.path,
        }

        return result

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "NcorrImage":
        """
        Load image from file.

        Convenience method to create an NcorrImage from an image file.

        Args:
            filepath: Path to image file

        Returns:
            Loaded NcorrImage instance
        """
        filepath = Path(filepath)

        with Image.open(filepath) as pil_img:
            img_array = np.array(pil_img)

        img = cls()
        img.set_image(
            ImageType.FILE,
            {
                "img": img_array,
                "name": filepath.name,
                "path": str(filepath.parent),
            }
        )

        return img

    @classmethod
    def from_array(cls, img_array: NDArray, name: str = "array") -> "NcorrImage":
        """
        Create image from numpy array.

        Args:
            img_array: Image data as numpy array
            name: Optional name for the image

        Returns:
            NcorrImage instance
        """
        img = cls()
        img.set_image(
            ImageType.LOAD,
            {"img": img_array, "name": name, "path": ""}
        )
        return img
