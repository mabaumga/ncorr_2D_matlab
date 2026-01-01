"""
Image loading utilities.

Equivalent to ncorr_util_loadimgs.m and ncorr_util_properimgfmt.m
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..core.image import NcorrImage, ImageType


# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_image_format(img_array: NDArray) -> Tuple[bool, str]:
    """
    Validate image format for DIC analysis.

    Supported formats:
    - Grayscale (H x W)
    - RGB (H x W x 3)
    - 8-bit or 16-bit depth

    Args:
        img_array: Image array to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if img_array is None:
        return False, "Image is None"

    if img_array.ndim not in (2, 3):
        return False, f"Invalid dimensions: {img_array.ndim}, expected 2 or 3"

    if img_array.ndim == 3:
        if img_array.shape[2] == 4:
            return False, "CMYK images (4 channels) are not supported"
        if img_array.shape[2] != 3:
            return False, f"Invalid number of channels: {img_array.shape[2]}"

    if img_array.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        return False, f"Unsupported dtype: {img_array.dtype}"

    if img_array.shape[0] < 10 or img_array.shape[1] < 10:
        return False, "Image too small (minimum 10x10)"

    return True, ""


def load_images(
    paths: Union[str, Path, List[Union[str, Path]]],
    lazy: bool = False,
) -> List[NcorrImage]:
    """
    Load images from file paths.

    Supports single image, list of images, or pattern-based batch loading.

    Args:
        paths: Single path, list of paths, or path with pattern
        lazy: If True, use lazy loading (load on demand)

    Returns:
        List of NcorrImage objects

    Raises:
        FileNotFoundError: If file not found
        ValueError: If invalid format
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    images = []
    for path in paths:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        if lazy:
            img = NcorrImage()
            img.set_image(
                ImageType.LAZY,
                {"name": path.name, "path": str(path.parent)}
            )
        else:
            with Image.open(path) as pil_img:
                img_array = np.array(pil_img)

            is_valid, error = validate_image_format(img_array)
            if not is_valid:
                raise ValueError(f"Invalid image format for {path}: {error}")

            img = NcorrImage()
            img.set_image(
                ImageType.FILE,
                {
                    "img": img_array,
                    "name": path.name,
                    "path": str(path.parent),
                }
            )

        images.append(img)

    return images


def load_image_sequence(
    pattern: Union[str, Path],
    start: int = 0,
    end: Optional[int] = None,
    lazy: bool = False,
) -> List[NcorrImage]:
    """
    Load a sequence of numbered images.

    Looks for files matching pattern with incrementing numbers.
    Pattern should contain '#' characters that will be replaced with numbers.

    Example:
        load_image_sequence("image_###.tif", 1, 10)
        -> loads image_001.tif through image_010.tif

    Args:
        pattern: Path pattern with # for number placeholder
        start: Starting number (inclusive)
        end: Ending number (inclusive), None to auto-detect
        lazy: Use lazy loading

    Returns:
        List of NcorrImage objects
    """
    pattern = str(pattern)

    # Find # placeholder
    match = re.search(r"#+", pattern)
    if not match:
        # No pattern, treat as single file
        return load_images(pattern, lazy)

    num_digits = len(match.group())
    prefix = pattern[:match.start()]
    suffix = pattern[match.end():]

    paths = []
    num = start

    while True:
        filename = f"{prefix}{num:0{num_digits}d}{suffix}"
        path = Path(filename)

        if not path.exists():
            if end is None:
                break  # Auto-detect end
            elif num > end:
                break
            else:
                raise FileNotFoundError(f"Missing file in sequence: {path}")

        paths.append(path)
        num += 1

        if end is not None and num > end:
            break

    if not paths:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    return load_images(paths, lazy)


def get_image_info(path: Union[str, Path]) -> dict:
    """
    Get image information without loading full image.

    Args:
        path: Path to image file

    Returns:
        Dictionary with image info
    """
    path = Path(path)

    with Image.open(path) as pil_img:
        info = {
            "path": str(path),
            "name": path.name,
            "format": pil_img.format,
            "mode": pil_img.mode,
            "width": pil_img.width,
            "height": pil_img.height,
            "size_bytes": path.stat().st_size,
        }

        # Estimate bit depth
        mode_to_bits = {
            "L": 8,
            "RGB": 8,
            "RGBA": 8,
            "I;16": 16,
            "I;16B": 16,
        }
        info["bit_depth"] = mode_to_bits.get(pil_img.mode, 8)

    return info


def images_compatible(img1: NcorrImage, img2: NcorrImage) -> Tuple[bool, str]:
    """
    Check if two images are compatible for DIC analysis.

    Images must have the same dimensions.

    Args:
        img1: First image
        img2: Second image

    Returns:
        Tuple of (is_compatible, reason)
    """
    if img1.height != img2.height:
        return False, f"Height mismatch: {img1.height} vs {img2.height}"

    if img1.width != img2.width:
        return False, f"Width mismatch: {img1.width} vs {img2.width}"

    return True, ""
