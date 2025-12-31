"""Pytest fixtures for Ncorr tests."""

import numpy as np
import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image."""
    np.random.seed(42)
    return (np.random.rand(100, 100) * 255).astype(np.uint8)


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image."""
    np.random.seed(42)
    return (np.random.rand(100, 100, 3) * 255).astype(np.uint8)


@pytest.fixture
def sample_16bit_image():
    """Create a sample 16-bit grayscale image."""
    np.random.seed(42)
    return (np.random.rand(100, 100) * 65535).astype(np.uint16)


@pytest.fixture
def sample_speckle_pattern():
    """Create a synthetic speckle pattern for DIC testing."""
    np.random.seed(42)
    size = 200

    # Create base pattern with multiple frequencies
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    pattern = np.zeros((size, size), dtype=np.float64)

    # Add multiple speckle sizes
    for freq in [10, 20, 30, 50]:
        phase_x = np.random.rand() * 2 * np.pi
        phase_y = np.random.rand() * 2 * np.pi
        pattern += np.sin(2 * np.pi * x / freq + phase_x) * np.sin(2 * np.pi * y / freq + phase_y)

    # Add random noise
    pattern += np.random.randn(size, size) * 0.3

    # Normalize to 0-255
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    return (pattern * 255).astype(np.uint8)


@pytest.fixture
def displaced_speckle_pattern(sample_speckle_pattern):
    """Create a displaced version of the speckle pattern."""
    from scipy.ndimage import shift

    # Apply known displacement
    u_disp = 2.5  # pixels in x
    v_disp = 1.3  # pixels in y

    displaced = shift(sample_speckle_pattern.astype(np.float64), [v_disp, u_disp], mode='nearest')

    return displaced.astype(np.uint8), u_disp, v_disp


@pytest.fixture
def sample_circular_mask():
    """Create a circular mask."""
    size = 100
    center = size // 2
    radius = 40

    y, x = np.ogrid[:size, :size]
    mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2

    return mask


@pytest.fixture
def sample_rectangular_mask():
    """Create a rectangular mask."""
    mask = np.zeros((100, 100), dtype=np.bool_)
    mask[20:80, 20:80] = True
    return mask


@pytest.fixture
def temp_image_file(sample_grayscale_image):
    """Create a temporary image file."""
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        filepath = Path(f.name)

    # Save image
    img = Image.fromarray(sample_grayscale_image)
    img.save(filepath)

    yield filepath

    # Cleanup
    if filepath.exists():
        filepath.unlink()


@pytest.fixture
def temp_image_sequence(sample_speckle_pattern):
    """Create a temporary sequence of images."""
    from PIL import Image
    from scipy.ndimage import shift

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sequence with increasing displacement
        files = []
        for i in range(5):
            displaced = shift(
                sample_speckle_pattern.astype(np.float64),
                [i * 0.5, i * 0.3],
                mode='nearest'
            ).astype(np.uint8)

            filepath = tmpdir / f"image_{i:03d}.png"
            Image.fromarray(displaced).save(filepath)
            files.append(filepath)

        yield files


@pytest.fixture
def dic_parameters():
    """Create default DIC parameters for testing."""
    from ncorr.core.dic_parameters import DICParameters

    return DICParameters(
        radius=15,
        spacing=2,
        cutoff_diffnorm=1e-3,
        cutoff_iteration=30,
        total_threads=1,
        subset_trunc=False,
    )
