"""
Test script for DIC with synthetic tensile test images.

This script helps diagnose propagation issues by:
1. Creating synthetic speckle patterns with known displacements
2. Running DIC analysis with diagnostic output
3. Calculating boundary constraints to explain why points fail

Key findings:
- With large displacement v, points at small y values fail due to boundary effects
- The minimum valid y = 2 + radius + |v| - border (approx)
- Displacement jump cutoff rarely triggers with correct initial guess
"""

import os
import tempfile
import numpy as np
from scipy.ndimage import map_coordinates, shift
from PIL import Image

from ncorr.core.image import NcorrImage
from ncorr.core.roi import NcorrROI
from ncorr.core.dic_parameters import DICParameters
from ncorr.algorithms.dic import DICAnalysis, SeedInfo

# Use temp directory that works on all platforms
TEMP_DIR = tempfile.gettempdir()


def create_synthetic_speckle(width, height, seed=42, speckle_size=3, speckle_density=0.3):
    """Create a synthetic speckle pattern with random dots."""
    np.random.seed(seed)

    # Start with medium gray background
    pattern = np.ones((height, width), dtype=np.float64) * 0.5

    # Add random Gaussian spots (speckles)
    n_speckles = int(width * height * speckle_density / (speckle_size ** 2))

    for _ in range(n_speckles):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        intensity = np.random.uniform(-0.5, 0.5)

        # Add Gaussian spot
        for dy in range(-speckle_size*2, speckle_size*2 + 1):
            for dx in range(-speckle_size*2, speckle_size*2 + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < width and 0 <= py < height:
                    dist_sq = dx*dx + dy*dy
                    weight = np.exp(-dist_sq / (2 * speckle_size**2))
                    pattern[py, px] += intensity * weight

    # Add fine noise
    pattern += np.random.randn(height, width) * 0.05

    # Normalize to 0-255
    pattern = np.clip(pattern, 0, 1)
    return (pattern * 255).astype(np.uint8)


def create_deformed_image_tensile(ref_image, strain_x=0.01):
    """
    Create a deformed image simulating a tensile test with UNIFORM displacement.

    This is simpler - just shift the whole image by a constant amount.
    """
    height, width = ref_image.shape

    # Uniform displacement: shift whole image
    u_shift = 5.0  # pixels

    # Create coordinate grid for the deformed image
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # For DIC convention: ref(x,y) -> deformed(x+u, y+v)
    # To create deformed from ref: deformed(x,y) = ref(x-u, y-v)
    x_source = x_coords - u_shift
    y_source = y_coords.astype(np.float64)

    # Interpolate
    deformed = map_coordinates(
        ref_image.astype(np.float64),
        [y_source, x_source],
        order=3,
        mode='nearest'
    )

    return deformed.astype(np.uint8), u_shift


def create_deformed_image_symmetric(ref_image, strain_x=0.01, strain_y=0.0):
    """
    Create a deformed image with SYMMETRIC strain around the center.

    This simulates a tensile test where:
    - The center of the image has zero displacement (u=0, v=0)
    - Displacement increases linearly away from the center
    - u(x) = strain_x * (x - center_x)
    - v(y) = strain_y * (y - center_y)

    DIC convention: point at (x,y) in reference moves to (x+u, y+v) in current.
    """
    height, width = ref_image.shape
    center_x = width / 2
    center_y = height / 2

    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # For symmetric deformation around center:
    # A point at (x, y) in reference moves to:
    #   x_new = center_x + (x - center_x) * (1 + strain_x)
    #   y_new = center_y + (y - center_y) * (1 + strain_y)
    #
    # To create current image from reference (inverse mapping):
    # current(x_new, y_new) = ref(x, y)
    # where x = center_x + (x_new - center_x) / (1 + strain_x)
    #       y = center_y + (y_new - center_y) / (1 + strain_y)

    x_source = center_x + (x_coords - center_x) / (1 + strain_x)
    y_source = center_y + (y_coords - center_y) / (1 + strain_y)

    # Interpolate
    deformed = map_coordinates(
        ref_image.astype(np.float64),
        [y_source, x_source],
        order=3,
        mode='nearest'
    )

    return deformed.astype(np.uint8), strain_x, strain_y


def main():
    """Run the test."""
    # Create images
    width, height = 800, 200
    print(f"Creating synthetic images: {width}x{height}")

    ref_image = create_synthetic_speckle(width, height)

    # Test with SYMMETRIC strain around center (like user's MATLAB images)
    # This creates a displacement field where u=0 at center
    strain_x = 0.01  # 1% strain in x-direction
    strain_y = -0.003  # Small negative strain in y (Poisson effect)

    cur_image, _, _ = create_deformed_image_symmetric(ref_image, strain_x, strain_y)

    # Expected displacement:
    # At center (x=400, y=100): u=0, v=0
    # At x=404 (one step right): u = strain_x * 4 = 0.04 pixels
    # At x=0: u = strain_x * (0 - 400) = -4 pixels
    # At x=800: u = strain_x * (800 - 400) = +4 pixels

    print(f"Applied SYMMETRIC strain around center:")
    print(f"  strain_x = {strain_x*100:.2f}%")
    print(f"  strain_y = {strain_y*100:.2f}%")
    print(f"Expected displacement at center: u=0, v=0")
    print(f"Expected displacement at edges: u=±{abs(strain_x * 400):.1f} pixels")
    print(f"Expected strain gradient: du/dx = {strain_x:.4f}")

    u_shift = 0.0  # At center
    v_shift = 0.0

    # Save images for inspection
    ref_path = os.path.join(TEMP_DIR, 'ref_tensile.png')
    cur_path = os.path.join(TEMP_DIR, 'cur_tensile.png')
    Image.fromarray(ref_image).save(ref_path)
    Image.fromarray(cur_image).save(cur_path)
    print(f"Saved images to {ref_path} and {cur_path}")

    # Create NcorrImages
    ref_img = NcorrImage.from_array(ref_image)
    cur_img = NcorrImage.from_array(cur_image)

    # Create ROI (full image minus borders)
    border = 20
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[border:height-border, border:width-border] = True

    roi = NcorrROI()
    roi.set_roi("load", {"mask": mask, "cutoff": 0})

    print(f"ROI: {border} to {width-border} in x, {border} to {height-border} in y")
    print(f"Number of regions: {len(roi.regions)}")

    # DIC parameters
    radius = 13
    spacing = 3
    step = spacing + 1

    params = DICParameters(
        radius=radius,
        spacing=spacing,
        cutoff_diffnorm=1e-4,
        cutoff_iteration=50,
        total_threads=1,
        subset_trunc=False,
    )

    print(f"\nDIC parameters: radius={radius}, spacing={spacing}, step={step}")

    # Create seed at center with CORRECT initial guess
    seed_x = 400
    seed_y = 100
    seed_u = u_shift
    seed_v = v_shift

    print(f"Seed: ({seed_x}, {seed_y}), initial guess: u={seed_u:.4f}, v={seed_v:.4f}")

    seeds = [SeedInfo(x=seed_x, y=seed_y, u=seed_u, v=seed_v, region_idx=0, valid=True)]

    # Run DIC
    dic = DICAnalysis(params)
    results = dic.analyze(ref_img, [cur_img], roi, seeds)

    result = results[0]

    # Analyze results
    valid_mask = result.roi
    n_valid = np.sum(valid_mask)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Valid points: {n_valid}")

    if n_valid > 0:
        u_valid = result.u[valid_mask]
        v_valid = result.v[valid_mask]
        cc_valid = result.corrcoef[valid_mask]

        print(f"  u: min={np.min(u_valid):.4f}, max={np.max(u_valid):.4f}, mean={np.mean(u_valid):.4f}")
        print(f"  v: min={np.min(v_valid):.4f}, max={np.max(v_valid):.4f}, mean={np.mean(v_valid):.4f}")
        print(f"  CC: min={np.min(cc_valid):.4f}, max={np.max(cc_valid):.4f}, mean={np.mean(cc_valid):.4f}")

        # Check spatial extent
        valid_indices = np.where(valid_mask)
        y_min, y_max = np.min(valid_indices[0]), np.max(valid_indices[0])
        x_min, x_max = np.min(valid_indices[1]), np.max(valid_indices[1])

        print(f"\n  Spatial extent (in output grid coordinates):")
        print(f"    x: {x_min} to {x_max} (original: {x_min*step} to {x_max*step})")
        print(f"    y: {y_min} to {y_max} (original: {y_min*step} to {y_max*step})")

        # Expected extent
        expected_x_min = border // step
        expected_x_max = (width - border) // step - 1
        expected_y_min = border // step
        expected_y_max = (height - border) // step - 1

        print(f"\n  Expected extent:")
        print(f"    x: {expected_x_min} to {expected_x_max}")
        print(f"    y: {expected_y_min} to {expected_y_max}")

        # Coverage
        expected_points = (expected_x_max - expected_x_min + 1) * (expected_y_max - expected_y_min + 1)
        coverage = n_valid / expected_points * 100
        print(f"\n  Coverage: {coverage:.1f}% ({n_valid}/{expected_points})")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
