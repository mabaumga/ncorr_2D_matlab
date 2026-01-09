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

import numpy as np
from scipy.ndimage import map_coordinates, shift
from PIL import Image

from ncorr.core.image import NcorrImage
from ncorr.core.roi import NcorrROI
from ncorr.core.dic_parameters import DICParameters
from ncorr.algorithms.dic import DICAnalysis, SeedInfo


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


def create_deformed_image_gradient(ref_image, strain_x=0.01):
    """
    Create a deformed image simulating a tensile test with displacement gradient.

    The left edge is fixed (u=0), and displacement increases linearly
    toward the right edge.

    DIC convention: point at (x,y) in reference moves to (x+u, y+v) in current.
    """
    height, width = ref_image.shape

    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Displacement field: u(x) = strain_x * x
    # At a point x in the reference, the displacement is u = strain_x * x
    # That point appears at x_new = x + u = x + strain_x * x = x*(1 + strain_x) in current
    #
    # To create the current image from reference:
    # current(x_new) = ref(x) where x_new = x*(1 + strain_x)
    # So: current(x_new) = ref(x_new / (1 + strain_x))
    #
    # For each pixel x in current image, find where it came from in reference
    x_source = x_coords / (1 + strain_x)
    y_source = y_coords.astype(np.float64)

    # Interpolate
    deformed = map_coordinates(
        ref_image.astype(np.float64),
        [y_source, x_source],
        order=3,
        mode='nearest'
    )

    return deformed.astype(np.uint8), strain_x


def main():
    """Run the test."""
    # Create images
    width, height = 800, 200
    print(f"Creating synthetic images: {width}x{height}")

    ref_image = create_synthetic_speckle(width, height)

    # Test with VERTICAL displacement (like user's case: v≈-37 to -41)
    u_shift = 0.0
    v_shift = -40.0  # Large negative v displacement
    cur_image = shift(ref_image.astype(np.float64), [v_shift, u_shift], mode='nearest').astype(np.uint8)
    print(f"Applied uniform displacement: u={u_shift:.2f}, v={v_shift:.2f} pixels")
    strain_x = 0  # For compatibility with seed setup below

    # Save images for inspection
    Image.fromarray(ref_image).save('/tmp/ref_tensile.png')
    Image.fromarray(cur_image).save('/tmp/cur_tensile.png')
    print("Saved images to /tmp/ref_tensile.png and /tmp/cur_tensile.png")

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

    # Calculate expected boundary constraints
    # For the warped subset to be valid, we need:
    #   y_min_warped = y - radius + v + border >= 2
    #   y_max_warped = y + radius + v + border < h - 3
    # This gives constraints on valid y range in reference image
    if v_shift < 0:
        y_min_valid = max(border, 2 + radius - v_shift - border)
        y_max_valid = min(height - border - 1, height + border - 3 - radius - v_shift - border)
        print(f"\nBoundary constraints (with v={v_shift:.1f}, radius={radius}):")
        print(f"  Minimum valid y: {y_min_valid:.0f} (subset bottom edge stays in image)")
        print(f"  Maximum valid y: {y_max_valid:.0f}")

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
