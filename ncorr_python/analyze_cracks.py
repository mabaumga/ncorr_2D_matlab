"""
Crack detection analysis for DIC image sequences.

Analyzes displacement fields to detect crack initiation and propagation
by identifying:
1. Displacement discontinuities (jumps in u or v)
2. Strain concentrations (high local strain)
3. Temporal changes between successive images

Usage:
    python analyze_cracks.py ref.jpg cur1.jpg cur2.jpg cur3.jpg ... [options]
"""

import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

from ncorr.core.image import NcorrImage
from ncorr.core.roi import NcorrROI
from ncorr.core.dic_parameters import DICParameters
from ncorr.algorithms.dic import DICAnalysis, SeedInfo, DICResult
from ncorr.algorithms.seeds import SeedCalculator


def compute_strain_field(result: DICResult, step: int) -> dict:
    """
    Compute strain fields from displacement fields.

    Returns:
        Dictionary with strain components:
        - exx: du/dx (strain in x-direction)
        - eyy: dv/dy (strain in y-direction)
        - exy: 0.5*(du/dy + dv/dx) (shear strain)
        - e_eff: effective strain (von Mises equivalent)
    """
    u = result.u.copy()
    v = result.v.copy()
    roi = result.roi

    # Replace invalid points with NaN for gradient calculation
    u[~roi] = np.nan
    v[~roi] = np.nan

    # Compute gradients (in grid coordinates, multiply by 1/step for real coordinates)
    # Using central differences where possible
    exx = np.gradient(u, axis=1) / step  # du/dx
    eyy = np.gradient(v, axis=0) / step  # dv/dy
    dudy = np.gradient(u, axis=0) / step
    dvdx = np.gradient(v, axis=1) / step
    exy = 0.5 * (dudy + dvdx)  # shear strain

    # Effective strain (von Mises equivalent for plane strain)
    e_eff = np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)

    return {
        'exx': exx,
        'eyy': eyy,
        'exy': exy,
        'e_eff': e_eff,
        'roi': roi,
    }


def detect_cracks(strain: dict, threshold_factor: float = 3.0) -> dict:
    """
    Detect potential crack locations based on strain concentration.

    A crack is indicated by:
    - Local strain significantly higher than surrounding area
    - Strain exceeding threshold_factor * median strain

    Args:
        strain: Dictionary from compute_strain_field
        threshold_factor: Multiple of median strain to flag as potential crack

    Returns:
        Dictionary with crack detection results
    """
    roi = strain['roi']
    e_eff = strain['e_eff']
    exx = strain['exx']

    # Use only valid points
    e_eff_valid = e_eff[roi]
    exx_valid = exx[roi]

    if len(e_eff_valid) == 0:
        return {'crack_detected': False, 'message': 'No valid strain data'}

    # Statistics
    e_eff_median = np.nanmedian(e_eff_valid)
    e_eff_std = np.nanstd(e_eff_valid)
    exx_median = np.nanmedian(exx_valid)
    exx_max = np.nanmax(exx_valid)

    # Threshold for crack detection
    threshold = e_eff_median + threshold_factor * e_eff_std

    # Find points exceeding threshold
    crack_mask = (e_eff > threshold) & roi
    crack_points = np.sum(crack_mask)

    # Find location of maximum strain
    if np.any(roi & ~np.isnan(e_eff)):
        e_eff_masked = np.where(roi, e_eff, -np.inf)
        max_idx = np.unravel_index(np.nanargmax(e_eff_masked), e_eff.shape)
        max_y, max_x = max_idx
        max_strain = e_eff[max_y, max_x]
    else:
        max_y, max_x, max_strain = 0, 0, 0

    return {
        'crack_detected': crack_points > 0,
        'crack_points': crack_points,
        'crack_mask': crack_mask,
        'threshold': threshold,
        'e_eff_median': e_eff_median,
        'e_eff_std': e_eff_std,
        'e_eff_max': max_strain,
        'max_location': (max_x, max_y),
        'exx_median': exx_median,
        'exx_max': exx_max,
    }


def compute_differential(result1: DICResult, result2: DICResult) -> DICResult:
    """
    Compute differential displacement between two results.

    This shows the CHANGE in displacement between images,
    which highlights crack opening.
    """
    # Both must have same shape and ROI
    roi_common = result1.roi & result2.roi

    u_diff = result2.u - result1.u
    v_diff = result2.v - result1.v

    # Set invalid points to NaN
    u_diff[~roi_common] = np.nan
    v_diff[~roi_common] = np.nan

    return DICResult(
        u=u_diff,
        v=v_diff,
        corrcoef=np.minimum(result1.corrcoef, result2.corrcoef),
        roi=roi_common,
    )


def analyze_sequence(ref_img, cur_imgs, roi, params, seeds):
    """
    Analyze a sequence of images for crack detection.
    """
    step = params.spacing + 1

    # Run DIC on all images
    print(f"\nAnalyzing {len(cur_imgs)} images...")
    dic = DICAnalysis(params)
    results = dic.analyze(ref_img, cur_imgs, roi, seeds)

    print("\n" + "="*70)
    print("CRACK ANALYSIS RESULTS")
    print("="*70)

    all_strains = []
    all_cracks = []

    for i, result in enumerate(results):
        print(f"\n--- Image {i+1} ---")

        # Compute strain field
        strain = compute_strain_field(result, step)
        all_strains.append(strain)

        # Detect cracks
        cracks = detect_cracks(strain, threshold_factor=3.0)
        all_cracks.append(cracks)

        print(f"  Effective strain: median={cracks['e_eff_median']:.6f}, "
              f"max={cracks['e_eff_max']:.6f}")
        print(f"  εxx: median={cracks['exx_median']:.6f}, max={cracks['exx_max']:.6f}")

        if cracks['crack_detected']:
            print(f"  ⚠️  POTENTIAL CRACK DETECTED!")
            print(f"      {cracks['crack_points']} points above threshold")
            print(f"      Max strain location: x={cracks['max_location'][0]*step}, "
                  f"y={cracks['max_location'][1]*step}")
        else:
            print(f"  ✓ No crack detected (threshold={cracks['threshold']:.6f})")

    # Differential analysis between successive images
    if len(results) > 1:
        print("\n" + "="*70)
        print("DIFFERENTIAL ANALYSIS (changes between successive images)")
        print("="*70)

        for i in range(1, len(results)):
            diff = compute_differential(results[i-1], results[i])
            diff_strain = compute_strain_field(diff, step)

            valid = diff.roi
            if np.any(valid):
                du_max = np.nanmax(np.abs(diff.u[valid]))
                dv_max = np.nanmax(np.abs(diff.v[valid]))
                de_max = np.nanmax(diff_strain['e_eff'][valid])

                print(f"\n  Image {i} → {i+1}:")
                print(f"    Max Δu: {du_max:.4f} pixels")
                print(f"    Max Δv: {dv_max:.4f} pixels")
                print(f"    Max Δε_eff: {de_max:.6f}")

                # Large displacement jump indicates crack opening
                if du_max > 1.0 or dv_max > 1.0:
                    print(f"    ⚠️  LARGE DISPLACEMENT CHANGE - possible crack opening!")

    return results, all_strains, all_cracks


def save_results(output_path, results, strains, step):
    """Save all results to NPZ file."""
    data = {
        'step': step,
        'n_images': len(results),
    }

    for i, (result, strain) in enumerate(zip(results, strains)):
        data[f'u_{i}'] = result.u
        data[f'v_{i}'] = result.v
        data[f'cc_{i}'] = result.corrcoef
        data[f'roi_{i}'] = result.roi
        data[f'exx_{i}'] = strain['exx']
        data[f'eyy_{i}'] = strain['eyy']
        data[f'exy_{i}'] = strain['exy']
        data[f'e_eff_{i}'] = strain['e_eff']

    np.savez(output_path, **data)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DIC image sequence for crack detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze 3 images relative to reference
  python analyze_cracks.py ref.jpg img1.jpg img2.jpg img3.jpg

  # With custom parameters
  python analyze_cracks.py ref.jpg img*.jpg --radius 26 --spacing 10

  # Save results
  python analyze_cracks.py ref.jpg img*.jpg --save results.npz
"""
    )
    parser.add_argument("ref_image", help="Path to reference image")
    parser.add_argument("cur_images", nargs='+', help="Paths to current (deformed) images")
    parser.add_argument("--radius", type=int, default=26, help="Subset radius (default: 26)")
    parser.add_argument("--spacing", type=int, default=10, help="Spacing between points (default: 10)")
    parser.add_argument("--border", type=int, default=50, help="ROI border (default: 50)")
    parser.add_argument("--search-radius", type=int, default=50, help="Seed search radius (default: 50)")
    parser.add_argument("--save", type=str, default=None, help="Save results to NPZ file")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Crack detection threshold (multiples of std, default: 3.0)")

    args = parser.parse_args()

    # Load reference image
    print(f"Loading reference: {args.ref_image}")
    ref_arr = np.array(Image.open(args.ref_image).convert('L'))
    ref_img = NcorrImage.from_array(ref_arr)

    # Load current images
    cur_imgs = []
    for path in args.cur_images:
        print(f"Loading: {path}")
        cur_arr = np.array(Image.open(path).convert('L'))
        cur_imgs.append(NcorrImage.from_array(cur_arr))

    print(f"\nImage size: {ref_arr.shape[1]} x {ref_arr.shape[0]}")
    print(f"Number of images to analyze: {len(cur_imgs)}")

    # Create ROI
    height, width = ref_arr.shape
    border = args.border
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[border:height-border, border:width-border] = True

    roi = NcorrROI()
    roi.set_roi("load", {"mask": mask, "cutoff": 0})

    # DIC parameters
    params = DICParameters(
        radius=args.radius,
        spacing=args.spacing,
        cutoff_diffnorm=1e-4,
        cutoff_iteration=50,
        total_threads=1,
        subset_trunc=False,
    )

    step = args.spacing + 1
    print(f"DIC parameters: radius={args.radius}, spacing={args.spacing}, step={step}")

    # Calculate seed using first current image
    print(f"\nCalculating seed (search_radius={args.search_radius})...")
    seed_calc = SeedCalculator(params)
    seeds = seed_calc.calculate_seeds(ref_img, cur_imgs[0], roi, search_radius=args.search_radius)

    if not seeds or not seeds[0].valid:
        print("ERROR: Could not find valid seed!")
        return

    seed = seeds[0]
    print(f"Seed found: ({seed.x}, {seed.y}), u={seed.u:.2f}, v={seed.v:.2f}")

    # Analyze sequence
    results, strains, cracks = analyze_sequence(ref_img, cur_imgs, roi, params, seeds)

    # Save results if requested
    if args.save:
        save_results(args.save, results, strains, step)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Images analyzed: {len(results)}")

    crack_images = [i+1 for i, c in enumerate(cracks) if c['crack_detected']]
    if crack_images:
        print(f"  ⚠️  Potential cracks detected in images: {crack_images}")
    else:
        print(f"  ✓ No cracks detected in any image")

    print("\nTo visualize results, load the NPZ file in Python:")
    print("  data = np.load('results.npz')")
    print("  plt.imshow(data['e_eff_0'], cmap='hot')  # Effective strain, image 0")


if __name__ == "__main__":
    main()
