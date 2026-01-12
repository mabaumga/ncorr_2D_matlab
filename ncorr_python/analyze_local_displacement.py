"""
Local displacement discontinuity analysis for crack detection.

Computes the LOCAL relative displacement between neighboring points
in the displacement field. A crack appears as a large local jump.

The crack opening displacement (COD) is the displacement difference
across the crack, measured in pixels.

Usage:
    python analyze_local_displacement.py ref.jpg cur.jpg [options]
"""

import sys
import argparse
import numpy as np
from PIL import Image

from ncorr.core.image import NcorrImage
from ncorr.core.roi import NcorrROI
from ncorr.core.dic_parameters import DICParameters
from ncorr.algorithms.dic import DICAnalysis, SeedInfo, DICResult
from ncorr.algorithms.seeds import SeedCalculator


def compute_local_discontinuity(result: DICResult, step: int) -> dict:
    """
    Compute local displacement discontinuity (jump between neighbors).

    This is different from strain - strain is the continuous gradient,
    while discontinuity captures sudden jumps that indicate cracks.

    Args:
        result: DIC result with u, v displacement fields
        step: Grid spacing in pixels

    Returns:
        Dictionary with:
        - delta_u_x: u difference in x-direction (pixels)
        - delta_u_y: u difference in y-direction (pixels)
        - delta_v_x: v difference in x-direction (pixels)
        - delta_v_y: v difference in y-direction (pixels)
        - delta_total: total displacement jump magnitude
    """
    u = result.u.copy()
    v = result.v.copy()
    roi = result.roi

    # Set invalid points to NaN
    u[~roi] = np.nan
    v[~roi] = np.nan

    # Compute differences between neighboring points
    # Using np.diff gives the difference: arr[i+1] - arr[i]

    # Difference in x-direction (between columns)
    delta_u_x = np.full_like(u, np.nan)
    delta_u_x[:, :-1] = np.diff(u, axis=1)  # u[i, j+1] - u[i, j]

    delta_v_x = np.full_like(v, np.nan)
    delta_v_x[:, :-1] = np.diff(v, axis=1)

    # Difference in y-direction (between rows)
    delta_u_y = np.full_like(u, np.nan)
    delta_u_y[:-1, :] = np.diff(u, axis=0)  # u[i+1, j] - u[i, j]

    delta_v_y = np.full_like(v, np.nan)
    delta_v_y[:-1, :] = np.diff(v, axis=0)

    # Total displacement jump magnitude
    # Maximum of horizontal and vertical jumps
    delta_horiz = np.sqrt(delta_u_x**2 + delta_v_x**2)  # Jump to right neighbor
    delta_vert = np.sqrt(delta_u_y**2 + delta_v_y**2)   # Jump to bottom neighbor
    delta_total = np.fmax(delta_horiz, delta_vert)      # Max of both

    return {
        'delta_u_x': delta_u_x,  # Opening in u when moving right
        'delta_u_y': delta_u_y,  # Opening in u when moving down
        'delta_v_x': delta_v_x,  # Opening in v when moving right
        'delta_v_y': delta_v_y,  # Opening in v when moving down
        'delta_horiz': delta_horiz,  # Total jump to right neighbor
        'delta_vert': delta_vert,    # Total jump to bottom neighbor
        'delta_total': delta_total,  # Maximum jump
        'step_pixels': step,  # Grid spacing in original image pixels
    }


def find_discontinuities(disc: dict, threshold: float = 1.0, margin: int = 0) -> list:
    """
    Find locations where displacement discontinuity exceeds threshold.

    Args:
        disc: Dictionary from compute_local_discontinuity
        threshold: Minimum jump in pixels to report
        margin: Number of grid points to exclude from edges (to avoid edge effects)

    Returns:
        List of (x_grid, y_grid, delta_u, delta_v, direction) tuples
    """
    step = disc['step_pixels']
    results = []

    delta_u_x = disc['delta_u_x']
    delta_v_x = disc['delta_v_x']
    delta_u_y = disc['delta_u_y']
    delta_v_y = disc['delta_v_y']

    h, w = delta_u_x.shape

    # Define valid range excluding edge margin
    y_min = margin
    y_max = h - margin
    x_min = margin
    x_max = w - margin

    # Check horizontal jumps (x-direction)
    for iy in range(y_min, y_max):
        for ix in range(max(x_min, 0), min(x_max, w - 1)):
            dux = delta_u_x[iy, ix]
            dvx = delta_v_x[iy, ix]
            if not (np.isnan(dux) or np.isnan(dvx)):
                mag = np.sqrt(dux**2 + dvx**2)
                if mag >= threshold:
                    results.append({
                        'x_grid': ix,
                        'y_grid': iy,
                        'x_pixel': ix * step,
                        'y_pixel': iy * step,
                        'delta_u': dux,
                        'delta_v': dvx,
                        'magnitude': mag,
                        'direction': 'horizontal',  # Crack is vertical (jump when moving horizontally)
                    })

    # Check vertical jumps (y-direction)
    for iy in range(max(y_min, 0), min(y_max, h - 1)):
        for ix in range(x_min, x_max):
            duy = delta_u_y[iy, ix]
            dvy = delta_v_y[iy, ix]
            if not (np.isnan(duy) or np.isnan(dvy)):
                mag = np.sqrt(duy**2 + dvy**2)
                if mag >= threshold:
                    results.append({
                        'x_grid': ix,
                        'y_grid': iy,
                        'x_pixel': ix * step,
                        'y_pixel': iy * step,
                        'delta_u': duy,
                        'delta_v': dvy,
                        'magnitude': mag,
                        'direction': 'vertical',  # Crack is horizontal (jump when moving vertically)
                    })

    # Sort by magnitude (largest first)
    results.sort(key=lambda x: x['magnitude'], reverse=True)

    return results


def print_displacement_summary(result: DICResult, step: int):
    """Print summary of displacement field."""
    roi = result.roi
    u_valid = result.u[roi]
    v_valid = result.v[roi]
    cc_valid = result.corrcoef[roi]

    print("\n" + "="*70)
    print("DISPLACEMENT FIELD SUMMARY")
    print("="*70)
    print(f"  Valid points: {np.sum(roi):,}")
    print(f"  Grid spacing: {step} pixels")
    print(f"\n  u-displacement (horizontal):")
    print(f"    min:  {np.min(u_valid):8.4f} pixels")
    print(f"    max:  {np.max(u_valid):8.4f} pixels")
    print(f"    mean: {np.mean(u_valid):8.4f} pixels")
    print(f"    std:  {np.std(u_valid):8.4f} pixels")
    print(f"\n  v-displacement (vertical):")
    print(f"    min:  {np.min(v_valid):8.4f} pixels")
    print(f"    max:  {np.max(v_valid):8.4f} pixels")
    print(f"    mean: {np.mean(v_valid):8.4f} pixels")
    print(f"    std:  {np.std(v_valid):8.4f} pixels")
    print(f"\n  Correlation coefficient:")
    print(f"    min:  {np.min(cc_valid):8.4f}")
    print(f"    max:  {np.max(cc_valid):8.4f}")
    print(f"    mean: {np.mean(cc_valid):8.4f}")


def print_discontinuity_summary(disc: dict):
    """Print summary of local discontinuities."""
    delta_total = disc['delta_total']
    valid = ~np.isnan(delta_total)

    if not np.any(valid):
        print("\n  No valid discontinuity data")
        return

    dt_valid = delta_total[valid]

    print("\n" + "="*70)
    print("LOCAL DISPLACEMENT DISCONTINUITY (jump between neighbors)")
    print("="*70)
    print(f"  Grid spacing: {disc['step_pixels']} pixels")
    print(f"\n  Displacement jump magnitude:")
    print(f"    min:    {np.min(dt_valid):8.4f} pixels")
    print(f"    max:    {np.max(dt_valid):8.4f} pixels")
    print(f"    mean:   {np.mean(dt_valid):8.4f} pixels")
    print(f"    median: {np.median(dt_valid):8.4f} pixels")
    print(f"    std:    {np.std(dt_valid):8.4f} pixels")

    # Percentiles
    p90 = np.percentile(dt_valid, 90)
    p95 = np.percentile(dt_valid, 95)
    p99 = np.percentile(dt_valid, 99)
    print(f"\n  Percentiles:")
    print(f"    90th: {p90:8.4f} pixels")
    print(f"    95th: {p95:8.4f} pixels")
    print(f"    99th: {p99:8.4f} pixels")

    # Horizontal vs vertical
    dh = disc['delta_horiz']
    dv = disc['delta_vert']
    dh_valid = dh[~np.isnan(dh)]
    dv_valid = dv[~np.isnan(dv)]

    print(f"\n  By direction:")
    print(f"    Horizontal jumps (→): max={np.max(dh_valid):8.4f}, mean={np.mean(dh_valid):8.4f}")
    print(f"    Vertical jumps (↓):   max={np.max(dv_valid):8.4f}, mean={np.mean(dv_valid):8.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze local displacement discontinuity for crack detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script computes the LOCAL relative displacement between neighboring
points in the DIC displacement field. A crack appears as a large jump.

The output gives the crack opening displacement (COD) in PIXELS.
To convert to physical units: COD_mm = COD_pixels * pixel_size_mm

Examples:
  python analyze_local_displacement.py ref.jpg cur.jpg
  python analyze_local_displacement.py ref.jpg cur.jpg --threshold 0.5
  python analyze_local_displacement.py ref.jpg cur.jpg --save results.npz
"""
    )
    parser.add_argument("ref_image", help="Path to reference image")
    parser.add_argument("cur_image", help="Path to current (deformed) image")
    parser.add_argument("--radius", type=int, default=26, help="Subset radius (default: 26)")
    parser.add_argument("--spacing", type=int, default=10, help="Spacing between points (default: 10)")
    parser.add_argument("--border", type=int, default=50, help="ROI border (default: 50)")
    parser.add_argument("--search-radius", type=int, default=50, help="Seed search radius (default: 50)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for reporting discontinuities in pixels (default: 0.5)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top discontinuities to report (default: 20)")
    parser.add_argument("--margin", type=int, default=5,
                        help="Edge margin in grid points to exclude (avoid edge effects, default: 5)")
    parser.add_argument("--save", type=str, default=None, help="Save results to NPZ file")
    parser.add_argument("--quiet", action="store_true", help="Suppress DIC debug output")
    parser.add_argument("--remove-rigid-body", action="store_true",
                        help="Remove rigid body motion (mean translation)")
    parser.add_argument("--remove-trend", action="store_true",
                        help="Remove linear trend (bending/rotation)")
    parser.add_argument("--relative", action="store_true",
                        help="Show relative displacement (removes both rigid body motion and linear trend)")

    args = parser.parse_args()

    # Load images
    print(f"Loading reference: {args.ref_image}")
    ref_arr = np.array(Image.open(args.ref_image).convert('L'))
    print(f"Loading current: {args.cur_image}")
    cur_arr = np.array(Image.open(args.cur_image).convert('L'))

    print(f"Image size: {ref_arr.shape[1]} x {ref_arr.shape[0]}")

    # Create NcorrImages
    ref_img = NcorrImage.from_array(ref_arr)
    cur_img = NcorrImage.from_array(cur_arr)

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

    # Calculate seed
    print(f"\nCalculating seed (search_radius={args.search_radius})...")
    seed_calc = SeedCalculator(params)
    seeds = seed_calc.calculate_seeds(ref_img, cur_img, roi, search_radius=args.search_radius)

    if not seeds or not seeds[0].valid:
        print("ERROR: Could not find valid seed!")
        return

    seed = seeds[0]
    print(f"Seed found: ({seed.x}, {seed.y}), u={seed.u:.2f}, v={seed.v:.2f}")

    # Run DIC
    print("\nRunning DIC analysis...")
    dic = DICAnalysis(params)
    results = dic.analyze(ref_img, [cur_img], roi, seeds)
    result = results[0]

    # Print original displacement summary
    print_displacement_summary(result, step)

    # Apply corrections if requested
    if args.relative:
        print("\n" + "="*70)
        print("APPLYING CORRECTIONS: Rigid body motion + Linear trend removal")
        print("="*70)
        result_corrected = result.get_relative_displacement(
            remove_translation=True, remove_trend=True
        )
        print("  → Rigid body motion (mean translation) removed")
        print("  → Linear trend (bending/rotation) removed")
        print("\nCORRECTED DISPLACEMENT SUMMARY:")
        print_displacement_summary(result_corrected, step)
        result = result_corrected

    elif args.remove_rigid_body or args.remove_trend:
        print("\n" + "="*70)
        print("APPLYING CORRECTIONS")
        print("="*70)

        if args.remove_rigid_body:
            result = result.remove_rigid_body_motion()
            print("  → Rigid body motion (mean translation) removed")

        if args.remove_trend:
            result = result.remove_linear_trend()
            print("  → Linear trend (bending/rotation) removed")

        print("\nCORRECTED DISPLACEMENT SUMMARY:")
        print_displacement_summary(result, step)

    # Compute local discontinuity
    disc = compute_local_discontinuity(result, step)
    print_discontinuity_summary(disc)

    # Find and report large discontinuities (excluding edge effects)
    discontinuities = find_discontinuities(disc, threshold=args.threshold, margin=args.margin)

    print("\n" + "="*70)
    print(f"DETECTED DISCONTINUITIES (threshold >= {args.threshold} pixels, margin={args.margin} grid points)")
    print("="*70)

    # Calculate excluded pixel range for info
    margin_pixels = args.margin * step

    if not discontinuities:
        print(f"  No discontinuities >= {args.threshold} pixels detected")
        print(f"  (Edge margin: {args.margin} grid points = {margin_pixels} pixels excluded from each edge)")
    else:
        print(f"  Found {len(discontinuities)} locations with jump >= {args.threshold} pixels")
        print(f"  (Edge margin: {args.margin} grid points = {margin_pixels} pixels excluded from each edge)")
        print(f"\n  Top {min(args.top, len(discontinuities))} largest jumps:")
        print(f"  {'#':>3} {'X_pixel':>8} {'Y_pixel':>8} {'Δu':>8} {'Δv':>8} {'|Δ|':>8} {'Direction':>12}")
        print(f"  {'-'*3} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

        for i, d in enumerate(discontinuities[:args.top]):
            print(f"  {i+1:3d} {d['x_pixel']:8d} {d['y_pixel']:8d} "
                  f"{d['delta_u']:8.3f} {d['delta_v']:8.3f} {d['magnitude']:8.3f} "
                  f"{d['direction']:>12}")

        # Summary statistics for detected discontinuities
        if len(discontinuities) > 0:
            mags = [d['magnitude'] for d in discontinuities]
            print(f"\n  Statistics of detected discontinuities:")
            print(f"    Max magnitude: {max(mags):.4f} pixels")
            print(f"    Mean magnitude: {np.mean(mags):.4f} pixels")

    # Save results if requested
    if args.save:
        np.savez(args.save,
                 # Displacement fields
                 u=result.u,
                 v=result.v,
                 corrcoef=result.corrcoef,
                 roi=result.roi,
                 # Discontinuity fields
                 delta_u_x=disc['delta_u_x'],
                 delta_u_y=disc['delta_u_y'],
                 delta_v_x=disc['delta_v_x'],
                 delta_v_y=disc['delta_v_y'],
                 delta_total=disc['delta_total'],
                 # Parameters
                 step=step,
                 radius=args.radius,
                 spacing=args.spacing)
        print(f"\nResults saved to {args.save}")
        print(f"  Load with: data = np.load('{args.save}')")
        print(f"  Fields: u, v, corrcoef, roi, delta_u_x, delta_u_y, delta_v_x, delta_v_y, delta_total")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"  - Displacement values are in PIXELS")
    print(f"  - To convert to mm: multiply by your pixel size (mm/pixel)")
    print(f"  - Large discontinuities indicate potential cracks")
    print(f"  - 'horizontal' direction = vertical crack (jump when moving →)")
    print(f"  - 'vertical' direction = horizontal crack (jump when moving ↓)")
    print(f"  - The discontinuity is measured over {step} pixels (grid spacing)")
    print(f"  - Edge margin of {args.margin} grid points excludes boundary artifacts")
    print(f"    (Use --margin 0 to include edges, --margin N to adjust)")


if __name__ == "__main__":
    main()
