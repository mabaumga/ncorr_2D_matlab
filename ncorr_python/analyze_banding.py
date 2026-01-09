"""
Script to analyze horizontal banding in DIC results.

This script loads DIC results and performs detailed row-by-row analysis
to understand the source of horizontal banding artifacts.

Usage:
    python analyze_banding.py <ref_image> <cur_image> [--radius 26] [--spacing 10]
"""

import sys
import argparse
import numpy as np
from PIL import Image

from ncorr.core.image import NcorrImage
from ncorr.core.roi import NcorrROI
from ncorr.core.dic_parameters import DICParameters
from ncorr.algorithms.dic import DICAnalysis, SeedInfo
from ncorr.algorithms.seeds import SeedCalculator


def analyze_banding(result, step):
    """Perform detailed banding analysis on DIC result."""
    u_plot = result.u
    v_plot = result.v
    cc_plot = result.corrcoef
    roi_plot = result.roi

    out_h, out_w = u_plot.shape

    print("\n" + "="*70)
    print("DETAILED BANDING ANALYSIS")
    print("="*70)

    # Collect row statistics
    row_stats = []
    for iy in range(out_h):
        row_mask = roi_plot[iy, :]
        if np.any(row_mask):
            u_row = u_plot[iy, row_mask]
            v_row = v_plot[iy, row_mask]
            cc_row = cc_plot[iy, row_mask]
            row_stats.append({
                'row': iy,
                'y_orig': iy * step,
                'n': len(u_row),
                'u_mean': np.mean(u_row),
                'u_std': np.std(u_row),
                'v_mean': np.mean(v_row),
                'v_std': np.std(v_row),
                'cc_mean': np.mean(cc_row),
            })

    if len(row_stats) < 2:
        print("Not enough rows with data for banding analysis.")
        return

    # Convert to arrays for easier analysis
    rows = np.array([rs['row'] for rs in row_stats])
    v_means = np.array([rs['v_mean'] for rs in row_stats])
    u_means = np.array([rs['u_mean'] for rs in row_stats])

    # 1. Row-to-row differences
    v_diffs = np.diff(v_means)
    u_diffs = np.diff(u_means)

    print(f"\n1. BASIC STATISTICS:")
    print(f"   Total rows with data: {len(row_stats)}")
    print(f"   V-mean across all rows: {np.mean(v_means):.4f}")
    print(f"   V-std across rows: {np.std(v_means):.4f}")
    print(f"   V-range: {np.min(v_means):.4f} to {np.max(v_means):.4f}")
    print(f"   Peak-to-peak V variation: {np.max(v_means) - np.min(v_means):.4f}")

    print(f"\n2. ROW-TO-ROW VARIATION:")
    print(f"   Mean |delta_v|: {np.mean(np.abs(v_diffs)):.4f}")
    print(f"   Max |delta_v|: {np.max(np.abs(v_diffs)):.4f}")
    print(f"   Mean |delta_u|: {np.mean(np.abs(u_diffs)):.4f}")
    print(f"   Max |delta_u|: {np.max(np.abs(u_diffs)):.4f}")

    # 2. Check for periodicity
    if len(v_means) >= 20:
        # Compute autocorrelation
        v_centered = v_means - np.mean(v_means)
        autocorr = np.correlate(v_centered, v_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, min(len(autocorr)-1, 30)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                peaks.append((i, autocorr[i]))

        print(f"\n3. PERIODICITY ANALYSIS:")
        if peaks:
            print(f"   Potential periodic patterns found:")
            for period, strength in peaks[:3]:
                print(f"     Period={period} rows ({period*step} pixels), strength={strength:.3f}")
        else:
            print(f"   No strong periodicity detected (correlation < 0.3)")

        print(f"   First 10 autocorrelation values: {autocorr[:10]}")

    # 3. Identify worst rows
    print(f"\n4. ROWS WITH LARGEST V-DEVIATION:")
    v_deviation = np.abs(v_means - np.mean(v_means))
    worst_indices = np.argsort(v_deviation)[-10:][::-1]
    print(f"   {'Row':>5} {'Y_orig':>7} {'V_mean':>8} {'Deviation':>10}")
    for idx in worst_indices:
        rs = row_stats[idx]
        print(f"   {rs['row']:5d} {rs['y_orig']:7d} {rs['v_mean']:8.4f} {v_deviation[idx]:10.4f}")

    # 4. Check if there's a gradient in v (should be ~0 for horizontal stretch)
    if len(v_means) > 5:
        from numpy.polynomial import polynomial as P
        coefs = P.polyfit(rows, v_means, 1)
        slope = coefs[1]

        print(f"\n5. V-GRADIENT (linear fit):")
        print(f"   Slope: {slope:.6f} pixels/row = {slope/step:.6f} strain")
        print(f"   (Should be ~0 for pure horizontal stretch)")

    # 5. Output full row table (every N rows)
    skip = max(1, len(row_stats) // 30)
    print(f"\n6. ROW-BY-ROW DATA (every {skip} rows):")
    print(f"   {'Row':>5} {'Y_orig':>7} {'N':>5} {'u_mean':>8} {'v_mean':>8} {'v_std':>7} {'CC_mean':>7}")
    for i, rs in enumerate(row_stats):
        if i % skip == 0:
            print(f"   {rs['row']:5d} {rs['y_orig']:7d} {rs['n']:5d} {rs['u_mean']:8.3f} "
                  f"{rs['v_mean']:8.3f} {rs['v_std']:7.4f} {rs['cc_mean']:7.4f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze DIC banding artifacts")
    parser.add_argument("ref_image", help="Path to reference image")
    parser.add_argument("cur_image", help="Path to current (deformed) image")
    parser.add_argument("--radius", type=int, default=26, help="Subset radius (default: 26)")
    parser.add_argument("--spacing", type=int, default=10, help="Spacing between points (default: 10)")
    parser.add_argument("--border", type=int, default=50, help="ROI border (default: 50)")
    parser.add_argument("--search-radius", type=int, default=50, help="Seed search radius (default: 50)")
    parser.add_argument("--smooth", type=int, default=0, help="Apply row smoothing with given window size (default: 0=no smoothing)")
    parser.add_argument("--save", type=str, default=None, help="Save results to NPZ file")

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

    print(f"ROI: {border} to {width-border} in x, {border} to {height-border} in y")

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

    # Calculate seed automatically
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

    # Analyze banding before smoothing
    print("\n" + "="*70)
    print("BEFORE SMOOTHING")
    print("="*70)

    banding = result.analyze_row_banding()
    print(f"Banding analysis:")
    print(f"  V row-to-row std:    {banding.get('v_row_std', 0):.4f}")
    print(f"  V row-to-row range:  {banding.get('v_row_range', 0):.4f}")
    print(f"  V row-to-row mean:   {banding.get('v_row_to_row_mean', 0):.4f}")
    print(f"  Banding detected:    {banding.get('banding_detected', False)}")

    # Perform detailed banding analysis
    analyze_banding(result, step)

    # Apply smoothing if requested
    if args.smooth > 0:
        print("\n" + "="*70)
        print(f"AFTER SMOOTHING (window_size={args.smooth})")
        print("="*70)

        result_smoothed = result.smooth_row_banding(window_size=args.smooth)

        banding_after = result_smoothed.analyze_row_banding()
        print(f"Banding analysis after smoothing:")
        print(f"  V row-to-row std:    {banding_after.get('v_row_std', 0):.4f}")
        print(f"  V row-to-row range:  {banding_after.get('v_row_range', 0):.4f}")
        print(f"  V row-to-row mean:   {banding_after.get('v_row_to_row_mean', 0):.4f}")
        print(f"  Banding detected:    {banding_after.get('banding_detected', False)}")

        # Use smoothed result for saving
        result = result_smoothed

    # Save results if requested
    if args.save:
        np.savez(args.save,
                 u=result.u,
                 v=result.v,
                 corrcoef=result.corrcoef,
                 roi=result.roi,
                 step=step)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
