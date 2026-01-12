"""
Analyze raw images for stripe/banding artifacts.

This script checks if stripes are present in the RAW IMAGES themselves
(before DIC analysis), which would indicate camera/lighting artifacts.

Usage:
    python diagnose_stripes.py image1.jpg image2.jpg
"""

import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def analyze_horizontal_stripes(img_array, name="Image"):
    """
    Analyze image for horizontal stripe artifacts.

    Stripes appear as row-to-row intensity variations that are
    correlated across the image width.
    """
    height, width = img_array.shape

    # Compute mean intensity per row
    row_means = np.mean(img_array, axis=1)

    # Row-to-row differences
    row_diffs = np.diff(row_means)

    print(f"\n{'='*60}")
    print(f"STRIPE ANALYSIS: {name}")
    print(f"{'='*60}")
    print(f"  Image size: {width} x {height}")
    print(f"  Overall mean intensity: {np.mean(img_array):.2f}")
    print(f"  Overall std intensity: {np.std(img_array):.2f}")

    print(f"\n  Row-wise intensity variation:")
    print(f"    Row mean range: {np.min(row_means):.2f} to {np.max(row_means):.2f}")
    print(f"    Row mean std: {np.std(row_means):.2f}")
    print(f"    Row-to-row diff mean: {np.mean(np.abs(row_diffs)):.3f}")
    print(f"    Row-to-row diff max: {np.max(np.abs(row_diffs)):.3f}")

    # Check for periodicity in row means
    if len(row_means) >= 20:
        # Autocorrelation
        rm_centered = row_means - np.mean(row_means)
        autocorr = np.correlate(rm_centered, rm_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find peaks
        peaks = []
        for i in range(1, min(len(autocorr)-1, 50)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                peaks.append((i, autocorr[i]))

        if peaks:
            print(f"\n  Periodic pattern detected:")
            for period, strength in peaks[:3]:
                print(f"    Period = {period} rows, strength = {strength:.3f}")
        else:
            print(f"\n  No strong periodic pattern in row intensities")

    # Severity assessment
    relative_variation = np.std(row_means) / np.mean(img_array) * 100
    print(f"\n  Stripe severity: {relative_variation:.2f}% (row std / mean intensity)")
    if relative_variation > 5:
        print(f"  ⚠️  HIGH - Stripes likely visible in DIC results")
    elif relative_variation > 2:
        print(f"  ⚠️  MEDIUM - May affect DIC accuracy")
    else:
        print(f"  ✓ LOW - Unlikely to cause DIC artifacts")

    return {
        'row_means': row_means,
        'row_diffs': row_diffs,
        'relative_variation': relative_variation,
    }


def compare_images(img1, img2, name1="Reference", name2="Current"):
    """Compare two images for relative stripe patterns."""

    # Analyze each image
    result1 = analyze_horizontal_stripes(img1, name1)
    result2 = analyze_horizontal_stripes(img2, name2)

    # Compare row means
    if img1.shape[0] == img2.shape[0]:
        row_diff = result2['row_means'] - result1['row_means']

        print(f"\n{'='*60}")
        print(f"COMPARISON: {name1} vs {name2}")
        print(f"{'='*60}")
        print(f"  Row-wise intensity difference:")
        print(f"    Mean: {np.mean(row_diff):.2f}")
        print(f"    Std: {np.std(row_diff):.2f}")
        print(f"    Range: {np.min(row_diff):.2f} to {np.max(row_diff):.2f}")

        # This difference pattern will show up in DIC
        print(f"\n  This intensity difference pattern will affect DIC:")
        print(f"    If std > 5: Strong stripe artifacts in displacement field")
        print(f"    Current std: {np.std(row_diff):.2f}")

        return row_diff
    else:
        print(f"\n  Cannot compare: different image heights")
        return None


def plot_analysis(img1, img2, name1="Reference", name2="Current", save_path=None):
    """Visualize stripe analysis."""

    row_means1 = np.mean(img1, axis=1)
    row_means2 = np.mean(img2, axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Images
    axes[0, 0].imshow(img1, cmap='gray', aspect='auto')
    axes[0, 0].set_title(f'{name1}')
    axes[0, 0].set_xlabel('x [pixel]')
    axes[0, 0].set_ylabel('y [pixel]')

    axes[0, 1].imshow(img2, cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'{name2}')
    axes[0, 1].set_xlabel('x [pixel]')

    # Difference image
    if img1.shape == img2.shape:
        diff = img2.astype(float) - img1.astype(float)
        vmax = np.percentile(np.abs(diff), 99)
        axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[0, 2].set_title(f'Difference ({name2} - {name1})')
        axes[0, 2].set_xlabel('x [pixel]')

    # Row means
    axes[1, 0].plot(row_means1, range(len(row_means1)), 'b-', linewidth=0.5)
    axes[1, 0].set_xlabel('Mean intensity')
    axes[1, 0].set_ylabel('Row (y)')
    axes[1, 0].set_title(f'{name1} row means')
    axes[1, 0].invert_yaxis()

    axes[1, 1].plot(row_means2, range(len(row_means2)), 'r-', linewidth=0.5)
    axes[1, 1].set_xlabel('Mean intensity')
    axes[1, 1].set_ylabel('Row (y)')
    axes[1, 1].set_title(f'{name2} row means')
    axes[1, 1].invert_yaxis()

    # Difference in row means
    if len(row_means1) == len(row_means2):
        row_diff = row_means2 - row_means1
        axes[1, 2].plot(row_diff, range(len(row_diff)), 'g-', linewidth=0.5)
        axes[1, 2].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Intensity difference')
        axes[1, 2].set_ylabel('Row (y)')
        axes[1, 2].set_title('Row mean difference\n(This affects DIC!)')
        axes[1, 2].invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nPlot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose horizontal stripe artifacts in images",
        epilog="""
This script analyzes RAW IMAGES for stripe patterns that would
affect DIC results. Run this BEFORE DIC analysis to understand
if stripes are in the source images.

Common causes of stripes:
- Camera sensor artifacts (rolling shutter, readout noise)
- Lighting variations (fluorescent lights, LED flicker)
- Motion during exposure
- JPEG compression artifacts
"""
    )
    parser.add_argument("images", nargs='+', help="Image files to analyze")
    parser.add_argument("--plot", action="store_true", help="Show visualization")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")

    args = parser.parse_args()

    # Load images
    images = []
    names = []
    for path in args.images:
        print(f"Loading: {path}")
        img = np.array(Image.open(path).convert('L'))
        images.append(img)
        names.append(path)

    # Analyze each image
    for img, name in zip(images, names):
        analyze_horizontal_stripes(img, name)

    # Compare if two images
    if len(images) == 2:
        compare_images(images[0], images[1], names[0], names[1])

        if args.plot or args.save:
            plot_analysis(images[0], images[1], names[0], names[1], args.save)

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print("If stripes are detected in the raw images:")
    print("  1. Check camera settings (exposure, gain)")
    print("  2. Check lighting (avoid flickering sources)")
    print("  3. Ensure probe is STATIONARY during exposure")
    print("  4. Use RAW format instead of JPEG if possible")
    print("  5. Take both images at the SAME load level")


if __name__ == "__main__":
    main()
