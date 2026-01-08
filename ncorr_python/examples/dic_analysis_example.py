"""
Example script for DIC analysis with Ncorr Python.

This example demonstrates:
1. Loading reference and current images
2. Setting up ROI and parameters
3. Running DIC analysis
4. Saving and loading results
5. Visualizing displacements and strains
6. Computing statistics

Usage:
    python dic_analysis_example.py --ref path/to/reference.jpg --cur path/to/current.jpg

Or modify the paths in the script directly.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ncorr import DICParameters
from ncorr.main import Ncorr
from ncorr.core.image import NcorrImage
from ncorr.algorithms.dic import SeedInfo


def run_example(ref_path: str = None, cur_path: str = None):
    """
    Run DIC analysis example.

    Args:
        ref_path: Path to reference image (optional, uses synthetic data if None)
        cur_path: Path to current/deformed image (optional, uses synthetic data if None)
    """

    # === 1. Load images ===
    if ref_path and cur_path:
        # Load from files
        ref_img = NcorrImage.from_file(ref_path)
        cur_img = NcorrImage.from_file(cur_path)
        print(f"Reference image: {ref_img.width} x {ref_img.height} pixels")
        print(f"Current image: {cur_img.width} x {cur_img.height} pixels")
    else:
        # Create synthetic speckle pattern for testing
        print("No images provided - creating synthetic speckle pattern...")
        np.random.seed(42)

        h, w = 200, 300

        # Create base speckle pattern
        ref_data = np.random.rand(h, w).astype(np.float64)
        # Apply gaussian blur for realistic speckles
        from scipy.ndimage import gaussian_filter
        ref_data = gaussian_filter(ref_data, sigma=2.0)
        ref_data = (ref_data - ref_data.min()) / (ref_data.max() - ref_data.min())

        ref_img = NcorrImage.from_array((ref_data * 255).astype(np.uint8))
        print(f"Synthetic reference: {ref_img.width} x {ref_img.height} pixels")

        # Create deformed image (5 pixel shift in x direction)
        cur_data = np.zeros_like(ref_data)
        shift_x = 5
        cur_data[:, :-shift_x] = ref_data[:, shift_x:]

        cur_img = NcorrImage.from_array((cur_data * 255).astype(np.uint8))
        print(f"Synthetic current (shifted {shift_x}px): {cur_img.width} x {cur_img.height} pixels")

    # === 2. Initialize Ncorr ===
    ncorr = Ncorr()

    def progress_callback(progress, message):
        print(f"[{progress*100:.0f}%] {message}")

    ncorr.set_progress_callback(progress_callback)

    ncorr.set_reference(ref_img)
    ncorr.set_current(cur_img)

    # === 3. Define ROI ===
    h, w = ref_img.height, ref_img.width
    mask = np.zeros((h, w), dtype=bool)

    # Use entire image as ROI (or define specific region)
    # mask[50:1650, 50:5950] = True  # Example of specific region
    mask[:, :] = True

    ncorr.set_roi_from_mask(mask)

    # === 4. Configure parameters ===
    params = DICParameters(
        radius=26,           # Subset radius in pixels
        spacing=5,           # Spacing between analysis points
        cutoff_diffnorm=1e-3,  # Convergence threshold
        cutoff_iteration=20,   # Maximum iterations
    )
    ncorr.set_parameters(params)

    print(f"Number of ROI regions: {len(ncorr.roi.regions)}")

    # Calculate seeds automatically
    ncorr.calculate_seeds()
    print(f"Number of seeds: {len(ncorr.seeds)}")
    for s in ncorr.seeds:
        print(f"  Seed: x={s.x}, y={s.y}, region={s.region_idx}")

    # Optionally, set a single seed manually
    # single_seed = SeedInfo(
    #     x=w // 2,
    #     y=h // 2,
    #     u=0.0,
    #     v=0.0,
    #     region_idx=0,
    #     valid=True
    # )
    # ncorr._seeds = [single_seed]

    # === 5. Run analysis ===
    print("\nStarting DIC analysis...")
    results = ncorr.run_analysis()
    print("Analysis complete!")

    # === 5b. Save results ===
    out_base = Path("dic_results")
    results.save(out_base)
    print(f"Results saved as: {out_base.with_suffix('.npz')} and {out_base.with_suffix('.json')}")

    # === 5c. Load results (demonstration) ===
    # from ncorr.main import AnalysisResults
    # loaded_results = AnalysisResults.load("dic_results")

    # === 6. Visualize results ===
    disp = results.displacements[0]
    strain_ref = results.strains_ref[0]

    # 4 rows, 1 column layout
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))

    # Reference image
    axes[0].imshow(ref_img.get_gs(), cmap='gray')
    axes[0].set_title('Reference Image')
    axes[0].axis('off')

    # u-displacement (x direction)
    im1 = axes[1].imshow(disp.u, cmap='jet')
    axes[1].set_title('u-Displacement (pixels)')
    plt.colorbar(im1, ax=axes[1])

    # v-displacement (y direction)
    im2 = axes[2].imshow(disp.v, cmap='jet')
    axes[2].set_title('v-Displacement (pixels)')
    plt.colorbar(im2, ax=axes[2])

    # Correlation coefficient
    im3 = axes[3].imshow(disp.corrcoef, cmap='hot', vmin=0.9, vmax=1.0)
    axes[3].set_title('Correlation Coefficient')
    plt.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.savefig('dic_results_displacement.png', dpi=150)
    print("Saved: dic_results_displacement.png")

    # Strain visualization
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    im_exx = axes2[0, 0].imshow(strain_ref.exx, cmap='coolwarm')
    axes2[0, 0].set_title('εxx (Green-Lagrange)')
    plt.colorbar(im_exx, ax=axes2[0, 0])

    im_eyy = axes2[0, 1].imshow(strain_ref.eyy, cmap='coolwarm')
    axes2[0, 1].set_title('εyy (Green-Lagrange)')
    plt.colorbar(im_eyy, ax=axes2[0, 1])

    im_exy = axes2[1, 0].imshow(strain_ref.exy, cmap='coolwarm')
    axes2[1, 0].set_title('εxy (Green-Lagrange)')
    plt.colorbar(im_exy, ax=axes2[1, 0])

    # ROI mask
    axes2[1, 1].imshow(disp.roi, cmap='gray')
    axes2[1, 1].set_title('Analysis ROI')

    plt.tight_layout()
    plt.savefig('dic_results_strain.png', dpi=150)
    print("Saved: dic_results_strain.png")

    # === 7. Print statistics ===
    valid = disp.roi
    print("\n=== Result Statistics ===")
    print(f"Analyzed points: {np.sum(valid)}")
    print(f"\nDisplacements:")
    print(f"  u: {np.nanmean(disp.u[valid]):.4f} ± {np.nanstd(disp.u[valid]):.4f} pixels")
    print(f"  v: {np.nanmean(disp.v[valid]):.4f} ± {np.nanstd(disp.v[valid]):.4f} pixels")
    print(f"\nStrains:")
    print(f"  εxx: {np.nanmean(strain_ref.exx[strain_ref.roi]):.6f}")
    print(f"  εyy: {np.nanmean(strain_ref.eyy[strain_ref.roi]):.6f}")
    print(f"  εxy: {np.nanmean(strain_ref.exy[strain_ref.roi]):.6f}")
    print(f"\nCorrelation:")
    print(f"  Mean: {np.nanmean(disp.corrcoef[valid]):.4f}")
    print(f"  Min:  {np.nanmin(disp.corrcoef[valid]):.4f}")

    # Show plots
    plt.show()

    return results


def run_with_profiling():
    """Run example with cProfile for performance analysis."""
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    run_example()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    print("\n=== Profiling Results (Top 40) ===")
    stats.print_stats(40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DIC analysis example with Ncorr Python"
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="Path to reference image"
    )
    parser.add_argument(
        "--cur",
        type=str,
        help="Path to current/deformed image"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with profiling enabled"
    )

    args = parser.parse_args()

    if args.profile:
        run_with_profiling()
    else:
        run_example(args.ref, args.cur)
