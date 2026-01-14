#!/usr/bin/env python3
"""
Visualization of crack analysis results.

This script creates visualizations from batch crack analysis results:
1. Heatmap video of relative y-displacement over all load cycles
2. Relative displacement vs y-coordinate curves for all images

Usage:
    python plot_crack_results.py /path/to/results
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm

# Import result class
import sys
sys.path.insert(0, str(Path(__file__).parent))
from batch_crack_analysis import CrackAnalysisResult, AnalysisConfig


def get_ffmpeg_path() -> Optional[str]:
    """Get FFmpeg executable path, checking PATH and imageio-ffmpeg."""
    # First check system PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path

    # Try imageio-ffmpeg package
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path:
            return ffmpeg_path
    except ImportError:
        pass

    return None


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return get_ffmpeg_path() is not None


def _isotonic_regression_pava(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm for isotonic regression.

    Enforces monotonically increasing values.

    Args:
        y: Input array

    Returns:
        Monotonically increasing array
    """
    n = len(y)
    y_iso = y.copy().astype(float)

    # Forward pass: merge violations
    i = 0
    while i < n - 1:
        if y_iso[i] > y_iso[i + 1]:
            # Found a violation - pool and average
            j = i + 1
            while j < n and y_iso[i] > y_iso[j]:
                j += 1
            # Average the pooled region
            avg = np.mean(y_iso[i:j])
            y_iso[i:j] = avg
            # Go back to check if new average creates violation
            if i > 0:
                i -= 1
        else:
            i += 1

    return y_iso


def load_results(results_dir: Path) -> Tuple[List[CrackAnalysisResult], dict]:
    """
    Load all results from a results directory.

    Args:
        results_dir: Directory containing result .npz files

    Returns:
        Tuple of (list of results sorted by index, config dict)
    """
    results_dir = Path(results_dir)

    # Load config
    config_path = results_dir / "analysis_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Load all result files
    result_files = sorted(results_dir.glob("*_result.npz"))
    results = []

    for f in tqdm(result_files, desc="Loading results"):
        try:
            result = CrackAnalysisResult.load(f)
            results.append(result)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    # Sort by image index
    results.sort(key=lambda r: r.image_index)

    return results, config


def create_heatmap_video(
    results: List[CrackAnalysisResult],
    config: dict,
    output_path: Path,
    fps: int = 10,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_physical_coords: bool = True,
):
    """
    Create a video showing relative y-displacement heatmaps over all cycles.

    Args:
        results: List of analysis results
        config: Analysis configuration
        output_path: Path for output video file
        fps: Frames per second
        dpi: Resolution
        vmin: Minimum value for colorbar (auto if None)
        vmax: Maximum value for colorbar (auto if None)
        use_physical_coords: Use mm coordinates instead of pixels
    """
    if not results:
        print("No results to visualize")
        return

    # Determine color scale from all data
    if vmin is None or vmax is None:
        all_values = []
        for r in results:
            valid = r.relative_v[~np.isnan(r.relative_v)]
            if len(valid) > 0:
                all_values.extend(valid.flatten())
        if all_values:
            if vmin is None:
                vmin = np.percentile(all_values, 1)
            if vmax is None:
                vmax = np.percentile(all_values, 99)
        else:
            vmin, vmax = -1, 1

    # Use symmetric colorbar if data crosses zero
    if vmin < 0 < vmax:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get coordinate arrays from first result
    r0 = results[0]
    if use_physical_coords:
        extent = [
            r0.grid_x_mm[0], r0.grid_x_mm[-1],
            r0.grid_y_mm[-1], r0.grid_y_mm[0]  # Flip y for image orientation
        ]
        xlabel = "x [mm]"
        ylabel = "y [mm]"
    else:
        extent = [
            r0.grid_x[0], r0.grid_x[-1],
            r0.grid_y[-1], r0.grid_y[0]
        ]
        xlabel = "x [px]"
        ylabel = "y [px]"

    # Initial heatmap
    im = ax.imshow(
        results[0].relative_v,
        extent=extent,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        origin='upper',
    )

    cbar = fig.colorbar(im, ax=ax, label="Relative y-displacement [px]")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ref_distance = config.get('reference_distance_mm', 1.0)
    title = ax.set_title(f"Relative v (over {ref_distance:.1f} mm) - Image 1/{len(results)}")

    def update(frame):
        r = results[frame]
        im.set_array(r.relative_v)
        title.set_text(f"Relative v (over {ref_distance:.1f} mm) - {r.image_name}")
        return [im, title]

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(results),
        interval=1000/fps, blit=True
    )

    # Save video
    output_path = Path(output_path)

    # Check if FFmpeg is available for MP4 output
    ffmpeg_path = get_ffmpeg_path()
    use_ffmpeg = output_path.suffix.lower() != '.gif' and ffmpeg_path is not None

    if output_path.suffix.lower() != '.gif' and ffmpeg_path is None:
        print("FFmpeg not found. Falling back to GIF format.")
        print("To enable MP4: pip install imageio-ffmpeg")
        output_path = output_path.with_suffix('.gif')

    print(f"Saving video to {output_path}...")

    if use_ffmpeg:
        # Configure matplotlib to use the found FFmpeg path
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        writer = FFMpegWriter(fps=fps, metadata={'title': 'Crack Analysis'})
    else:
        writer = PillowWriter(fps=fps)

    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Video saved: {output_path}")


def create_displacement_curves(
    results: List[CrackAnalysisResult],
    config: dict,
    output_path: Path,
    dpi: int = 150,
    use_physical_coords: bool = True,
):
    """
    Create plot of maximum relative displacement vs x-coordinate for all images.

    Each curve represents one image, with colors from viridis colormap.
    Shows the maximum relative y-displacement at each x-position (crack opening).

    Args:
        results: List of analysis results
        config: Analysis configuration
        output_path: Path for output figure
        dpi: Resolution
        use_physical_coords: Use mm coordinates instead of pixels
    """
    if not results:
        print("No results to visualize")
        return

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map for curves
    n_curves = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_curves))

    # Plot each curve - max relative displacement vs x
    for i, r in enumerate(results):
        if use_physical_coords:
            x_coords = r.grid_x_mm
        else:
            x_coords = r.grid_x

        # Use the pre-computed max relative displacement per x-position
        max_rel_v = r.max_relative_v_per_x

        # Plot with color from viridis
        ax.plot(
            x_coords, max_rel_v,
            color=colors[i],
            alpha=0.7,
            linewidth=0.8,
            label=f"Image {r.image_index}" if i % max(1, n_curves // 10) == 0 else None
        )

    # Labels
    ref_distance = config.get('reference_distance_mm', 1.0)
    coord_unit = "mm" if use_physical_coords else "px"

    ax.set_xlabel(f"x [{coord_unit}]")
    ax.set_ylabel(f"Max. relative y-displacement [px]")
    ax.set_title(f"Maximum relative y-displacement (over {ref_distance:.1f} mm) vs x-position")

    # Colorbar for image progression
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=mcolors.Normalize(vmin=1, vmax=n_curves)
    )
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Image number")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Displacement curves saved: {output_path}")


def create_refined_crack_analysis(
    results: List[CrackAnalysisResult],
    config: dict,
    output_path: Path,
    crack_margin_mm: float = 2.0,
    sigma_x: float = 3.0,
    sigma_time: float = 2.0,
    smoothing_method: str = "gaussian",
    dpi: int = 150,
    use_physical_coords: bool = True,
    title: Optional[str] = None,
):
    """
    Create refined crack analysis with three panels:
    1. Raw data (max relative displacement vs x)
    2. Refined ROI (focused on crack region from final image)
    3. Smoothed data (Gaussian or isotonic regression)

    Args:
        results: List of analysis results (sorted by image index)
        config: Analysis configuration
        output_path: Path for output figure
        crack_margin_mm: Margin around detected crack region in mm
        sigma_x: Gaussian smoothing sigma along x-axis (in grid points)
        sigma_time: Gaussian smoothing sigma along time axis (in images)
        smoothing_method: "gaussian" or "isotonic"
        dpi: Resolution
        use_physical_coords: Use mm coordinates instead of pixels
        title: Custom title (default: parent directory name)
    """
    if not results or len(results) < 2:
        print("Need at least 2 results for refined analysis")
        return

    # Sort results by image index
    results = sorted(results, key=lambda r: r.image_index)
    n_images = len(results)

    # Get coordinates - always use PIXELS for x-axis display
    r0 = results[0]
    x_coords = r0.grid_x  # Always pixels
    y_coords = r0.grid_y  # Always pixels

    # Calculate crack margin in grid indices
    pixels_per_mm = config.get('pixels_per_mm', 50.0)
    grid_step = config.get('grid_step', 4)
    crack_margin_idx = int(crack_margin_mm * pixels_per_mm / grid_step)

    nx = len(x_coords)
    ny = len(y_coords)
    ref_distance = config.get('reference_distance_mm', 1.0)

    # Determine title: use provided title, or parent directory name
    output_path = Path(output_path)
    if title is None:
        # Get parent directory of the results folder
        # e.g., "FN3_2_75-86/results/file.png" -> "FN3_2_75-86"
        parent_dir = output_path.parent
        if parent_dir.name in ('results', 'output', 'ergebnisse'):
            title = parent_dir.parent.name
        else:
            title = parent_dir.name

    # =========================================================================
    # Step 1: Build raw data matrix (n_images x nx)
    # =========================================================================
    raw_data = np.zeros((n_images, nx))
    for i, r in enumerate(results):
        raw_data[i, :] = r.max_relative_v_per_x

    # =========================================================================
    # Step 2: Identify crack region from LAST image (largest crack)
    # =========================================================================
    last_result = results[-1]
    rel_v_last = last_result.relative_v  # 2D array (ny, nx)

    # Find y-position of max displacement for each x in last image
    crack_y_positions = last_result.max_relative_v_y_position  # in pixels

    # Determine crack region bounds (y indices where crack is detected)
    valid_crack_y = crack_y_positions[~np.isnan(crack_y_positions)]
    if len(valid_crack_y) == 0:
        print("No valid crack positions detected in last image")
        return

    # Convert to grid indices
    grid_step = config.get('grid_step', 4)
    crack_y_min_px = np.nanmin(valid_crack_y)
    crack_y_max_px = np.nanmax(valid_crack_y)

    # Convert to y-index in grid
    crack_y_min_idx = max(0, int(crack_y_min_px / grid_step) - crack_margin_idx)
    crack_y_max_idx = min(ny - 1, int(crack_y_max_px / grid_step) + crack_margin_idx)

    print(f"Detected crack region: y = {y_coords[crack_y_min_idx]:.0f} to {y_coords[crack_y_max_idx]:.0f} px")

    # =========================================================================
    # Step 3: Compute refined data (average in crack region instead of max)
    # =========================================================================
    refined_data = np.zeros((n_images, nx))

    for i, r in enumerate(results):
        rel_v = r.relative_v  # 2D array (ny, nx)
        # Extract crack region and take mean of absolute values
        crack_region = rel_v[crack_y_min_idx:crack_y_max_idx+1, :]
        # Use max in the crack region (more robust than mean for crack detection)
        with np.errstate(all='ignore'):
            refined_data[i, :] = np.nanmax(np.abs(crack_region), axis=0)

    # =========================================================================
    # Step 4: Apply smoothing (Gaussian or Isotonic)
    # =========================================================================
    smoothed_data = refined_data.copy()
    nan_mask = np.isnan(smoothed_data)

    if smoothing_method == "isotonic":
        # Isotonic regression: enforce monotonic increase over time
        # This is physically motivated - crack opening should only grow
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(increasing=True)
        except ImportError:
            print("sklearn not installed. Falling back to manual isotonic regression.")
            ir = None

        for j in range(nx):
            col = smoothed_data[:, j]
            valid = ~np.isnan(col)
            if np.sum(valid) < 2:
                continue

            x_valid = np.where(valid)[0]
            y_valid = col[valid]

            if ir is not None:
                # Use sklearn
                y_isotonic = ir.fit_transform(x_valid, y_valid)
            else:
                # Manual Pool Adjacent Violators Algorithm (PAVA)
                y_isotonic = _isotonic_regression_pava(y_valid)

            col[valid] = y_isotonic
            smoothed_data[:, j] = col

        # Apply spatial smoothing along x-axis only
        if sigma_x > 0:
            for i in range(n_images):
                row = smoothed_data[i, :]
                valid = ~np.isnan(row)
                if np.sum(valid) > 3:
                    row[valid] = gaussian_filter1d(row[valid], sigma=sigma_x)
                    smoothed_data[i, :] = row

        smoothing_label = "isotonic + σ_x={:.1f}".format(sigma_x)

    else:  # Gaussian smoothing
        # Replace NaN with interpolated values for smoothing
        if np.any(nan_mask):
            col_means = np.nanmean(smoothed_data, axis=0)
            for j in range(nx):
                smoothed_data[nan_mask[:, j], j] = col_means[j]

        # Apply 2D Gaussian filter (sigma_time along axis 0, sigma_x along axis 1)
        smoothed_data = gaussian_filter(smoothed_data, sigma=[sigma_time, sigma_x])

        # Restore NaN where original was NaN
        smoothed_data[nan_mask] = np.nan

        smoothing_label = f"σ_x={sigma_x:.1f}, σ_t={sigma_time:.1f}"

    # Physical constraint: first image should have ~zero displacement
    # Subtract baseline (small offset from first few images)
    baseline = np.nanmean(smoothed_data[:3, :], axis=0)
    smoothed_data = smoothed_data - baseline[np.newaxis, :]
    smoothed_data = np.maximum(smoothed_data, 0)  # Crack opening is positive

    # =========================================================================
    # Step 5: Create three-panel figure
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Color settings
    n_curves = n_images
    colors = plt.cm.viridis(np.linspace(0, 1, n_curves))

    # Panel 1: Raw data
    ax1 = axes[0]
    for i in range(n_images):
        ax1.plot(x_coords, raw_data[i, :], color=colors[i], alpha=0.7, linewidth=0.8)
    ax1.set_xlabel("x [px]")
    ax1.set_ylabel("Max. relative y-displacement [px]")
    ax1.set_title("Raw data\n(max over all y)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Refined (crack region only)
    ax2 = axes[1]
    for i in range(n_images):
        ax2.plot(x_coords, refined_data[i, :], color=colors[i], alpha=0.7, linewidth=0.8)
    ax2.set_xlabel("x [px]")
    ax2.set_ylabel("Max. relative y-displacement [px]")
    ax2.set_title(f"Refined ROI\n(crack region: y = {y_coords[crack_y_min_idx]:.0f}-{y_coords[crack_y_max_idx]:.0f} px)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Smoothed data
    ax3 = axes[2]
    for i in range(n_images):
        ax3.plot(x_coords, smoothed_data[i, :], color=colors[i], alpha=0.7, linewidth=0.8)
    ax3.set_xlabel("x [px]")
    ax3.set_ylabel("Smoothed relative y-displacement [px]")
    ax3.set_title(f"Smoothed data\n({smoothing_label}, baseline corrected)")
    ax3.grid(True, alpha=0.3)

    # Add colorbar for image progression - placed outside on the right
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mcolors.Normalize(vmin=1, vmax=n_curves))
    sm.set_array([])  # Required for ScalarMappable
    # Create space for colorbar on the right side
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Image number")

    plt.suptitle(f"{title}", fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Refined crack analysis saved: {output_path}")

    return {
        'raw_data': raw_data,
        'refined_data': refined_data,
        'smoothed_data': smoothed_data,
        'crack_y_range': (crack_y_min_idx, crack_y_max_idx),
        'x_coords': x_coords,
    }


def create_max_displacement_evolution(
    results: List[CrackAnalysisResult],
    config: dict,
    output_path: Path,
    dpi: int = 150,
    use_physical_coords: bool = True,
):
    """
    Create plot showing evolution of maximum relative displacement per x-position.

    Args:
        results: List of analysis results
        config: Analysis configuration
        output_path: Path for output figure
        dpi: Resolution
        use_physical_coords: Use mm coordinates instead of pixels
    """
    if not results:
        print("No results to visualize")
        return

    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    r0 = results[0]
    n_curves = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_curves))

    # Plot max relative displacement vs x for each image
    for i, r in enumerate(results):
        if use_physical_coords:
            x_coords = r.grid_x_mm
            y_pos = r.max_relative_v_y_position / config.get('pixels_per_mm', 50.0)
        else:
            x_coords = r.grid_x
            y_pos = r.max_relative_v_y_position

        ax1.plot(x_coords, r.max_relative_v_per_x, color=colors[i], alpha=0.7, linewidth=0.8)
        ax2.plot(x_coords, y_pos, color=colors[i], alpha=0.7, linewidth=0.8)

    # Labels
    coord_unit = "mm" if use_physical_coords else "px"
    ref_distance = config.get('reference_distance_mm', 1.0)

    ax1.set_ylabel(f"Max relative v [px]")
    ax1.set_title(f"Maximum relative y-displacement (over {ref_distance:.1f} mm) per x-position")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(f"x [{coord_unit}]")
    ax2.set_ylabel(f"y-position of max [{coord_unit}]")
    ax2.set_title("y-position of maximum relative displacement (crack location)")
    ax2.grid(True, alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=mcolors.Normalize(vmin=1, vmax=n_curves)
    )
    cbar = fig.colorbar(sm, ax=[ax1, ax2], location='right', pad=0.02)
    cbar.set_label("Image number")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Max displacement evolution saved: {output_path}")


def create_crack_position_plot(
    results: List[CrackAnalysisResult],
    config: dict,
    output_path: Path,
    threshold: float = 0.5,
    dpi: int = 150,
    use_physical_coords: bool = True,
):
    """
    Create plot showing crack tip position over cycles.

    The crack tip is estimated as the leftmost x-position where
    max relative displacement exceeds a threshold.

    Args:
        results: List of analysis results
        config: Analysis configuration
        output_path: Path for output figure
        threshold: Threshold for crack detection (in pixels)
        dpi: Resolution
        use_physical_coords: Use mm coordinates instead of pixels
    """
    if not results:
        print("No results to visualize")
        return

    r0 = results[0]
    pixels_per_mm = config.get('pixels_per_mm', 50.0)

    # Find crack tip for each image
    image_indices = []
    crack_tips = []

    for r in results:
        if use_physical_coords:
            x_coords = r.grid_x_mm
        else:
            x_coords = r.grid_x

        # Find where max relative displacement exceeds threshold
        exceeds = np.abs(r.max_relative_v_per_x) > threshold
        valid = ~np.isnan(r.max_relative_v_per_x)
        mask = exceeds & valid

        if np.any(mask):
            # Leftmost position exceeding threshold
            crack_tip_idx = np.where(mask)[0][0]
            crack_tips.append(x_coords[crack_tip_idx])
            image_indices.append(r.image_index)

    if not crack_tips:
        print("No crack detected in any image")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(image_indices, crack_tips, 'o-', markersize=4)

    coord_unit = "mm" if use_physical_coords else "px"
    ax.set_xlabel("Image number")
    ax.set_ylabel(f"Crack tip x-position [{coord_unit}]")
    ax.set_title(f"Crack propagation (threshold = {threshold:.2f} px)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Crack position plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualization of crack analysis results."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing analysis results"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for plots (default: results_dir)"
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for video (default: 10)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for plots (default: 150)"
    )
    parser.add_argument(
        "--crack-threshold",
        type=float,
        default=0.5,
        help="Threshold for crack detection in pixels (default: 0.5)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video generation"
    )
    parser.add_argument(
        "--pixels",
        action="store_true",
        help="Use pixel coordinates instead of mm"
    )
    parser.add_argument(
        "--crack-margin",
        type=float,
        default=2.0,
        help="Margin around crack region in mm (default: 2.0)"
    )
    parser.add_argument(
        "--sigma-x",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma along x-axis (default: 3.0)"
    )
    parser.add_argument(
        "--sigma-time",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma along time axis (default: 2.0)"
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        choices=["gaussian", "isotonic"],
        default="gaussian",
        help="Smoothing method: 'gaussian' (2D filter) or 'isotonic' (monotonic increase over time)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for plots (default: parent directory name)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = args.output if args.output else results_dir

    # Load results
    print(f"Loading results from {results_dir}...")
    results, config = load_results(results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} results")

    use_physical = not args.pixels

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    # 1. Displacement curves (max relative displacement vs x)
    create_displacement_curves(
        results, config,
        output_dir / "displacement_curves.png",
        dpi=args.dpi,
        use_physical_coords=use_physical,
    )

    # 2. Refined crack analysis (three-panel comparison)
    create_refined_crack_analysis(
        results, config,
        output_dir / "refined_crack_analysis.png",
        crack_margin_mm=args.crack_margin,
        sigma_x=args.sigma_x,
        sigma_time=args.sigma_time,
        smoothing_method=args.smoothing,
        dpi=args.dpi,
        use_physical_coords=use_physical,
        title=args.title,
    )

    # 3. Max displacement evolution
    create_max_displacement_evolution(
        results, config,
        output_dir / "max_displacement_evolution.png",
        dpi=args.dpi,
        use_physical_coords=use_physical,
    )

    # 4. Crack position plot
    create_crack_position_plot(
        results, config,
        output_dir / "crack_propagation.png",
        threshold=args.crack_threshold,
        dpi=args.dpi,
        use_physical_coords=use_physical,
    )

    # 5. Heatmap video (optional)
    if not args.no_video:
        create_heatmap_video(
            results, config,
            output_dir / "relative_displacement.mp4",
            fps=args.video_fps,
            dpi=args.dpi,
            use_physical_coords=use_physical,
        )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
