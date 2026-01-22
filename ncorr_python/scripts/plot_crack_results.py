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
    field: str = "relative_v",
    fps: int = 10,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_physical_coords: bool = True,
):
    """
    Create a video showing field heatmaps over all cycles.

    Args:
        results: List of analysis results
        config: Analysis configuration
        output_path: Path for output video file
        field: Field to visualize. Options:
            - "displacement_u": x-displacement [px]
            - "displacement_v": y-displacement [px]
            - "strain_exx": normal strain in x-direction [-]
            - "strain_eyy": normal strain in y-direction [-]
            - "strain_exy": shear strain [-]
            - "relative_v": relative y-displacement over reference distance [px]
        fps: Frames per second
        dpi: Resolution
        vmin: Minimum value for colorbar (auto if None)
        vmax: Maximum value for colorbar (auto if None)
        use_physical_coords: Use mm coordinates instead of pixels
    """
    if not results:
        print("No results to visualize")
        return

    # Field configuration: attribute name, colorbar label, title prefix, colormap
    field_config = {
        "displacement_u": {
            "attr": "displacement_u",
            "label": "x-displacement [px]",
            "title": "Displacement u",
            "cmap": "RdBu_r",
            "symmetric": True,
        },
        "displacement_v": {
            "attr": "displacement_v",
            "label": "y-displacement [px]",
            "title": "Displacement v",
            "cmap": "RdBu_r",
            "symmetric": True,
        },
        "strain_exx": {
            "attr": "strain_exx",
            "label": "Strain $\\varepsilon_{xx}$ [-]",
            "title": "Strain exx",
            "cmap": "RdBu_r",
            "symmetric": True,
        },
        "strain_eyy": {
            "attr": "strain_eyy",
            "label": "Strain $\\varepsilon_{yy}$ [-]",
            "title": "Strain eyy",
            "cmap": "RdBu_r",
            "symmetric": True,
        },
        "strain_exy": {
            "attr": "strain_exy",
            "label": "Strain $\\varepsilon_{xy}$ [-]",
            "title": "Strain exy",
            "cmap": "RdBu_r",
            "symmetric": True,
        },
        "relative_v": {
            "attr": "relative_v",
            "label": "Relative y-displacement [px]",
            "title": f"Relative v (over {config.get('reference_distance_mm', 1.0):.1f} mm)",
            "cmap": "RdBu_r",
            "symmetric": True,
        },
    }

    if field not in field_config:
        print(f"Unknown field '{field}'. Available fields: {list(field_config.keys())}")
        return

    fc = field_config[field]
    attr_name = fc["attr"]

    # Determine color scale from all data
    if vmin is None or vmax is None:
        all_values = []
        for r in results:
            data = getattr(r, attr_name)
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                all_values.extend(valid.flatten())
        if all_values:
            if vmin is None:
                vmin = np.percentile(all_values, 1)
            if vmax is None:
                vmax = np.percentile(all_values, 99)
        else:
            vmin, vmax = -1, 1

    # Use symmetric colorbar if configured and data crosses zero
    if fc["symmetric"] and vmin < 0 < vmax:
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
    initial_data = getattr(results[0], attr_name)
    im = ax.imshow(
        initial_data,
        extent=extent,
        aspect='auto',
        cmap=fc["cmap"],
        vmin=vmin,
        vmax=vmax,
        origin='upper',
    )

    cbar = fig.colorbar(im, ax=ax, label=fc["label"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    title = ax.set_title(f"{fc['title']} - Image 1/{len(results)}")

    def update(frame):
        r = results[frame]
        data = getattr(r, attr_name)
        im.set_array(data)
        title.set_text(f"{fc['title']} - {r.image_name}")
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

    print(f"Saving video ({field}) to {output_path}...")

    if use_ffmpeg:
        # Configure matplotlib to use the found FFmpeg path
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        writer = FFMpegWriter(fps=fps, metadata={'title': f'Crack Analysis - {fc["title"]}'})
    else:
        writer = PillowWriter(fps=fps)

    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Video saved: {output_path}")


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


def create_damage_evolution(
    results: List[CrackAnalysisResult],
    config: dict,
    plot_path: Path,
    data_path: Path,
    dpi: int = 150,
):
    """
    Create damage evolution plot with two y-axes:
    - Left axis: Maximum relative displacement per image (absolute values in pixels)
    - Right axis: Normalized damage ratio (0 to 1) vs. relative image number (0 to 1)

    Also saves the damage ratio array to a numpy file.

    Args:
        results: List of analysis results (sorted by image index)
        config: Analysis configuration
        plot_path: Path for output figure
        data_path: Path for saving damage ratio numpy array
        dpi: Resolution
    """
    if not results or len(results) < 2:
        print("Need at least 2 results for damage evolution analysis")
        return

    # Sort results by image index
    results = sorted(results, key=lambda r: r.image_index)
    n_images = len(results)

    # Compute maximum relative displacement for each image
    # (maximum over all x-positions)
    max_disp_per_image = np.zeros(n_images)
    for i, r in enumerate(results):
        valid_values = r.max_relative_v_per_x[~np.isnan(r.max_relative_v_per_x)]
        if len(valid_values) > 0:
            max_disp_per_image[i] = np.max(np.abs(valid_values))
        else:
            max_disp_per_image[i] = np.nan

    # Overall maximum across all images
    overall_max = np.nanmax(max_disp_per_image)

    # Normalized damage ratio: max_disp(image_i) / max_disp(all_images)
    damage_ratio = max_disp_per_image / overall_max

    # Relative image number (0 to 1)
    image_numbers = np.arange(1, n_images + 1)
    relative_image_number = (image_numbers - 1) / (n_images - 1)  # 0 to 1

    # Save damage ratio data
    np.save(data_path, {
        'image_numbers': image_numbers,
        'relative_image_number': relative_image_number,
        'max_displacement_per_image': max_disp_per_image,
        'damage_ratio': damage_ratio,
        'overall_max_displacement': overall_max,
    })
    print(f"Damage ratio data saved: {data_path}")

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot 1 (left axis): Max displacement vs image number
    color1 = 'tab:blue'
    line1 = ax1.plot(image_numbers, max_disp_per_image, 'o-', color=color1,
                     markersize=4, linewidth=1.5, label='Max. rel. displacement')
    ax1.set_xlabel('Image number')
    ax1.set_ylabel('Max. relative displacement [px]', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(0, n_images + 1)
    ax1.set_ylim(bottom=0)

    # Plot 2 (right axis): Damage ratio vs relative image number
    # We create a secondary x-axis at the top for relative image number
    ax_top = ax1.secondary_xaxis('top', functions=(
        lambda x: (x - 1) / (n_images - 1) if n_images > 1 else 0,
        lambda x: x * (n_images - 1) + 1
    ))
    ax_top.set_xlabel('Relative image number')

    color2 = 'tab:red'
    line2 = ax2.plot(image_numbers, damage_ratio, 's--', color=color2,
                     markersize=4, linewidth=1.5, label='Damage ratio')
    ax2.set_ylabel('Damage ratio (normalized)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')

    # Grid on primary axis only
    ax1.grid(True, alpha=0.3)

    ref_distance = config.get('reference_distance_mm', 1.0)
    plt.title(f'Damage Evolution\n(relative displacement over {ref_distance:.1f} mm)')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    plt.close(fig)
    print(f"Damage evolution plot saved: {plot_path}")


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
        "--no-video",
        action="store_true",
        help="Skip video generation"
    )
    parser.add_argument(
        "--video-field",
        type=str,
        choices=["displacement_u", "displacement_v", "strain_exx", "strain_eyy", "strain_exy", "relative_v"],
        default="relative_v",
        help="Field to show in video: displacement_u/v, strain_exx/eyy/exy, relative_v (default: relative_v)"
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

    # 1. Refined crack analysis (three-panel comparison)
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

    # 2. Damage evolution (max displacement + normalized ratio)
    create_damage_evolution(
        results, config,
        output_dir / "damage_evolution.png",
        output_dir / "damage_ratio.npy",
        dpi=args.dpi,
    )

    # 3. Heatmap video (optional)
    if not args.no_video:
        video_filename = f"{args.video_field}.mp4"
        create_heatmap_video(
            results, config,
            output_dir / video_filename,
            field=args.video_field,
            fps=args.video_fps,
            dpi=args.dpi,
            use_physical_coords=use_physical,
        )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
