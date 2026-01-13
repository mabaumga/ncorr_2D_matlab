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
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from tqdm import tqdm

# Import result class
import sys
sys.path.insert(0, str(Path(__file__).parent))
from batch_crack_analysis import CrackAnalysisResult, AnalysisConfig


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
    print(f"Saving video to {output_path}...")

    # Try to save as MP4 first, fall back to GIF if FFmpeg not available
    saved = False

    if output_path.suffix.lower() != '.gif':
        try:
            writer = FFMpegWriter(fps=fps, metadata={'title': 'Crack Analysis'})
            anim.save(output_path, writer=writer, dpi=dpi)
            saved = True
        except (FileNotFoundError, OSError) as e:
            print(f"FFmpeg not available ({e}), falling back to GIF...")
            output_path = output_path.with_suffix('.gif')

    if not saved:
        # Use GIF format
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Video saved: {output_path}")


def create_displacement_curves(
    results: List[CrackAnalysisResult],
    config: dict,
    output_path: Path,
    x_position_mm: Optional[float] = None,
    x_position_idx: Optional[int] = None,
    dpi: int = 150,
    use_physical_coords: bool = True,
):
    """
    Create plot of relative displacement vs y-coordinate for all images.

    Each curve represents one image, with colors from viridis colormap.

    Args:
        results: List of analysis results
        config: Analysis configuration
        output_path: Path for output figure
        x_position_mm: x-position to extract curve (in mm, uses center if None)
        x_position_idx: x-position index to extract curve (alternative to mm)
        dpi: Resolution
        use_physical_coords: Use mm coordinates instead of pixels
    """
    if not results:
        print("No results to visualize")
        return

    # Determine x position to use
    r0 = results[0]
    nx = len(r0.grid_x)

    if x_position_idx is not None:
        x_idx = x_position_idx
    elif x_position_mm is not None:
        # Find closest x position
        x_idx = np.argmin(np.abs(r0.grid_x_mm - x_position_mm))
    else:
        # Use center
        x_idx = nx // 2

    x_value = r0.grid_x_mm[x_idx] if use_physical_coords else r0.grid_x[x_idx]

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map for curves
    n_curves = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_curves))

    # Plot each curve
    for i, r in enumerate(results):
        if use_physical_coords:
            y_coords = r.grid_y_mm
        else:
            y_coords = r.grid_y

        # Extract column at x position
        rel_v_col = r.relative_v[:, x_idx]

        # Plot with color from viridis
        ax.plot(
            y_coords, rel_v_col,
            color=colors[i],
            alpha=0.7,
            linewidth=0.8,
            label=f"Image {r.image_index}" if i % max(1, n_curves // 10) == 0 else None
        )

    # Labels
    ref_distance = config.get('reference_distance_mm', 1.0)
    coord_unit = "mm" if use_physical_coords else "px"

    ax.set_xlabel(f"y [{coord_unit}]")
    ax.set_ylabel(f"Relative y-displacement [px]")
    ax.set_title(f"Relative v (over {ref_distance:.1f} mm) at x = {x_value:.1f} {coord_unit}")

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
        "--x-position",
        type=float,
        default=None,
        help="x-position in mm for displacement curves (default: center)"
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

    # 1. Displacement curves
    create_displacement_curves(
        results, config,
        output_dir / "displacement_curves.png",
        x_position_mm=args.x_position,
        dpi=args.dpi,
        use_physical_coords=use_physical,
    )

    # 2. Max displacement evolution
    create_max_displacement_evolution(
        results, config,
        output_dir / "max_displacement_evolution.png",
        dpi=args.dpi,
        use_physical_coords=use_physical,
    )

    # 3. Crack position plot
    create_crack_position_plot(
        results, config,
        output_dir / "crack_propagation.png",
        threshold=args.crack_threshold,
        dpi=args.dpi,
        use_physical_coords=use_physical,
    )

    # 4. Heatmap video (optional)
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
