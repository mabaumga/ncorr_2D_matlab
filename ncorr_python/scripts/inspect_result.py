#!/usr/bin/env python3
"""
Inspect and visualize a single DIC result file.

This script loads a .npz result file and displays all available fields,
helping to diagnose issues with the analysis.

Usage:
    python inspect_result.py /path/to/result.npz
    python inspect_result.py /path/to/result.npz --save output.png
    python inspect_result.py /path/to/result.npz --image-dir /path/to/images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from batch_crack_analysis import CrackAnalysisResult


def print_field_stats(name: str, data: np.ndarray, is_strain: bool = False) -> dict:
    """Print statistics for a data field and return them."""
    if data is None:
        print(f"  {name}: None")
        return {}

    stats = {
        'shape': data.shape,
        'dtype': str(data.dtype),
    }

    if np.issubdtype(data.dtype, np.number):
        valid_data = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data.flatten()
        stats['total_elements'] = data.size
        stats['valid_elements'] = len(valid_data)
        stats['nan_count'] = data.size - len(valid_data)
        stats['nan_percent'] = 100 * stats['nan_count'] / data.size if data.size > 0 else 0

        if len(valid_data) > 0:
            stats['min'] = float(np.min(valid_data))
            stats['max'] = float(np.max(valid_data))
            stats['mean'] = float(np.mean(valid_data))
            stats['std'] = float(np.std(valid_data))
            stats['abs_max'] = float(np.max(np.abs(valid_data)))
        else:
            stats['min'] = stats['max'] = stats['mean'] = stats['std'] = stats['abs_max'] = np.nan

    unit = "%" if is_strain else ""
    multiplier = 100 if is_strain else 1

    print(f"  {name}:")
    print(f"    Shape: {stats['shape']}, dtype: {stats['dtype']}")
    if 'valid_elements' in stats:
        print(f"    Valid: {stats['valid_elements']}/{stats['total_elements']} ({100-stats['nan_percent']:.1f}%)")
        if stats['valid_elements'] > 0:
            print(f"    Range: [{stats['min']*multiplier:.6g}{unit}, {stats['max']*multiplier:.6g}{unit}]")
            print(f"    Mean: {stats['mean']*multiplier:.6g}{unit}, Std: {stats['std']*multiplier:.6g}{unit}")
            print(f"    |max|: {stats['abs_max']*multiplier:.6g}{unit}")

    return stats


def load_image(image_name: str, image_dir: Path = None) -> np.ndarray | None:
    """Try to load the original image."""
    if image_dir is None:
        return None

    # Try common image extensions
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    # First try the exact name
    image_path = image_dir / image_name
    if image_path.exists():
        try:
            return plt.imread(str(image_path))
        except Exception:
            pass

    # Try with different extensions
    stem = Path(image_name).stem
    for ext in extensions:
        image_path = image_dir / f"{stem}{ext}"
        if image_path.exists():
            try:
                return plt.imread(str(image_path))
            except Exception:
                continue

    return None


def visualize_result(result: CrackAnalysisResult, output_path: Path = None,
                     dpi: int = 150, image_dir: Path = None):
    """Visualize all fields from a result file."""

    print(f"\n{'='*60}")
    print(f"Result: {result.image_name}")
    print(f"Index: {result.image_index}, Reference: {result.reference_image}")
    print(f"{'='*60}")

    # Handle backward compatibility for relative_u
    relative_u = getattr(result, 'relative_u', None)
    if relative_u is None:
        relative_u = np.full_like(result.relative_v, np.nan)

    # Define field configurations: (label, data, is_strain, colormap)
    # Strains will be displayed in percent (multiplied by 100)
    fields_config = {
        'displacement_u': ('Displacement u [px]', result.displacement_u, False, 'viridis'),
        'displacement_v': ('Displacement v [px]', result.displacement_v, False, 'viridis'),
        'strain_exx': ('Strain $\\varepsilon_{xx}$ [%]', result.strain_exx, True, 'RdBu_r'),
        'strain_eyy': ('Strain $\\varepsilon_{yy}$ [%]', result.strain_eyy, True, 'RdBu_r'),
        'strain_exy': ('Strain $\\varepsilon_{xy}$ [%]', result.strain_exy, True, 'RdBu_r'),
        'relative_u': ('Relative u [px]', relative_u, False, 'viridis'),
        'relative_v': ('Relative v [px]', result.relative_v, False, 'viridis'),
        'valid_mask': ('Valid Mask', result.valid_mask.astype(float), False, 'gray'),
    }

    # Collect 1D fields
    fields_1d = {
        'max_relative_v_per_x': ('Max rel. v per x [px]', result.max_relative_v_per_x),
        'max_relative_v_y_position': ('Y-position of max [px]', result.max_relative_v_y_position),
    }

    # Print statistics
    print("\n2D Fields:")
    stats_2d = {}
    for name, (label, data, is_strain, _) in fields_config.items():
        stats_2d[name] = print_field_stats(name, data, is_strain=is_strain)

    print("\n1D Fields:")
    stats_1d = {}
    for name, (label, data) in fields_1d.items():
        stats_1d[name] = print_field_stats(name, data)

    print("\nCoordinates:")
    print_field_stats('grid_x', result.grid_x)
    print_field_stats('grid_y', result.grid_y)
    print_field_stats('grid_x_mm', result.grid_x_mm)
    print_field_stats('grid_y_mm', result.grid_y_mm)

    # Try to load the original image
    original_image = load_image(result.image_name, image_dir)
    if original_image is not None:
        print(f"\nOriginal image loaded: {result.image_name}")
    elif image_dir is not None:
        print(f"\nWarning: Could not load image '{result.image_name}' from {image_dir}")

    # Create visualization
    fig = plt.figure(figsize=(20, 22))

    # Grid layout: 4 rows, 3 columns
    # Row 1: displacement_u, displacement_v, original_image
    # Row 2: strain_exx, strain_eyy, strain_exy
    # Row 3: relative_u, relative_v, (empty)
    # Row 4: max_relative_v_per_x, max_relative_v_y_position, valid_mask

    # Get extent for 2D plots
    extent = [result.grid_x[0], result.grid_x[-1],
              result.grid_y[-1], result.grid_y[0]]

    # Field order for plotting (excluding valid_mask which goes to position 12)
    field_order = [
        ('displacement_u', 1),
        ('displacement_v', 2),
        ('strain_exx', 4),
        ('strain_eyy', 5),
        ('strain_exy', 6),
        ('relative_u', 7),
        ('relative_v', 8),
    ]

    # Plot 2D fields
    for name, pos in field_order:
        ax = fig.add_subplot(4, 3, pos)
        label, data, is_strain, cmap = fields_config[name]

        # For strain, multiply by 100 for percent display
        if is_strain:
            data = data * 100

        # Get valid data range
        valid = data[~np.isnan(data)]
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, [1, 99])
            # Use symmetric colorbar for strains (always RdBu_r)
            if is_strain or (vmin < 0 < vmax):
                abs_max = max(abs(vmin), abs(vmax))
                vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -1, 1

        im = ax.imshow(data, extent=extent, aspect='equal', cmap=cmap,
                      vmin=vmin, vmax=vmax, origin='upper')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_title(f"{label}\n({name})")
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

        # Add stats annotation
        if name in stats_2d and 'abs_max' in stats_2d[name]:
            s = stats_2d[name]
            if not np.isnan(s['abs_max']):
                if is_strain:
                    ax.text(0.02, 0.98, f"|max|={s['abs_max']*100:.3g}%\nvalid={100-s['nan_percent']:.0f}%",
                           transform=ax.transAxes, va='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.02, 0.98, f"|max|={s['abs_max']:.3g}\nvalid={100-s['nan_percent']:.0f}%",
                           transform=ax.transAxes, va='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot original image (top right, position 3)
    ax_img = fig.add_subplot(4, 3, 3)
    if original_image is not None:
        ax_img.imshow(original_image, cmap='gray' if original_image.ndim == 2 else None,
                     aspect='equal', origin='upper')
        ax_img.set_title(f"Original Image\n({result.image_name})")
    else:
        ax_img.text(0.5, 0.5, f"Image not found\n{result.image_name}\n\nUse --image-dir to specify\nthe image directory",
                   ha='center', va='center', transform=ax_img.transAxes,
                   fontsize=10, color='gray')
        ax_img.set_title("Original Image\n(not loaded)")
    ax_img.set_xlabel('x [px]')
    ax_img.set_ylabel('y [px]')

    # Plot 1D fields (row 4, positions 10 and 11)
    ax10 = fig.add_subplot(4, 3, 10)
    label, data = fields_1d['max_relative_v_per_x']
    valid = ~np.isnan(data)
    ax10.plot(result.grid_x[valid], data[valid], 'b-', linewidth=1)
    ax10.set_xlabel('x [px]')
    ax10.set_ylabel(label)
    ax10.set_title('Max relative v per x-position')
    ax10.grid(True, alpha=0.3)
    if np.any(valid):
        ax10.text(0.02, 0.98, f"max={np.nanmax(np.abs(data)):.3g}",
                transform=ax10.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax11 = fig.add_subplot(4, 3, 11)
    label, data = fields_1d['max_relative_v_y_position']
    valid = ~np.isnan(data)
    ax11.plot(result.grid_x[valid], data[valid], 'r-', linewidth=1)
    ax11.set_xlabel('x [px]')
    ax11.set_ylabel(label)
    ax11.set_title('Y-position of max relative v (crack location)')
    ax11.grid(True, alpha=0.3)

    # Plot valid mask (bottom right, position 12)
    ax_mask = fig.add_subplot(4, 3, 12)
    label, data, _, cmap = fields_config['valid_mask']
    im = ax_mask.imshow(data, extent=extent, aspect='equal', cmap=cmap,
                       vmin=0, vmax=1, origin='upper')
    cbar = fig.colorbar(im, ax=ax_mask, shrink=0.8)
    ax_mask.set_title(f"{label}\n(valid_mask)")
    ax_mask.set_xlabel('x [px]')
    ax_mask.set_ylabel('y [px]')

    plt.suptitle(f"Result Inspection: {result.image_name}\n"
                 f"(Index: {result.image_index}, Reference: {result.reference_image})",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)

    return stats_2d, stats_1d


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and visualize a single DIC result file."
    )
    parser.add_argument(
        "result_file",
        type=Path,
        help="Path to result .npz file"
    )
    parser.add_argument(
        "--save", "-s",
        type=Path,
        default=None,
        help="Save figure to file instead of displaying"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)"
    )
    parser.add_argument(
        "--image-dir", "-i",
        type=Path,
        default=None,
        help="Directory containing the original images"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print raw .npz file contents"
    )

    args = parser.parse_args()

    result_path = Path(args.result_file)
    if not result_path.exists():
        print(f"Error: File not found: {result_path}")
        sys.exit(1)

    # Print raw contents if requested
    if args.raw:
        print("\nRaw .npz file contents:")
        print("-" * 40)
        data = np.load(result_path, allow_pickle=True)
        for key in data.files:
            arr = data[key]
            print(f"  {key}: shape={arr.shape if hasattr(arr, 'shape') else 'scalar'}, "
                  f"dtype={arr.dtype if hasattr(arr, 'dtype') else type(arr)}")
        print("-" * 40)

    # Load and visualize
    print(f"\nLoading: {result_path}")
    result = CrackAnalysisResult.load(result_path)

    visualize_result(result, args.save, args.dpi, args.image_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
