#!/usr/bin/env python3
"""
Inspect and visualize a single DIC result file.

This script loads a .npz result file and displays all available fields,
helping to diagnose issues with the analysis.

Usage:
    python inspect_result.py /path/to/result.npz
    python inspect_result.py /path/to/result.npz --save output.png
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


def print_field_stats(name: str, data: np.ndarray) -> dict:
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

    print(f"  {name}:")
    print(f"    Shape: {stats['shape']}, dtype: {stats['dtype']}")
    if 'valid_elements' in stats:
        print(f"    Valid: {stats['valid_elements']}/{stats['total_elements']} ({100-stats['nan_percent']:.1f}%)")
        if stats['valid_elements'] > 0:
            print(f"    Range: [{stats['min']:.6g}, {stats['max']:.6g}]")
            print(f"    Mean: {stats['mean']:.6g}, Std: {stats['std']:.6g}")
            print(f"    |max|: {stats['abs_max']:.6g}")

    return stats


def visualize_result(result: CrackAnalysisResult, output_path: Path = None, dpi: int = 150):
    """Visualize all fields from a result file."""

    print(f"\n{'='*60}")
    print(f"Result: {result.image_name}")
    print(f"Index: {result.image_index}, Reference: {result.reference_image}")
    print(f"{'='*60}")

    # Collect all 2D fields
    # Handle backward compatibility for relative_u
    relative_u = getattr(result, 'relative_u', None)
    if relative_u is None:
        relative_u = np.full_like(result.relative_v, np.nan)

    fields_2d = {
        'displacement_u': ('Displacement u [px]', result.displacement_u),
        'displacement_v': ('Displacement v [px]', result.displacement_v),
        'strain_exx': ('Strain $\\varepsilon_{xx}$ [-]', result.strain_exx),
        'strain_eyy': ('Strain $\\varepsilon_{yy}$ [-]', result.strain_eyy),
        'strain_exy': ('Strain $\\varepsilon_{xy}$ [-]', result.strain_exy),
        'relative_u': ('Relative u [px]', relative_u),
        'relative_v': ('Relative v [px]', result.relative_v),
        'valid_mask': ('Valid Mask', result.valid_mask.astype(float)),
    }

    # Collect 1D fields
    fields_1d = {
        'max_relative_v_per_x': ('Max rel. v per x [px]', result.max_relative_v_per_x),
        'max_relative_v_y_position': ('Y-position of max [px]', result.max_relative_v_y_position),
    }

    # Print statistics
    print("\n2D Fields:")
    stats_2d = {}
    for name, (label, data) in fields_2d.items():
        stats_2d[name] = print_field_stats(name, data)

    print("\n1D Fields:")
    stats_1d = {}
    for name, (label, data) in fields_1d.items():
        stats_1d[name] = print_field_stats(name, data)

    print("\nCoordinates:")
    print_field_stats('grid_x', result.grid_x)
    print_field_stats('grid_y', result.grid_y)
    print_field_stats('grid_x_mm', result.grid_x_mm)
    print_field_stats('grid_y_mm', result.grid_y_mm)

    # Create visualization
    fig = plt.figure(figsize=(20, 20))

    # 2D fields (4 rows x 3 cols for 8 2D fields + 2 1D fields + 2 empty)
    n_2d = len(fields_2d)
    n_1d = len(fields_1d)

    # Grid layout: 4 rows, 3 columns
    # Row 1: displacement_u, displacement_v, valid_mask
    # Row 2: strain_exx, strain_eyy, strain_exy
    # Row 3: relative_u, relative_v, (empty)
    # Row 4: max_relative_v_per_x, max_relative_v_y_position, (empty)

    field_order_2d = ['displacement_u', 'displacement_v', 'valid_mask',
                      'strain_exx', 'strain_eyy', 'strain_exy',
                      'relative_u', 'relative_v']

    # Get extent for 2D plots
    extent = [result.grid_x[0], result.grid_x[-1],
              result.grid_y[-1], result.grid_y[0]]

    # Plot 2D fields
    for idx, name in enumerate(field_order_2d):
        ax = fig.add_subplot(4, 3, idx + 1)
        label, data = fields_2d[name]

        if name == 'valid_mask':
            im = ax.imshow(data, extent=extent, aspect='auto', cmap='gray',
                          vmin=0, vmax=1, origin='upper')
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            # Get valid data range
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                vmin, vmax = np.percentile(valid, [1, 99])
                # Use symmetric colorbar if data crosses zero
                if vmin < 0 < vmax:
                    abs_max = max(abs(vmin), abs(vmax))
                    vmin, vmax = -abs_max, abs_max
                    cmap = 'RdBu_r'
                else:
                    cmap = 'viridis'
            else:
                vmin, vmax = -1, 1
                cmap = 'RdBu_r'

            im = ax.imshow(data, extent=extent, aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, origin='upper')
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_title(f"{label}\n({name})")
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

        # Add stats annotation
        if name in stats_2d and 'abs_max' in stats_2d[name]:
            s = stats_2d[name]
            if not np.isnan(s['abs_max']):
                ax.text(0.02, 0.98, f"|max|={s['abs_max']:.3g}\nvalid={100-s['nan_percent']:.0f}%",
                       transform=ax.transAxes, va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 1D fields (row 4)
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

    visualize_result(result, args.save, args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
