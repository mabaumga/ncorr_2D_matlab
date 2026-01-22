#!/usr/bin/env python3
"""
Export statistics (min, max, mean) from all DIC result files to ASCII.

This script loads all .npz result files from a directory and exports
the statistics for each field to a text file.

Usage:
    python export_statistics.py /path/to/results
    python export_statistics.py /path/to/results --output stats.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from batch_crack_analysis import CrackAnalysisResult


def compute_field_stats(data: np.ndarray, is_strain: bool = False) -> Dict[str, float]:
    """Compute statistics for a data field."""
    if data is None:
        return {'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan}

    # Get valid (non-NaN) data
    if np.issubdtype(data.dtype, np.floating):
        valid_data = data[~np.isnan(data)]
    else:
        valid_data = data.flatten()

    if len(valid_data) == 0:
        return {'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan}

    # Multiply by 100 for strain (percent)
    multiplier = 100 if is_strain else 1

    return {
        'min': float(np.min(valid_data)) * multiplier,
        'max': float(np.max(valid_data)) * multiplier,
        'mean': float(np.mean(valid_data)) * multiplier,
        'std': float(np.std(valid_data)) * multiplier,
    }


def load_all_results(results_dir: Path) -> List[CrackAnalysisResult]:
    """Load all result files from a directory."""
    result_files = sorted(results_dir.glob("*.npz"))

    if not result_files:
        print(f"No .npz files found in {results_dir}")
        return []

    results = []
    for f in result_files:
        try:
            result = CrackAnalysisResult.load(f)
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    # Sort by image index
    results.sort(key=lambda r: r.image_index)

    return results


def export_statistics(results: List[CrackAnalysisResult], output_path: Path):
    """Export statistics for all results to ASCII file."""

    # Define fields to export: (name, attribute, is_strain, unit)
    fields = [
        ('displacement_u', 'displacement_u', False, 'px'),
        ('displacement_v', 'displacement_v', False, 'px'),
        ('strain_exx', 'strain_exx', True, '%'),
        ('strain_eyy', 'strain_eyy', True, '%'),
        ('strain_exy', 'strain_exy', True, '%'),
        ('relative_u', 'relative_u', False, 'px'),
        ('relative_v', 'relative_v', False, 'px'),
        ('max_relative_v', 'max_relative_v_per_x', False, 'px'),
    ]

    # Build header
    header_parts = ['index', 'image_name']
    for name, _, _, unit in fields:
        header_parts.extend([
            f'{name}_min[{unit}]',
            f'{name}_max[{unit}]',
            f'{name}_mean[{unit}]',
            f'{name}_std[{unit}]',
        ])

    # Compute statistics for each result
    rows = []
    for result in results:
        row = [str(result.image_index), result.image_name]

        for name, attr, is_strain, _ in fields:
            # Handle backward compatibility for relative_u
            if attr == 'relative_u':
                data = getattr(result, 'relative_u', None)
                if data is None:
                    data = np.full_like(result.relative_v, np.nan)
            else:
                data = getattr(result, attr, None)

            stats = compute_field_stats(data, is_strain=is_strain)
            row.extend([
                f"{stats['min']:.6g}",
                f"{stats['max']:.6g}",
                f"{stats['mean']:.6g}",
                f"{stats['std']:.6g}",
            ])

        rows.append(row)

    # Write to file
    with open(output_path, 'w') as f:
        # Write header comment
        f.write("# DIC Result Statistics\n")
        f.write(f"# Number of images: {len(results)}\n")
        f.write(f"# Strain values in percent (%)\n")
        f.write(f"# Displacement values in pixels (px)\n")
        f.write("#\n")

        # Write column header
        f.write('\t'.join(header_parts) + '\n')

        # Write data rows
        for row in rows:
            f.write('\t'.join(row) + '\n')

    print(f"Statistics exported to: {output_path}")
    print(f"  - {len(results)} images")
    print(f"  - {len(fields)} fields")
    print(f"  - {len(header_parts)} columns")


def print_summary(results: List[CrackAnalysisResult]):
    """Print a summary of all results to console."""

    print(f"\n{'='*80}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*80}")

    # Define fields for summary
    fields = [
        ('displacement_u', 'displacement_u', False, 'px'),
        ('displacement_v', 'displacement_v', False, 'px'),
        ('strain_exx', 'strain_exx', True, '%'),
        ('strain_eyy', 'strain_eyy', True, '%'),
        ('strain_exy', 'strain_exy', True, '%'),
        ('relative_v', 'relative_v', False, 'px'),
    ]

    # Collect global statistics
    for name, attr, is_strain, unit in fields:
        all_data = []
        for result in results:
            data = getattr(result, attr, None)
            if data is not None:
                valid = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data.flatten()
                all_data.extend(valid.tolist())

        if all_data:
            all_data = np.array(all_data)
            multiplier = 100 if is_strain else 1
            print(f"\n{name} [{unit}]:")
            print(f"  Global min:  {np.min(all_data) * multiplier:12.6g}")
            print(f"  Global max:  {np.max(all_data) * multiplier:12.6g}")
            print(f"  Global mean: {np.mean(all_data) * multiplier:12.6g}")
            print(f"  Global std:  {np.std(all_data) * multiplier:12.6g}")


def main():
    parser = argparse.ArgumentParser(
        description="Export statistics from all DIC result files to ASCII."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing result .npz files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: results_dir/statistics.txt)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print global summary statistics"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    # Set default output path
    if args.output is None:
        output_path = results_dir / "statistics.txt"
    else:
        output_path = args.output

    # Load all results
    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir)

    if not results:
        print("No results to export.")
        sys.exit(1)

    print(f"Loaded {len(results)} result files")

    # Export statistics
    export_statistics(results, output_path)

    # Print summary if requested
    if args.summary:
        print_summary(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
