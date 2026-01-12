"""
Visualize DIC displacement and discontinuity fields.

Usage:
    python visualize_dic.py results.npz
    python visualize_dic.py results.npz --field delta_total --cmap hot
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_results(npz_path):
    """Load results from NPZ file."""
    return np.load(npz_path)


def visualize_displacement(data, step, save_path=None):
    """Visualize u and v displacement fields."""
    u = data['u']
    v = data['v']
    roi = data['roi']

    # Mask invalid points
    u_masked = np.where(roi, u, np.nan)
    v_masked = np.where(roi, v, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # u-displacement
    im0 = axes[0].imshow(u_masked, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('u-Verschiebung (horizontal) [Pixel]')
    axes[0].set_xlabel(f'x (Grid-Index, ×{step} = Pixel)')
    axes[0].set_ylabel(f'y (Grid-Index, ×{step} = Pixel)')
    plt.colorbar(im0, ax=axes[0], label='u [Pixel]')

    # v-displacement
    im1 = axes[1].imshow(v_masked, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('v-Verschiebung (vertikal) [Pixel]')
    axes[1].set_xlabel(f'x (Grid-Index, ×{step} = Pixel)')
    axes[1].set_ylabel(f'y (Grid-Index, ×{step} = Pixel)')
    plt.colorbar(im1, ax=axes[1], label='v [Pixel]')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_displacement.png'), dpi=150)
        print(f"Saved: {save_path.replace('.png', '_displacement.png')}")
    else:
        plt.show()


def visualize_discontinuity(data, step, vmax=None, save_path=None):
    """Visualize displacement discontinuity fields."""
    delta_total = data['delta_total']
    delta_u_x = data['delta_u_x']
    delta_v_y = data['delta_v_y']
    roi = data['roi']

    # Auto vmax if not specified
    if vmax is None:
        valid = ~np.isnan(delta_total)
        if np.any(valid):
            vmax = np.percentile(delta_total[valid], 99)
        else:
            vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Total discontinuity
    im0 = axes[0].imshow(delta_total, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0].set_title('Gesamt-Diskontinuität |Δ| [Pixel]')
    axes[0].set_xlabel(f'x (Grid-Index, ×{step} = Pixel)')
    axes[0].set_ylabel(f'y (Grid-Index, ×{step} = Pixel)')
    plt.colorbar(im0, ax=axes[0], label='|Δ| [Pixel]')

    # Horizontal discontinuity (du/dx - indicates vertical cracks)
    du_max = np.nanmax(np.abs(delta_u_x)) if not np.all(np.isnan(delta_u_x)) else 1.0
    im1 = axes[1].imshow(delta_u_x, cmap='RdBu_r', vmin=-du_max, vmax=du_max, aspect='auto')
    axes[1].set_title('Δu in x-Richtung [Pixel]\n(Vertikaler Riss = großer Wert)')
    axes[1].set_xlabel(f'x (Grid-Index, ×{step} = Pixel)')
    axes[1].set_ylabel(f'y (Grid-Index, ×{step} = Pixel)')
    plt.colorbar(im1, ax=axes[1], label='Δu_x [Pixel]')

    # Vertical discontinuity (dv/dy - indicates horizontal cracks)
    dv_max = np.nanmax(np.abs(delta_v_y)) if not np.all(np.isnan(delta_v_y)) else 1.0
    im2 = axes[2].imshow(delta_v_y, cmap='RdBu_r', vmin=-dv_max, vmax=dv_max, aspect='auto')
    axes[2].set_title('Δv in y-Richtung [Pixel]\n(Horizontaler Riss = großer Wert)')
    axes[2].set_xlabel(f'x (Grid-Index, ×{step} = Pixel)')
    axes[2].set_ylabel(f'y (Grid-Index, ×{step} = Pixel)')
    plt.colorbar(im2, ax=axes[2], label='Δv_y [Pixel]')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_discontinuity.png'), dpi=150)
        print(f"Saved: {save_path.replace('.png', '_discontinuity.png')}")
    else:
        plt.show()


def visualize_correlation(data, step, save_path=None):
    """Visualize correlation coefficient field."""
    cc = data['corrcoef']
    roi = data['roi']

    cc_masked = np.where(roi, cc, np.nan)

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(cc_masked, cmap='viridis', vmin=0.8, vmax=1.0, aspect='auto')
    ax.set_title('Korrelationskoeffizient')
    ax.set_xlabel(f'x (Grid-Index, ×{step} = Pixel)')
    ax.set_ylabel(f'y (Grid-Index, ×{step} = Pixel)')
    plt.colorbar(im, ax=ax, label='CC')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_correlation.png'), dpi=150)
        print(f"Saved: {save_path.replace('.png', '_correlation.png')}")
    else:
        plt.show()


def visualize_single_field(data, field_name, step, cmap='viridis', vmin=None, vmax=None, save_path=None):
    """Visualize a single field from the data."""
    if field_name not in data:
        print(f"Field '{field_name}' not found in data.")
        print(f"Available fields: {list(data.keys())}")
        return

    field = data[field_name]
    roi = data['roi'] if 'roi' in data else np.ones_like(field, dtype=bool)

    field_masked = np.where(roi, field, np.nan)

    # Auto-scale if not specified
    valid = ~np.isnan(field_masked)
    if vmin is None and np.any(valid):
        vmin = np.nanpercentile(field_masked[valid], 1)
    if vmax is None and np.any(valid):
        vmax = np.nanpercentile(field_masked[valid], 99)

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(field_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'{field_name}')
    ax.set_xlabel(f'x (Grid-Index, ×{step} = Pixel)')
    ax.set_ylabel(f'y (Grid-Index, ×{step} = Pixel)')
    plt.colorbar(im, ax=ax, label=field_name)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def visualize_all(data, step, vmax_disc=None, save_prefix=None):
    """Visualize all fields in one overview figure."""
    u = data['u']
    v = data['v']
    cc = data['corrcoef']
    delta_total = data['delta_total']
    roi = data['roi']

    # Mask invalid points
    u_masked = np.where(roi, u, np.nan)
    v_masked = np.where(roi, v, np.nan)
    cc_masked = np.where(roi, cc, np.nan)

    # Auto vmax for discontinuity
    if vmax_disc is None:
        valid = ~np.isnan(delta_total)
        if np.any(valid):
            vmax_disc = np.percentile(delta_total[valid], 99)
        else:
            vmax_disc = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # u-displacement
    im0 = axes[0, 0].imshow(u_masked, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('u-Verschiebung [Pixel]')
    plt.colorbar(im0, ax=axes[0, 0])

    # v-displacement
    im1 = axes[0, 1].imshow(v_masked, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('v-Verschiebung [Pixel]')
    plt.colorbar(im1, ax=axes[0, 1])

    # Correlation
    im2 = axes[1, 0].imshow(cc_masked, cmap='viridis', vmin=0.8, vmax=1.0, aspect='auto')
    axes[1, 0].set_title('Korrelationskoeffizient')
    plt.colorbar(im2, ax=axes[1, 0])

    # Discontinuity
    im3 = axes[1, 1].imshow(delta_total, cmap='hot', vmin=0, vmax=vmax_disc, aspect='auto')
    axes[1, 1].set_title('Lokale Diskontinuität |Δ| [Pixel]')
    plt.colorbar(im3, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel(f'x (×{step} Pixel)')
        ax.set_ylabel(f'y (×{step} Pixel)')

    plt.tight_layout()

    if save_prefix:
        plt.savefig(f'{save_prefix}_overview.png', dpi=150)
        print(f"Saved: {save_prefix}_overview.png")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DIC displacement and discontinuity fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all fields
  python visualize_dic.py results.npz

  # Show specific field
  python visualize_dic.py results.npz --field delta_total --cmap hot

  # Save to file
  python visualize_dic.py results.npz --save output.png

  # Adjust discontinuity color scale
  python visualize_dic.py results.npz --vmax 2.0
"""
    )
    parser.add_argument("npz_file", help="Path to NPZ file with DIC results")
    parser.add_argument("--field", type=str, default=None,
                        help="Specific field to visualize (u, v, corrcoef, delta_total, etc.)")
    parser.add_argument("--cmap", type=str, default='viridis',
                        help="Colormap for single field (default: viridis)")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum value for colormap")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for colormap")
    parser.add_argument("--save", type=str, default=None, help="Save figures to file (prefix)")
    parser.add_argument("--list-fields", action="store_true", help="List available fields and exit")

    args = parser.parse_args()

    # Load data
    data = load_results(args.npz_file)
    step = int(data['step']) if 'step' in data else 11

    if args.list_fields:
        print("Available fields:")
        for key in data.keys():
            arr = data[key]
            if hasattr(arr, 'shape'):
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"  {key}: {arr}")
        return

    # Visualize
    if args.field:
        visualize_single_field(data, args.field, step,
                               cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                               save_path=args.save)
    else:
        # Show all
        if args.save:
            visualize_displacement(data, step, save_path=args.save)
            visualize_discontinuity(data, step, vmax=args.vmax, save_path=args.save)
            visualize_correlation(data, step, save_path=args.save)
            visualize_all(data, step, vmax_disc=args.vmax, save_prefix=args.save.replace('.png', ''))
        else:
            visualize_all(data, step, vmax_disc=args.vmax)


if __name__ == "__main__":
    main()
