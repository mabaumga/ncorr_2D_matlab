#!/usr/bin/env python3
"""
Batch analysis for crack propagation in image sequences.

This script processes a directory of images, using the first image (alphabetically)
as reference and computing displacement/strain fields for all subsequent images.
Designed for analyzing crack opening and propagation under cyclic loading.

Usage:
    python batch_crack_analysis.py /path/to/images --output results
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncorr import Ncorr, DICParameters, NcorrImage, NcorrROI
from ncorr.algorithms.dic import DICAnalysis, SeedInfo
from ncorr.algorithms.strain import StrainCalculator


@dataclass
class AnalysisConfig:
    """Configuration for batch crack analysis."""

    # Image scaling - resolution in micrometers per pixel
    resolution_um_per_px: float = 20.0  # µm/pixel (e.g., 6000px = 120mm → 20 µm/px)

    # Relative displacement calculation
    reference_distance_mm: float = 1.0  # Distance over which to compute relative displacement

    # DIC parameters
    subset_radius: int = 20
    subset_spacing: int = 3
    strain_radius: int = 5

    # ROI margin (pixels from edge to exclude)
    roi_margin: int = 30

    # Processing options
    skip_existing: bool = True    # Skip already processed images
    seed_retry_grid: int = 5      # Grid size for seed retry (5 = try 5x5 grid around center)

    # Coordinate transformation (applied after DIC)
    rotation_deg: float = 0.0     # Rotation angle in degrees (counter-clockwise positive)

    @property
    def pixels_per_mm(self) -> float:
        """Pixels per millimeter."""
        return 1000.0 / self.resolution_um_per_px

    @property
    def um_per_px(self) -> float:
        """Micrometers per pixel (alias for resolution)."""
        return self.resolution_um_per_px

    @property
    def reference_distance_px(self) -> int:
        """Reference distance in pixels."""
        return int(round(self.reference_distance_mm * self.pixels_per_mm))

    @property
    def grid_step(self) -> int:
        """Grid step size (spacing + 1)."""
        return self.subset_spacing + 1

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['pixels_per_mm'] = self.pixels_per_mm
        d['um_per_px'] = self.um_per_px
        d['reference_distance_px'] = self.reference_distance_px
        d['grid_step'] = self.grid_step
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'AnalysisConfig':
        """Create from dictionary."""
        # Filter out computed properties
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class CrackAnalysisResult:
    """Result of crack analysis for a single image."""

    image_name: str
    image_index: int
    reference_image: str  # Name of reference image used

    # Grid coordinates (in pixels)
    grid_x: NDArray[np.float64]  # 1D array of x positions
    grid_y: NDArray[np.float64]  # 1D array of y positions

    # Full displacement fields (2D arrays)
    displacement_u: NDArray[np.float64]  # x-displacement
    displacement_v: NDArray[np.float64]  # y-displacement

    # Full strain fields (2D arrays)
    strain_exx: NDArray[np.float64]
    strain_eyy: NDArray[np.float64]
    strain_exy: NDArray[np.float64]

    # Valid mask
    valid_mask: NDArray[np.bool_]

    # Relative displacement fields (computed over reference distance)
    relative_v: NDArray[np.float64]  # Relative y-displacement

    # Crack analysis results
    max_relative_v_per_x: NDArray[np.float64]  # Max relative v for each x position
    max_relative_v_y_position: NDArray[np.float64]  # y-position of max relative v

    # Physical coordinates (in mm)
    grid_x_mm: NDArray[np.float64]
    grid_y_mm: NDArray[np.float64]

    def save(self, filepath: Path):
        """Save result to .npz file."""
        np.savez_compressed(
            filepath,
            image_name=self.image_name,
            image_index=self.image_index,
            reference_image=self.reference_image,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            displacement_u=self.displacement_u,
            displacement_v=self.displacement_v,
            strain_exx=self.strain_exx,
            strain_eyy=self.strain_eyy,
            strain_exy=self.strain_exy,
            valid_mask=self.valid_mask,
            relative_v=self.relative_v,
            max_relative_v_per_x=self.max_relative_v_per_x,
            max_relative_v_y_position=self.max_relative_v_y_position,
            grid_x_mm=self.grid_x_mm,
            grid_y_mm=self.grid_y_mm,
        )

    @classmethod
    def load(cls, filepath: Path) -> 'CrackAnalysisResult':
        """Load result from .npz file."""
        data = np.load(filepath, allow_pickle=True)
        return cls(
            image_name=str(data['image_name']),
            image_index=int(data['image_index']),
            reference_image=str(data['reference_image']) if 'reference_image' in data else 'unknown',
            grid_x=data['grid_x'],
            grid_y=data['grid_y'],
            displacement_u=data['displacement_u'],
            displacement_v=data['displacement_v'],
            strain_exx=data['strain_exx'],
            strain_eyy=data['strain_eyy'],
            strain_exy=data['strain_exy'],
            valid_mask=data['valid_mask'],
            relative_v=data['relative_v'],
            max_relative_v_per_x=data['max_relative_v_per_x'],
            max_relative_v_y_position=data['max_relative_v_y_position'],
            grid_x_mm=data['grid_x_mm'],
            grid_y_mm=data['grid_y_mm'],
        )


def get_image_files(input_dir: Path, extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')) -> List[Path]:
    """Get sorted list of image files in directory."""
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f'*{ext}'))
        files.extend(input_dir.glob(f'*{ext.upper()}'))
    return sorted(set(files))


def compute_relative_displacement(
    displacement_v: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    reference_distance_px: int,
    grid_step: int,
) -> NDArray[np.float64]:
    """
    Compute relative displacement over a reference distance.

    For each point, computes v(y + d/2) - v(y - d/2) where d is the reference distance.
    This approximates strain_yy * d for small deformations.

    Args:
        displacement_v: 2D array of y-displacements (ny, nx)
        grid_y: 1D array of y-coordinates
        reference_distance_px: Distance in pixels over which to compute relative displacement
        grid_step: Grid spacing in pixels

    Returns:
        2D array of relative displacements (same shape as input)
    """
    ny, nx = displacement_v.shape
    relative_v = np.full_like(displacement_v, np.nan)

    # Number of grid points for half the reference distance
    half_dist_points = int(round(reference_distance_px / (2 * grid_step)))

    if half_dist_points < 1:
        half_dist_points = 1

    # Compute relative displacement
    for i in range(half_dist_points, ny - half_dist_points):
        v_plus = displacement_v[i + half_dist_points, :]
        v_minus = displacement_v[i - half_dist_points, :]
        relative_v[i, :] = v_plus - v_minus

    return relative_v


def find_max_relative_displacement_per_x(
    relative_v: NDArray[np.float64],
    grid_y: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Find maximum relative displacement and its y-position for each x-column.

    Args:
        relative_v: 2D array of relative displacements (ny, nx)
        grid_y: 1D array of y-coordinates

    Returns:
        Tuple of (max_values, y_positions) - both 1D arrays of length nx
    """
    ny, nx = relative_v.shape
    max_values = np.full(nx, np.nan)
    y_positions = np.full(nx, np.nan)

    for j in range(nx):
        col = relative_v[:, j]
        valid_mask = ~np.isnan(col)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            valid_values = col[valid_mask]
            max_idx = np.argmax(np.abs(valid_values))
            max_values[j] = valid_values[max_idx]
            y_positions[j] = grid_y[valid_indices[max_idx]]

    return max_values, y_positions


def apply_rotation_to_displacements(
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    displacement_u: NDArray[np.float64],
    displacement_v: NDArray[np.float64],
    rotation_deg: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply rotation transformation to coordinates and displacement vectors.

    This is a post-DIC coordinate transformation. The rotation is applied to both
    the coordinate system and the displacement vectors.

    For a rotation by angle θ (counter-clockwise positive):
    - Coordinates: x' = x*cos(θ) - y*sin(θ), y' = x*sin(θ) + y*cos(θ)
    - Displacements: u' = u*cos(θ) - v*sin(θ), v' = u*sin(θ) + v*cos(θ)

    Args:
        grid_x: 1D array of x-coordinates (will be used to create meshgrid)
        grid_y: 1D array of y-coordinates
        displacement_u: 2D array of x-displacements (ny, nx)
        displacement_v: 2D array of y-displacements (ny, nx)
        rotation_deg: Rotation angle in degrees (counter-clockwise positive)

    Returns:
        Tuple of (grid_x_rot, grid_y_rot, u_rot, v_rot)
        Note: Rotated coordinates are 2D arrays matching displacement shape
    """
    if abs(rotation_deg) < 1e-10:
        # No rotation needed
        return grid_x, grid_y, displacement_u, displacement_v

    # Convert to radians
    theta = np.radians(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Create meshgrid for coordinates
    X, Y = np.meshgrid(grid_x, grid_y)

    # Rotate coordinates (around origin, which is top-left corner)
    # For image coordinates, we typically want to rotate around the center
    center_x = (grid_x.max() + grid_x.min()) / 2
    center_y = (grid_y.max() + grid_y.min()) / 2

    # Translate to center, rotate, translate back
    X_centered = X - center_x
    Y_centered = Y - center_y

    X_rot = X_centered * cos_t - Y_centered * sin_t + center_x
    Y_rot = X_centered * sin_t + Y_centered * cos_t + center_y

    # Rotate displacement vectors
    u_rot = displacement_u * cos_t - displacement_v * sin_t
    v_rot = displacement_u * sin_t + displacement_v * cos_t

    return X_rot, Y_rot, u_rot, v_rot


class BatchCrackAnalyzer:
    """Batch analyzer for crack propagation in image sequences."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._ref_img: Optional[NcorrImage] = None
        self._roi: Optional[NcorrROI] = None
        self._params: Optional[DICParameters] = None
        self._seeds: List[SeedInfo] = []

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as grayscale numpy array."""
        with Image.open(path) as img:
            if img.mode != 'L':
                img = img.convert('L')
            return np.array(img)

    def _setup_reference(self, reference_path: Path) -> None:
        """Set up reference image and ROI."""
        # Load reference image
        ref_array = self._load_image(reference_path)
        self._ref_img = NcorrImage.from_array(ref_array)

        height, width = ref_array.shape

        # Create ROI (full image minus margins)
        margin = self.config.roi_margin
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[margin:height-margin, margin:width-margin] = True

        self._roi = NcorrROI()
        self._roi.set_roi("load", {"mask": mask, "cutoff": 0})

        # Set up DIC parameters
        self._params = DICParameters(
            radius=self.config.subset_radius,
            spacing=self.config.subset_spacing,
            cutoff_diffnorm=1e-4,
            cutoff_iteration=50,
            subset_trunc=False,
        )

        # Create initial seed positions for retry (grid around center)
        center_x = width // 2
        center_y = height // 2
        self._image_size = (height, width)

        # Generate grid of seed positions to try
        grid_size = self.config.seed_retry_grid
        step_x = (width - 2 * margin) // (grid_size + 1)
        step_y = (height - 2 * margin) // (grid_size + 1)

        self._seed_positions = []
        # Start with center
        self._seed_positions.append((center_x, center_y))
        # Add grid positions (spiral out from center)
        for radius in range(1, grid_size):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only edge of current ring
                        sx = center_x + dx * step_x
                        sy = center_y + dy * step_y
                        if margin < sx < width - margin and margin < sy < height - margin:
                            self._seed_positions.append((sx, sy))

        self._seeds = [SeedInfo(x=center_x, y=center_y, u=0.0, v=0.0, region_idx=0, valid=True)]

    def _get_output_path(self, output_dir: Path, image_name: str) -> Path:
        """Get output path for a result file."""
        stem = Path(image_name).stem
        return output_dir / f"{stem}_result.npz"

    def _is_already_processed(self, output_dir: Path, image_name: str) -> bool:
        """Check if image has already been processed."""
        if not self.config.skip_existing:
            return False
        output_path = self._get_output_path(output_dir, image_name)
        return output_path.exists()

    def _compute_grid_coordinates(self, shape: Tuple[int, int]) -> Tuple[NDArray, NDArray]:
        """Compute grid coordinates from result shape."""
        ny, nx = shape
        step = self.config.grid_step
        grid_x = np.arange(nx) * step
        grid_y = np.arange(ny) * step
        return grid_x, grid_y

    def analyze_image(
        self,
        image_path: Path,
        image_index: int,
    ) -> Optional[CrackAnalysisResult]:
        """
        Analyze a single image against the reference.

        Args:
            image_path: Path to current image
            image_index: Index of image in sequence

        Returns:
            CrackAnalysisResult or None if analysis fails
        """
        if self._ref_img is None or self._roi is None or self._params is None:
            raise RuntimeError("Reference not set up. Call _setup_reference first.")

        # Load current image
        cur_array = self._load_image(image_path)
        cur_img = NcorrImage.from_array(cur_array)

        # Try DIC analysis with current seeds, retry with alternative positions if needed
        disp = None
        seeds_to_try = [self._seeds]

        # If first attempt fails, try alternative seed positions
        for seed_pos in self._seed_positions:
            alt_seed = [SeedInfo(x=seed_pos[0], y=seed_pos[1], u=0.0, v=0.0, region_idx=0, valid=True)]
            seeds_to_try.append(alt_seed)

        for attempt, seeds in enumerate(seeds_to_try):
            dic = DICAnalysis(self._params)
            try:
                results = dic.analyze(self._ref_img, [cur_img], self._roi, seeds)
            except Exception as e:
                if attempt == 0:
                    # First attempt failed, will try alternatives
                    continue
                else:
                    continue

            if results and len(results) > 0:
                disp = results[0]
                # Check if we have enough valid points (more than 10% of expected)
                valid_count = np.sum(disp.roi)
                if valid_count > 100:  # Reasonable minimum
                    if attempt > 0:
                        print(f"\n  Seed retry successful at position {attempt}: ({seeds[0].x}, {seeds[0].y})")
                    break
                else:
                    disp = None  # Not enough points, try next seed

        if disp is None:
            print(f"\nDIC failed for {image_path.name}: All seed positions exhausted")
            return None

        # Extract displacement fields
        displacement_u = disp.u.copy()
        displacement_v = disp.v.copy()
        valid_mask = disp.roi.copy()

        # Set invalid points to NaN
        displacement_u[~valid_mask] = np.nan
        displacement_v[~valid_mask] = np.nan

        # Compute grid coordinates
        grid_x, grid_y = self._compute_grid_coordinates(displacement_u.shape)

        # Apply rotation transformation if configured (post-DIC)
        if abs(self.config.rotation_deg) > 1e-10:
            _, _, displacement_u, displacement_v = apply_rotation_to_displacements(
                grid_x, grid_y, displacement_u, displacement_v, self.config.rotation_deg
            )

        # Compute strain (on rotated displacements if applicable)
        strain_calc = StrainCalculator(strain_radius=self.config.strain_radius)
        try:
            strain = strain_calc.calculate_green_lagrange(
                disp.u, disp.v, self._roi, self.config.grid_step
            )
            strain_exx = strain.exx.copy()
            strain_eyy = strain.eyy.copy()
            strain_exy = strain.exy.copy()
        except Exception as e:
            print(f"\n  Warning: Strain calculation failed: {e}")
            strain_exx = np.full_like(displacement_u, np.nan)
            strain_eyy = np.full_like(displacement_u, np.nan)
            strain_exy = np.full_like(displacement_u, np.nan)

        # Compute relative displacement (uses rotated v if rotation was applied)
        relative_v = compute_relative_displacement(
            displacement_v, grid_y,
            self.config.reference_distance_px,
            self.config.grid_step
        )

        # Find max relative displacement per x-position
        max_rel_v, max_rel_y = find_max_relative_displacement_per_x(relative_v, grid_y)

        # Convert to physical coordinates
        grid_x_mm = grid_x / self.config.pixels_per_mm
        grid_y_mm = grid_y / self.config.pixels_per_mm

        # Update seeds for next image (use results from this one)
        if disp.seed_info:
            self._seeds = disp.seed_info

        return CrackAnalysisResult(
            image_name=image_path.name,
            image_index=image_index,
            reference_image=getattr(self, '_reference_name', 'unknown'),
            grid_x=grid_x,
            grid_y=grid_y,
            displacement_u=displacement_u,
            displacement_v=displacement_v,
            strain_exx=strain_exx,
            strain_eyy=strain_eyy,
            strain_exy=strain_exy,
            valid_mask=valid_mask,
            relative_v=relative_v,
            max_relative_v_per_x=max_rel_v,
            max_relative_v_y_position=max_rel_y,
            grid_x_mm=grid_x_mm,
            grid_y_mm=grid_y_mm,
        )

    def run(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> List[CrackAnalysisResult]:
        """
        Run batch analysis on all images in directory.

        Args:
            input_dir: Directory containing images
            output_dir: Directory for results (default: input_dir/results)

        Returns:
            List of analysis results
        """
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir / "results"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_files = get_image_files(input_dir)
        if len(image_files) < 2:
            raise ValueError(f"Need at least 2 images, found {len(image_files)}")

        # First image is reference
        reference_image = image_files[0]
        current_images = image_files[1:]

        print(f"Reference image (FIXED): {reference_image.name}")
        print(f"Images to process: {len(current_images)} (from {current_images[0].name} to {current_images[-1].name})")
        print(f"Output directory: {output_dir}")
        print(f"Resolution: {self.config.resolution_um_per_px:.2f} µm/pixel ({self.config.pixels_per_mm:.1f} px/mm)")
        print(f"Reference distance: {self.config.reference_distance_mm} mm = {self.config.reference_distance_px} px")
        if abs(self.config.rotation_deg) > 1e-10:
            print(f"Post-DIC rotation: {self.config.rotation_deg}°")
        print(f"NOTE: All images are compared against the reference: {reference_image.name}")

        # Save configuration
        config_path = output_dir / "analysis_config.json"
        config_data = self.config.to_dict()
        config_data['reference_image'] = reference_image.name
        config_data['analysis_timestamp'] = datetime.now().isoformat()
        config_data['total_images'] = len(image_files)

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Initialize with reference (this stays fixed for ALL images)
        print(f"\nSetting up reference image: {reference_image.name}")
        self._setup_reference(reference_image)
        self._reference_name = reference_image.name  # Store for results

        # Process images in forward order (first to last)
        results = []
        skipped = 0

        pbar = tqdm(range(len(current_images)), desc="Analyzing images", unit="img")

        for idx in pbar:
            image_path = current_images[idx]
            image_index = idx + 1  # 0 is reference

            # Check if already processed
            if self._is_already_processed(output_dir, image_path.name):
                # Load existing result
                output_path = self._get_output_path(output_dir, image_path.name)
                try:
                    result = CrackAnalysisResult.load(output_path)
                    results.append(result)
                    skipped += 1
                    pbar.set_postfix({"skipped": skipped, "current": image_path.name[:20]})
                    continue
                except Exception:
                    pass  # Re-process if loading fails

            pbar.set_postfix({"current": image_path.name[:20]})

            try:
                result = self.analyze_image(image_path, image_index)
                if result is not None:
                    # Save result
                    output_path = self._get_output_path(output_dir, image_path.name)
                    result.save(output_path)
                    results.append(result)
            except Exception as e:
                print(f"\nError processing {image_path.name}: {e}")
                continue

        print(f"\nProcessed: {len(results) - skipped}, Skipped: {skipped}, Total: {len(results)}")

        # Sort results by image index
        results.sort(key=lambda r: r.image_index)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch analysis for crack propagation in image sequences."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing images to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for results (default: input_dir/results)"
    )
    parser.add_argument(
        "--reference-distance",
        type=float,
        default=1.0,
        help="Reference distance in mm for relative displacement (default: 1.0)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=20.0,
        help="Image resolution in µm/pixel (default: 20.0)"
    )
    parser.add_argument(
        "--subset-radius",
        type=int,
        default=20,
        help="DIC subset radius in pixels (default: 20)"
    )
    parser.add_argument(
        "--subset-spacing",
        type=int,
        default=3,
        help="DIC subset spacing in pixels (default: 3)"
    )
    parser.add_argument(
        "--strain-radius",
        type=int,
        default=5,
        help="Strain calculation radius (default: 5)"
    )
    parser.add_argument(
        "--roi-margin",
        type=int,
        default=30,
        help="Margin from image edges for ROI (default: 30)"
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="Post-DIC rotation in degrees (counter-clockwise positive, default: 0)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip already processed images"
    )

    args = parser.parse_args()

    # Create configuration
    config = AnalysisConfig(
        resolution_um_per_px=args.resolution,
        reference_distance_mm=args.reference_distance,
        subset_radius=args.subset_radius,
        subset_spacing=args.subset_spacing,
        strain_radius=args.strain_radius,
        roi_margin=args.roi_margin,
        rotation_deg=args.rotation,
        skip_existing=not args.no_skip,
    )

    # Run analysis
    analyzer = BatchCrackAnalyzer(config)
    results = analyzer.run(args.input_dir, args.output)

    print(f"\nAnalysis complete. {len(results)} images processed.")


if __name__ == "__main__":
    main()
