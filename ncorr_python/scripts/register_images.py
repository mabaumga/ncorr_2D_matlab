#!/usr/bin/env python3
"""
Image registration tool for aligning two images using corresponding points.

This tool corrects for translation, rotation, and scaling between two images
by using 3 or more corresponding reference points/regions.

Usage:
    # Interactive mode - click on corresponding points
    python register_images.py image1.png image2.png --interactive

    # With predefined points (x1,y1 for image1, x2,y2 for image2)
    python register_images.py image1.png image2.png --points "100,100:102,98" "500,100:503,97" "300,400:305,402"

    # Using template matching on specified regions (x,y,radius)
    python register_images.py image1.png image2.png --regions "100,100,25" "500,100,25" "300,400,25"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2


@dataclass
class PointPair:
    """A pair of corresponding points in two images."""
    pt1: Tuple[float, float]  # Point in image 1 (reference)
    pt2: Tuple[float, float]  # Point in image 2 (to be transformed)


class InteractivePointSelector:
    """Interactive GUI for selecting corresponding points in two images."""

    def __init__(self, img1: np.ndarray, img2: np.ndarray, num_points: int = 3):
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        self.num_points = num_points
        self.points1: List[Tuple[int, int]] = []
        self.points2: List[Tuple[int, int]] = []
        self.current_image = 1  # Start with image 1
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_image == 1 and len(self.points1) < self.num_points:
                self.points1.append((x, y))
                print(f"  Point {len(self.points1)} in Image 1: ({x}, {y})")
                if len(self.points1) == self.num_points:
                    self.current_image = 2
                    print(f"\nNow select {self.num_points} corresponding points in Image 2:")
            elif self.current_image == 2 and len(self.points2) < self.num_points:
                self.points2.append((x, y))
                print(f"  Point {len(self.points2)} in Image 2: ({x}, {y})")
                if len(self.points2) == self.num_points:
                    self.done = True

    def draw_points(self, img: np.ndarray, points: List[Tuple[int, int]],
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw points and labels on image."""
        img_draw = img.copy()
        for i, pt in enumerate(points):
            cv2.circle(img_draw, pt, 8, color, 2)
            cv2.circle(img_draw, pt, 2, color, -1)
            cv2.putText(img_draw, str(i + 1), (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img_draw

    def run(self) -> List[PointPair]:
        """Run the interactive point selection."""
        window_name = "Select Corresponding Points (Press 'q' to quit, 'r' to reset)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print(f"\nSelect {self.num_points} corresponding points in both images.")
        print("These should be clearly identifiable features OUTSIDE the crack region.")
        print(f"\nFirst, select {self.num_points} points in Image 1 (Reference):")

        while not self.done:
            # Create display image
            if self.current_image == 1:
                display = self.draw_points(self.img1, self.points1, (0, 255, 0))
                title = f"IMAGE 1 (Reference) - Select point {len(self.points1) + 1}/{self.num_points}"
            else:
                display = self.draw_points(self.img2, self.points2, (0, 0, 255))
                title = f"IMAGE 2 (To transform) - Select point {len(self.points2) + 1}/{self.num_points}"

            # Add title to image
            cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return []
            elif key == ord('r'):
                # Reset
                self.points1 = []
                self.points2 = []
                self.current_image = 1
                print("\nReset. Select points again:")

        cv2.destroyAllWindows()

        # Create point pairs
        pairs = [PointPair(pt1=p1, pt2=p2)
                for p1, p2 in zip(self.points1, self.points2)]
        return pairs


def find_region_correspondence(img1: np.ndarray, img2: np.ndarray,
                                region: Tuple[int, int, int],
                                search_margin: int = 50,
                                min_quality: float = 0.5) -> Optional[PointPair]:
    """
    Find corresponding point in img2 for a region in img1 using template matching.

    Args:
        img1: Reference image
        img2: Image to search in
        region: (x, y, radius) - center point and radius of circular region in img1
        search_margin: How far to search around the original position
        min_quality: Minimum match quality threshold (0.0-1.0)

    Returns:
        PointPair with the center of matched regions, or None if failed
    """
    x, y, r = region

    # Convert center + radius to bounding box
    # Template is a square with side length 2*r
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(img1.shape[1], x + r)
    y2 = min(img1.shape[0], y + r)

    w = x2 - x1
    h = y2 - y1

    # Extract template from img1
    template = img1[y1:y2, x1:x2]

    if template.size == 0:
        print(f"  Warning: Empty template for region ({x}, {y}, r={r})")
        return None

    # Define search area in img2 (expanded region)
    search_x1 = max(0, x - r - search_margin)
    search_y1 = max(0, y - r - search_margin)
    search_x2 = min(img2.shape[1], x + r + search_margin)
    search_y2 = min(img2.shape[0], y + r + search_margin)

    search_area = img2[search_y1:search_y2, search_x1:search_x2]

    if search_area.shape[0] < h or search_area.shape[1] < w:
        print(f"  Warning: Search area too small for region ({x}, {y}, r={r})")
        return None

    # Template matching
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
        search_gray = search_area

    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < min_quality:
        print(f"  Warning: Low match quality ({max_val:.2f} < {min_quality}) for region ({x}, {y}, r={r})")
        return None

    # Calculate center points
    # pt1 is the original center
    pt1_center = (x, y)
    # pt2 is where the template was found + offset to center
    pt2_x = search_x1 + max_loc[0] + w // 2
    pt2_y = search_y1 + max_loc[1] + h // 2
    pt2_center = (pt2_x, pt2_y)

    print(f"  Region ({x}, {y}, r={r}): matched at ({pt2_x}, {pt2_y}), quality={max_val:.3f}")

    return PointPair(pt1=pt1_center, pt2=pt2_center)


def compute_transformation(pairs: List[PointPair],
                           use_affine: bool = True) -> Tuple[np.ndarray, str]:
    """
    Compute transformation matrix from point pairs.

    Args:
        pairs: List of corresponding point pairs
        use_affine: If True, use affine (3 points). If False, use homography (4+ points)

    Returns:
        Transformation matrix and type string
    """
    pts1 = np.float32([p.pt1 for p in pairs])
    pts2 = np.float32([p.pt2 for p in pairs])

    if len(pairs) == 3 and use_affine:
        # Affine transformation (handles translation, rotation, scaling, shearing)
        M = cv2.getAffineTransform(pts2, pts1)
        return M, "affine"
    elif len(pairs) >= 4:
        # Perspective transformation (homography)
        M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        return M, "perspective"
    else:
        # Estimate affine with more than 3 points using least squares
        M, inliers = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC)
        return M, "affine"


def apply_transformation(img: np.ndarray, M: np.ndarray,
                         output_size: Tuple[int, int],
                         transform_type: str) -> np.ndarray:
    """Apply transformation to image."""
    if transform_type == "affine":
        return cv2.warpAffine(img, M, output_size,
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    else:  # perspective
        return cv2.warpPerspective(img, M, output_size,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)


def visualize_registration(img1: np.ndarray, img2: np.ndarray,
                           img2_registered: np.ndarray,
                           pairs: List[PointPair],
                           output_path: Path = None):
    """Visualize the registration result."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Original images and overlay
    axes[0, 0].imshow(img1, cmap='gray' if img1.ndim == 2 else None)
    axes[0, 0].set_title("Image 1 (Reference)")
    for i, p in enumerate(pairs):
        axes[0, 0].plot(p.pt1[0], p.pt1[1], 'go', markersize=10)
        axes[0, 0].annotate(str(i+1), p.pt1, color='green', fontsize=12)

    axes[0, 1].imshow(img2, cmap='gray' if img2.ndim == 2 else None)
    axes[0, 1].set_title("Image 2 (Original)")
    for i, p in enumerate(pairs):
        axes[0, 1].plot(p.pt2[0], p.pt2[1], 'ro', markersize=10)
        axes[0, 1].annotate(str(i+1), p.pt2, color='red', fontsize=12)

    axes[0, 2].imshow(img2_registered, cmap='gray' if img2_registered.ndim == 2 else None)
    axes[0, 2].set_title("Image 2 (Registered)")

    # Row 2: Difference images
    # Convert to grayscale for difference
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(float)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(float)
        gray2_reg = cv2.cvtColor(img2_registered, cv2.COLOR_BGR2GRAY).astype(float)
    else:
        gray1 = img1.astype(float)
        gray2 = img2.astype(float)
        gray2_reg = img2_registered.astype(float)

    diff_before = gray1 - gray2
    diff_after = gray1 - gray2_reg

    vmax = max(np.abs(diff_before).max(), np.abs(diff_after).max())

    axes[1, 0].imshow(diff_before, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title(f"Difference BEFORE registration\nRMS: {np.sqrt(np.mean(diff_before**2)):.2f}")

    axes[1, 1].imshow(diff_after, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title(f"Difference AFTER registration\nRMS: {np.sqrt(np.mean(diff_after**2)):.2f}")

    # Overlay (checkerboard)
    h, w = gray1.shape
    checker_size = 50
    checker = np.zeros_like(gray1)
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                checker[i:i+checker_size, j:j+checker_size] = gray1[i:i+checker_size, j:j+checker_size]
            else:
                checker[i:i+checker_size, j:j+checker_size] = gray2_reg[i:i+checker_size, j:j+checker_size]

    axes[1, 2].imshow(checker, cmap='gray')
    axes[1, 2].set_title("Checkerboard overlay (registered)")

    for ax in axes.flat:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def parse_point_pair(s: str) -> PointPair:
    """Parse point pair from string format 'x1,y1:x2,y2'."""
    parts = s.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid point pair format: {s}")

    p1 = tuple(map(float, parts[0].split(',')))
    p2 = tuple(map(float, parts[1].split(',')))

    if len(p1) != 2 or len(p2) != 2:
        raise ValueError(f"Invalid coordinates in: {s}")

    return PointPair(pt1=p1, pt2=p2)


def parse_region(s: str) -> Tuple[int, int, int]:
    """Parse region from string format 'x,y,radius'."""
    parts = list(map(int, s.split(',')))
    if len(parts) != 3:
        raise ValueError(f"Invalid region format: {s} (expected 'x,y,radius')")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Register two images using corresponding points.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive point selection
  python register_images.py ref.png target.png --interactive

  # With predefined point pairs (x1,y1:x2,y2)
  python register_images.py ref.png target.png \\
      --points "100,100:102,98" "500,100:503,97" "300,400:305,402"

  # Using template matching on regions (x,y,radius)
  python register_images.py ref.png target.png \\
      --regions "100,100,25" "500,100,25" "300,400,25"
"""
    )
    parser.add_argument(
        "image1",
        type=Path,
        help="Reference image (target geometry)"
    )
    parser.add_argument(
        "image2",
        type=Path,
        help="Image to be transformed"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for registered image (default: image2_registered.ext)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactively select corresponding points"
    )
    parser.add_argument(
        "--points", "-p",
        nargs='+',
        type=str,
        default=None,
        help="Point pairs in format 'x1,y1:x2,y2' (at least 3)"
    )
    parser.add_argument(
        "--regions", "-r",
        nargs='+',
        type=str,
        default=None,
        help="Regions for template matching in format 'x,y,radius' (at least 3)"
    )
    parser.add_argument(
        "--search-margin",
        type=int,
        default=100,
        help="Search margin for template matching (default: 100)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum match quality threshold 0.0-1.0 (default: 0.5)"
    )
    parser.add_argument(
        "--num-points", "-n",
        type=int,
        default=3,
        help="Number of points to select in interactive mode (default: 3)"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show visualization of registration result"
    )
    parser.add_argument(
        "--save-visualization",
        type=Path,
        default=None,
        help="Save visualization to file"
    )

    args = parser.parse_args()

    # Check inputs
    if not args.image1.exists():
        print(f"Error: Image not found: {args.image1}")
        sys.exit(1)
    if not args.image2.exists():
        print(f"Error: Image not found: {args.image2}")
        sys.exit(1)

    # Load images
    print(f"Loading images...")
    img1 = cv2.imread(str(args.image1))
    img2 = cv2.imread(str(args.image2))

    if img1 is None:
        print(f"Error: Could not load image: {args.image1}")
        sys.exit(1)
    if img2 is None:
        print(f"Error: Could not load image: {args.image2}")
        sys.exit(1)

    print(f"  Image 1: {img1.shape[1]}x{img1.shape[0]}")
    print(f"  Image 2: {img2.shape[1]}x{img2.shape[0]}")

    # Get point pairs
    pairs: List[PointPair] = []

    if args.interactive:
        selector = InteractivePointSelector(img1, img2, args.num_points)
        pairs = selector.run()
        if not pairs:
            print("No points selected. Exiting.")
            sys.exit(1)

    elif args.points:
        print(f"\nUsing predefined point pairs:")
        for p_str in args.points:
            pair = parse_point_pair(p_str)
            pairs.append(pair)
            print(f"  {pair.pt1} -> {pair.pt2}")

    elif args.regions:
        print(f"\nFinding correspondences using template matching:")
        for r_str in args.regions:
            region = parse_region(r_str)
            pair = find_region_correspondence(img1, img2, region, args.search_margin, args.min_quality)
            if pair:
                pairs.append(pair)

    else:
        print("Error: Specify --interactive, --points, or --regions")
        sys.exit(1)

    if len(pairs) < 3:
        print(f"Error: Need at least 3 point pairs, got {len(pairs)}")
        sys.exit(1)

    print(f"\nUsing {len(pairs)} point pairs for registration")

    # Compute transformation
    print("\nComputing transformation...")
    M, transform_type = compute_transformation(pairs, use_affine=(len(pairs) == 3))
    print(f"  Transformation type: {transform_type}")

    if transform_type == "affine":
        # Extract transformation parameters
        # Affine matrix: [[a, b, tx], [c, d, ty]]
        a, b, tx = M[0]
        c, d, ty = M[1]
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)
        rotation = np.arctan2(c, a) * 180 / np.pi
        print(f"  Scale X: {scale_x:.4f}")
        print(f"  Scale Y: {scale_y:.4f}")
        print(f"  Rotation: {rotation:.2f} deg")
        print(f"  Translation: ({tx:.1f}, {ty:.1f}) px")

    # Apply transformation
    print("\nApplying transformation...")
    output_size = (img1.shape[1], img1.shape[0])
    img2_registered = apply_transformation(img2, M, output_size, transform_type)

    # Save result
    if args.output is None:
        output_path = args.image2.parent / f"{args.image2.stem}_registered{args.image2.suffix}"
    else:
        output_path = args.output

    cv2.imwrite(str(output_path), img2_registered)
    print(f"\nRegistered image saved to: {output_path}")

    # Save transformation matrix
    matrix_path = output_path.parent / f"{output_path.stem}_transform.txt"
    np.savetxt(matrix_path, M, fmt='%.6f', header=f"Transformation matrix ({transform_type})")
    print(f"Transformation matrix saved to: {matrix_path}")

    # Visualize if requested
    if args.visualize or args.save_visualization:
        print("\nCreating visualization...")
        visualize_registration(img1, img2, img2_registered, pairs, args.save_visualization)

    print("\nDone.")


if __name__ == "__main__":
    main()
