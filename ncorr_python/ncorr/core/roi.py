"""
Region of Interest (ROI) class for Ncorr.

Equivalent to MATLAB's ncorr_class_roi.m
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray


class ROIType(str, Enum):
    """Types of ROI creation."""

    LOAD = "load"        # Mask loaded from image file
    DRAW = "draw"        # Mask was drawn
    REGION = "region"    # Region provided directly
    BOUNDARY = "boundary"  # Boundary provided directly


@dataclass
class Region:
    """
    Represents a connected region within an ROI.

    The region uses a run-length encoding (RLE) based on node lists and ranges.
    For each column (x-coordinate), we store pairs of (top, bottom) y-coordinates
    that define contiguous segments within the region.

    Attributes:
        nodelist: Array of y-coordinate pairs for each column
        noderange: Number of entries used in nodelist for each column
        leftbound: Left boundary x-coordinate
        rightbound: Right boundary x-coordinate
        upperbound: Upper boundary y-coordinate
        lowerbound: Lower boundary y-coordinate
        totalpoints: Total number of pixels in region
    """

    nodelist: NDArray[np.int32] = field(default_factory=lambda: np.array([[-1, -1]], dtype=np.int32))
    noderange: NDArray[np.int32] = field(default_factory=lambda: np.array([0], dtype=np.int32))
    leftbound: int = 0
    rightbound: int = 0
    upperbound: int = 0
    lowerbound: int = 0
    totalpoints: int = 0

    def is_empty(self) -> bool:
        """Check if region is empty."""
        return self.totalpoints == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "nodelist": self.nodelist.astype(np.int32),
            "noderange": self.noderange.astype(np.int32),
            "leftbound": np.int32(self.leftbound),
            "rightbound": np.int32(self.rightbound),
            "upperbound": np.int32(self.upperbound),
            "lowerbound": np.int32(self.lowerbound),
            "totalpoints": np.int32(self.totalpoints),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Region":
        """Create Region from dictionary."""
        return cls(
            nodelist=np.asarray(d["nodelist"], dtype=np.int32),
            noderange=np.asarray(d["noderange"], dtype=np.int32),
            leftbound=int(d["leftbound"]),
            rightbound=int(d["rightbound"]),
            upperbound=int(d["upperbound"]),
            lowerbound=int(d["lowerbound"]),
            totalpoints=int(d["totalpoints"]),
        )

    @classmethod
    def empty_placeholder(cls) -> "Region":
        """Create an empty placeholder region."""
        return cls(
            nodelist=np.array([[-1, -1]], dtype=np.int32),
            noderange=np.array([0], dtype=np.int32),
            leftbound=0,
            rightbound=0,
            upperbound=0,
            lowerbound=0,
            totalpoints=0,
        )


@dataclass
class Boundary:
    """
    Represents boundary contours of a region.

    Attributes:
        add: Outer boundary coordinates (N x 2 array of x, y)
        sub: List of inner boundary coordinates (holes)
    """

    add: NDArray[np.float64] = field(default_factory=lambda: np.array([[-1, -1]], dtype=np.float64))
    sub: List[NDArray[np.float64]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "add": self.add,
            "sub": self.sub,
        }


@dataclass
class CircularROI:
    """
    Circular subset ROI used for DIC analysis.

    Attributes:
        mask: Binary mask of the circular subset (2*radius+1 x 2*radius+1)
        region: Region describing the subset
        boundary: Boundary of the subset (currently unused)
        x: X-coordinate of center
        y: Y-coordinate of center
        radius: Radius of the subset
    """

    mask: NDArray[np.bool_]
    region: Region
    boundary: Optional[NDArray[np.float64]]
    x: int
    y: int
    radius: int


class NcorrROI:
    """
    Region of Interest class for Ncorr DIC analysis.

    This class represents a region of interest (mask) for DIC analysis,
    along with associated region and boundary data.

    Attributes:
        roi_type: How the ROI was created
        mask: Binary mask (logical array)
        regions: List of connected component regions
        boundaries: List of boundary contours
        data: Additional data specific to ROI type
    """

    def __init__(self):
        """Initialize empty ROI."""
        self.roi_type: Optional[ROIType] = None
        self.mask: NDArray[np.bool_] = np.zeros((0, 0), dtype=np.bool_)
        self.regions: List[Region] = []
        self.boundaries: List[Boundary] = []
        self.data: Dict[str, Any] = {}

    @property
    def is_set(self) -> bool:
        """Check if ROI has been set."""
        return self.roi_type is not None

    def set_roi(self, roi_type: str, data: Dict[str, Any]) -> None:
        """
        Set the region of interest.

        Args:
            roi_type: How ROI was created:
                - 'load': Mask loaded from image file
                - 'draw': Mask was drawn
                - 'region': Region provided directly
                - 'boundary': Boundary provided directly
            data: Input data dictionary, contents depend on roi_type:
                - 'load'/'draw': {'mask': ndarray, 'cutoff': int}
                - 'region': {'region': list[Region], 'size_mask': tuple}
                - 'boundary': {'boundary': list[Boundary], 'size_mask': tuple}
        """
        try:
            roi_type_enum = ROIType(roi_type)
        except ValueError:
            raise ValueError(f"Incorrect type provided: {roi_type}")

        self.roi_type = roi_type_enum
        self.data = data

        if roi_type_enum in (ROIType.LOAD, ROIType.DRAW):
            self._set_from_mask(data)
        elif roi_type_enum == ROIType.REGION:
            self._set_from_region(data)
        elif roi_type_enum == ROIType.BOUNDARY:
            self._set_from_boundary(data)

    def _set_from_mask(self, data: Dict[str, Any]) -> None:
        """Set ROI from a mask."""
        from ..algorithms.regions import RegionProcessor

        mask = data["mask"].astype(np.bool_)
        cutoff = data.get("cutoff", 20)

        # Create regions (4-way connected components)
        regions, removed = RegionProcessor.form_regions(mask, cutoff, False)

        # Keep only the 20 largest regions
        max_regions = 20
        if len(regions) > max_regions:
            regions = sorted(regions, key=lambda r: r.totalpoints, reverse=True)[:max_regions]
            removed = True

        # Update mask if regions were removed
        if removed:
            mask = np.zeros_like(mask, dtype=np.bool_)
            for region in regions:
                mask = self._fill_mask_from_region(mask, region)

        # Form boundaries
        boundaries = []
        for i, region in enumerate(regions):
            # Get mask for this specific region
            region_mask = self._get_region_mask(mask.shape, region)

            # Get outer boundary using leftmost top point
            start_point = (region.leftbound, region.nodelist[0, 0])
            add_boundary = RegionProcessor.form_boundary(start_point, 0, region_mask)

            # Get subtraction boundaries (holes)
            sub_boundaries = self._find_holes(region, region_mask)

            boundaries.append(Boundary(add=add_boundary, sub=sub_boundaries))

        self.mask = mask
        self.regions = regions
        self.boundaries = boundaries

    def _set_from_region(self, data: Dict[str, Any]) -> None:
        """Set ROI from regions directly."""
        size_mask = tuple(data["size_mask"])
        regions = data["region"]

        # Convert dicts to Region objects if needed
        if regions and isinstance(regions[0], dict):
            regions = [Region.from_dict(r) for r in regions]

        self.mask = np.zeros(size_mask, dtype=np.bool_)
        for region in regions:
            self.mask = self._fill_mask_from_region(self.mask, region)

        # Create placeholder boundaries
        self.boundaries = [Boundary(add=np.array([[-1, -1]]), sub=[]) for _ in regions]
        self.regions = regions

    def _set_from_boundary(self, data: Dict[str, Any]) -> None:
        """Set ROI from boundaries."""
        from ..algorithms.regions import RegionProcessor

        size_mask = tuple(data["size_mask"])
        boundaries = data["boundary"]

        self.mask = np.zeros(size_mask, dtype=np.bool_)
        regions = []

        for boundary in boundaries:
            if isinstance(boundary, dict):
                boundary = Boundary(
                    add=boundary["add"],
                    sub=boundary.get("sub", [])
                )

            # Form mask from boundary
            boundary_mask = RegionProcessor.form_mask_from_boundary(
                boundary, size_mask
            )
            self.mask |= boundary_mask

            # Get region from mask
            region_list, _ = RegionProcessor.form_regions(boundary_mask, 0, False)

            if not region_list:
                regions.append(Region.empty_placeholder())
            elif len(region_list) > 1:
                # Select largest region
                largest = max(region_list, key=lambda r: r.totalpoints)
                regions.append(largest)
            else:
                regions.append(region_list[0])

        # Update mask from regions
        self.mask = np.zeros(size_mask, dtype=np.bool_)
        for region in regions:
            self.mask = self._fill_mask_from_region(self.mask, region)

        self.regions = regions
        self.boundaries = boundaries if isinstance(boundaries[0], Boundary) else [
            Boundary(add=b["add"], sub=b.get("sub", [])) for b in boundaries
        ]

    @staticmethod
    def _fill_mask_from_region(mask: NDArray[np.bool_], region: Region) -> NDArray[np.bool_]:
        """Fill mask pixels based on region node data."""
        if region.is_empty():
            return mask

        for j in range(region.noderange.shape[0]):
            x = j + region.leftbound
            for k in range(0, region.noderange[j], 2):
                y_start = region.nodelist[j, k]
                y_end = region.nodelist[j, k + 1]
                mask[y_start:y_end + 1, x] = True

        return mask

    @staticmethod
    def _get_region_mask(mask_shape: Tuple[int, int], region: Region) -> NDArray[np.bool_]:
        """Create mask for a single region."""
        mask = np.zeros(mask_shape, dtype=np.bool_)
        return NcorrROI._fill_mask_from_region(mask, region)

    def _find_holes(
        self, region: Region, region_mask: NDArray[np.bool_]
    ) -> List[NDArray[np.float64]]:
        """Find hole boundaries within a region."""
        from ..algorithms.regions import RegionProcessor

        holes = []

        # Get inverse mask within region bounds
        filled_mask = RegionProcessor.fill_polygon(
            self.boundaries[-1].add if self.boundaries else np.array([]),
            region_mask.shape
        )

        inv_mask = filled_mask & ~region_mask

        # Track analyzed boundary points
        analyzed = np.zeros_like(region_mask, dtype=np.bool_)

        # Scan for holes
        for j in range(region.noderange.shape[0]):
            x = j + region.leftbound
            if region.noderange[j] > 2:
                for k in range(0, region.noderange[j] - 2, 2):
                    y_bottom = region.nodelist[j, k + 1]

                    # Check one pixel below
                    if (y_bottom + 1 < region_mask.shape[0] and
                        inv_mask[y_bottom + 1, x] and
                        not analyzed[y_bottom + 1, x]):

                        hole_boundary = RegionProcessor.form_boundary(
                            (x, y_bottom + 1), 0, inv_mask
                        )
                        holes.append(hole_boundary)

                        # Mark as analyzed
                        for pt in hole_boundary:
                            analyzed[int(pt[1]), int(pt[0])] = True

        return holes

    def reduce(self, spacing: int) -> "NcorrROI":
        """
        Reduce the ROI by a spacing factor.

        Some regions may become empty after reduction but placeholders
        are used to preserve region count.

        Args:
            spacing: Spacing parameter (0 = no reduction)

        Returns:
            Reduced NcorrROI instance
        """
        if not self.is_set:
            raise RuntimeError("ROI has not been set yet")

        if spacing == 0:
            # Return deep copy
            reduced = NcorrROI()
            reduced.roi_type = self.roi_type
            reduced.mask = self.mask.copy()
            reduced.regions = [Region.from_dict(r.to_dict()) for r in self.regions]
            reduced.boundaries = self.boundaries.copy()
            reduced.data = self.data.copy()
            return reduced

        step = spacing + 1
        reduced_regions = []

        for region in self.regions:
            reduced_region = self._reduce_region(region, step)
            reduced_regions.append(reduced_region)

        # Create new ROI
        reduced = NcorrROI()
        new_mask_shape = (
            (self.mask.shape[0] + step - 1) // step,
            (self.mask.shape[1] + step - 1) // step,
        )
        reduced.set_roi(
            "region",
            {"region": reduced_regions, "size_mask": new_mask_shape}
        )

        return reduced

    def _reduce_region(self, region: Region, step: int) -> Region:
        """Reduce a single region by step factor."""
        if region.is_empty():
            return Region.empty_placeholder()

        # Calculate new bounds
        new_left = int(np.ceil(region.leftbound / step))
        new_right = int(np.floor(region.rightbound / step))

        if new_right < new_left:
            return Region.empty_placeholder()

        width = new_right - new_left + 1
        max_pairs = region.nodelist.shape[1]

        new_nodelist = -np.ones((width, max_pairs), dtype=np.int32)
        new_noderange = np.zeros(width, dtype=np.int32)
        totalpoints = 0
        upperbound = float("inf")
        lowerbound = float("-inf")

        for i, x_new in enumerate(range(new_left, new_right + 1)):
            x_old = x_new * step - region.leftbound

            if 0 <= x_old < region.noderange.shape[0]:
                for k in range(0, region.noderange[x_old], 2):
                    node_top = int(np.ceil(region.nodelist[x_old, k] / step))
                    node_bottom = int(np.floor(region.nodelist[x_old, k + 1] / step))

                    if node_bottom >= node_top:
                        idx = new_noderange[i]
                        new_nodelist[i, idx] = node_top
                        new_nodelist[i, idx + 1] = node_bottom
                        new_noderange[i] += 2
                        totalpoints += node_bottom - node_top + 1
                        upperbound = min(upperbound, node_top)
                        lowerbound = max(lowerbound, node_bottom)

        if totalpoints == 0:
            return Region.empty_placeholder()

        return Region(
            nodelist=new_nodelist,
            noderange=new_noderange,
            leftbound=new_left,
            rightbound=new_right,
            upperbound=int(upperbound),
            lowerbound=int(lowerbound),
            totalpoints=totalpoints,
        )

    def get_union(self, mask: NDArray[np.bool_], spacing: int) -> "NcorrROI":
        """
        Get union of this ROI with another mask.

        Args:
            mask: Mask to union with (may be reduced)
            spacing: Reduction spacing factor

        Returns:
            Unioned NcorrROI
        """
        if not self.is_set:
            raise RuntimeError("ROI has not been set yet")

        from ..algorithms.regions import RegionProcessor

        step = spacing + 1
        reduced_mask = self.mask[::step, ::step]

        if reduced_mask.shape != mask.shape:
            raise ValueError("Reduced mask and input mask sizes don't match")

        # Get reduced ROI
        roi_reduced = self.reduce(spacing)

        # Union masks
        union_mask = reduced_mask & mask

        # Get unioned regions
        union_regions = RegionProcessor.form_union(roi_reduced, union_mask)

        # Create new ROI
        roi_union = NcorrROI()
        roi_union.set_roi("region", {"region": union_regions, "size_mask": union_mask.shape})

        return roi_union

    def get_num_region(
        self, x: int, y: int, skip_regions: Optional[NDArray[np.bool_]] = None
    ) -> Tuple[int, int]:
        """
        Determine which region contains point (x, y).

        Args:
            x: X-coordinate
            y: Y-coordinate
            skip_regions: Boolean array indicating regions to skip

        Returns:
            Tuple of (region_index, nodelist_index) or (-1, 0) if not found
        """
        if not self.is_set:
            raise RuntimeError("ROI has not been set yet")

        if skip_regions is None:
            skip_regions = np.zeros(len(self.regions), dtype=np.bool_)

        for i, region in enumerate(self.regions):
            if skip_regions[i]:
                continue

            if region.leftbound <= x <= region.rightbound:
                idx = x - region.leftbound
                for j in range(0, region.noderange[idx], 2):
                    if region.nodelist[idx, j] <= y <= region.nodelist[idx, j + 1]:
                        return i, j

        return -1, 0

    def get_region_mask(self, num_region: int) -> NDArray[np.bool_]:
        """
        Get mask for a specific region.

        Args:
            num_region: Index of region

        Returns:
            Boolean mask for the specified region
        """
        if not self.is_set:
            raise RuntimeError("ROI has not been set yet")

        return self._get_region_mask(self.mask.shape, self.regions[num_region])

    def get_full_regions_count(self) -> int:
        """Get count of non-empty regions."""
        if not self.is_set:
            raise RuntimeError("ROI has not been set yet")

        return sum(1 for r in self.regions if r.totalpoints > 0)

    def get_circular_roi(
        self,
        x: int,
        y: int,
        radius: int,
        subset_trunc: bool = False,
    ) -> CircularROI:
        """
        Get circular subset ROI centered at (x, y).

        Args:
            x: X-coordinate of center
            y: Y-coordinate of center
            radius: Subset radius
            subset_trunc: Enable subset truncation for discontinuities

        Returns:
            CircularROI structure
        """
        if not self.is_set:
            raise RuntimeError("ROI has not been set yet")

        num_region, idx_nodelist = self.get_num_region(x, y)
        if num_region == -1:
            raise ValueError("x and y coordinates are not within the ROI")

        region = self.regions[num_region]
        size = 2 * radius + 1

        # Initialize circular mask
        circ_mask = np.zeros((size, size), dtype=np.bool_)

        # Initialize region structure
        max_width = max(r.nodelist.shape[1] for r in self.regions)
        circ_nodelist = -np.ones((size, max_width), dtype=np.int32)
        circ_noderange = np.zeros(size, dtype=np.int32)
        totalpoints = 0
        is_truncated = False

        # Active lines tracker
        region_width = region.rightbound - region.leftbound + 1
        active = np.ones((region_width, max_width // 2), dtype=np.bool_)

        # Start with center column
        node_top, node_bottom, is_truncated = self._get_initial_nodes(
            region, x, y, radius, idx_nodelist
        )

        # Update mask and queue
        circ_mask[node_top - (y - radius):node_bottom - (y - radius) + 1, radius] = True

        # BFS to fill circular region
        queue = [(node_top, node_bottom, radius)]
        active[x - region.leftbound, idx_nodelist // 2] = False

        while queue:
            curr_top, curr_bottom, curr_x = queue.pop(0)

            # Add to region structure
            idx = circ_noderange[curr_x]
            circ_nodelist[curr_x, idx:idx + 2] = [curr_top, curr_bottom]
            circ_noderange[curr_x] += 2
            totalpoints += curr_bottom - curr_top + 1

            # Check left neighbor
            if curr_x > 0:
                new_nodes = self._check_neighbors(
                    region, x, y, radius, curr_x - 1, curr_top, curr_bottom,
                    active, circ_mask, is_truncated
                )
                for nt, nb, trunc in new_nodes:
                    queue.append((nt, nb, curr_x - 1))
                    is_truncated = is_truncated or trunc

            # Check right neighbor
            if curr_x < size - 1:
                new_nodes = self._check_neighbors(
                    region, x, y, radius, curr_x + 1, curr_top, curr_bottom,
                    active, circ_mask, is_truncated
                )
                for nt, nb, trunc in new_nodes:
                    queue.append((nt, nb, curr_x + 1))
                    is_truncated = is_truncated or trunc

        # Build circular region
        circ_region = Region(
            nodelist=circ_nodelist,
            noderange=circ_noderange,
            leftbound=x - radius,
            rightbound=x + radius,
            upperbound=int(circ_nodelist[circ_noderange > 0, 0].min()),
            lowerbound=int(max(circ_nodelist[i, circ_noderange[i] - 1]
                               for i in range(size) if circ_noderange[i] > 0)),
            totalpoints=totalpoints,
        )

        # Handle subset truncation if needed
        if subset_trunc and is_truncated:
            circ_mask, circ_region = self._truncate_subset(
                circ_mask, circ_region, x, y, radius
            )

        return CircularROI(
            mask=circ_mask,
            region=circ_region,
            boundary=None,
            x=x,
            y=y,
            radius=radius,
        )

    def _get_initial_nodes(
        self,
        region: Region,
        x: int,
        y: int,
        radius: int,
        idx_nodelist: int,
    ) -> Tuple[int, int, bool]:
        """Get initial nodes for circular ROI."""
        idx = x - region.leftbound
        is_truncated = False

        node_top = region.nodelist[idx, idx_nodelist]
        node_bottom = region.nodelist[idx, idx_nodelist + 1]

        if node_top < y - radius:
            node_top = y - radius
        else:
            is_truncated = True

        if node_bottom > y + radius:
            node_bottom = y + radius
        else:
            is_truncated = True

        return node_top, node_bottom, is_truncated

    def _check_neighbors(
        self,
        region: Region,
        x: int,
        y: int,
        radius: int,
        local_x: int,
        curr_top: int,
        curr_bottom: int,
        active: NDArray[np.bool_],
        circ_mask: NDArray[np.bool_],
        is_truncated: bool,
    ) -> List[Tuple[int, int, bool]]:
        """Check neighboring columns for connected nodes."""
        results = []

        global_x = local_x + (x - radius)
        idx_region = global_x - region.leftbound

        if idx_region < 0 or idx_region >= region.noderange.shape[0]:
            return results

        # Calculate circle limits at this column
        dx = abs(global_x - x)
        if dx > radius:
            return results

        circle_half = np.sqrt(radius**2 - dx**2)
        upper_lim = int(np.ceil(y - circle_half))
        lower_lim = int(np.floor(y + circle_half))

        for k in range(0, region.noderange[idx_region], 2):
            if not active[idx_region, k // 2]:
                continue

            node_top = region.nodelist[idx_region, k]
            node_bottom = region.nodelist[idx_region, k + 1]

            # Check for overlap with current nodes
            if node_bottom < curr_top or node_top > curr_bottom:
                continue

            trunc = False

            # Clamp to circle
            if node_top < upper_lim:
                node_top = upper_lim
            else:
                trunc = True

            if node_bottom > lower_lim:
                node_bottom = lower_lim
            else:
                trunc = True

            # Validate
            if node_top > node_bottom:
                continue
            if node_top > curr_bottom or node_bottom < curr_top:
                continue

            # Add to results
            results.append((node_top, node_bottom, trunc))
            active[idx_region, k // 2] = False

            # Update mask
            mask_y_start = node_top - (y - radius)
            mask_y_end = node_bottom - (y - radius) + 1
            circ_mask[mask_y_start:mask_y_end, local_x] = True

        return results

    def _truncate_subset(
        self,
        mask: NDArray[np.bool_],
        region: Region,
        x: int,
        y: int,
        radius: int,
    ) -> Tuple[NDArray[np.bool_], Region]:
        """Truncate subset for discontinuity handling."""
        from ..algorithms.regions import RegionProcessor

        # Find boundary
        boundary = RegionProcessor.form_boundary_from_mask(mask)

        if len(boundary) == 0:
            return mask, region

        # Find closest point to center
        center = np.array([radius, radius])
        dists = np.sum((boundary - center) ** 2, axis=1)
        idx_min = np.argmin(dists)
        min_dist = np.sqrt(dists[idx_min])

        # Check if closest point is on the circular edge
        if np.ceil(min_dist) >= radius:
            return mask, region

        # Calculate tangent direction at closest point
        idx_space = 3
        n = len(boundary)
        idx_plus = (idx_min + idx_space) % n
        idx_minus = (idx_min - idx_space) % n

        p0 = boundary[idx_minus]
        p1 = boundary[idx_plus]

        # Determine which side to keep
        p_center = center
        sign_keep = np.sign(
            (p1[0] - p0[0]) * (p_center[1] - p0[1]) -
            (p_center[0] - p0[0]) * (p1[1] - p0[1])
        )

        # Clear points on wrong side
        new_mask = mask.copy()
        for i in range(mask.shape[1]):
            for j in range(mask.shape[0]):
                if mask[j, i]:
                    p = np.array([i, j])
                    side = np.sign(
                        (p1[0] - p0[0]) * (p[1] - p0[1]) -
                        (p[0] - p0[0]) * (p1[1] - p0[1])
                    )
                    if side != sign_keep and side != 0:
                        new_mask[j, i] = False

        # Rebuild region from mask
        regions, _ = RegionProcessor.form_regions(new_mask, 0, True)

        if not regions:
            return mask, region

        # Keep largest region
        largest = max(regions, key=lambda r: r.totalpoints)

        # Adjust coordinates back to global
        largest.nodelist += (y - radius)
        largest.leftbound += (x - radius)
        largest.rightbound += (x - radius)
        largest.upperbound += (y - radius)
        largest.lowerbound += (y - radius)

        return new_mask, largest

    def formatted(self) -> "NcorrROI":
        """
        Get formatted ROI for computation functions.

        Ensures all numeric arrays are int32 for compatibility.

        Returns:
            Formatted NcorrROI copy
        """
        if not self.is_set:
            raise RuntimeError("ROI has not been set yet")

        formatted = NcorrROI()
        formatted.roi_type = self.roi_type
        formatted.mask = self.mask.copy()
        formatted.data = self.data.copy()
        formatted.boundaries = self.boundaries.copy()

        formatted.regions = []
        for region in self.regions:
            formatted.regions.append(Region(
                nodelist=region.nodelist.astype(np.int32),
                noderange=region.noderange.astype(np.int32),
                leftbound=int(region.leftbound),
                rightbound=int(region.rightbound),
                upperbound=int(region.upperbound),
                lowerbound=int(region.lowerbound),
                totalpoints=int(region.totalpoints),
            ))

        return formatted

    def to_dict(self) -> Dict[str, Any]:
        """Convert ROI to dictionary."""
        return {
            "type": self.roi_type.value if self.roi_type else None,
            "mask": self.mask,
            "region": [r.to_dict() for r in self.regions],
            "boundary": [b.to_dict() for b in self.boundaries],
            "data": self.data,
        }
