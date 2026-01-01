"""
Region and boundary processing algorithms.

Equivalent to ncorr_alg_formregions.cpp, ncorr_alg_formboundary.cpp,
ncorr_alg_formmask.cpp, ncorr_alg_formunion.cpp
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from numba import jit

if TYPE_CHECKING:
    from ..core.roi import Region, Boundary, NcorrROI


class RegionProcessor:
    """
    Processor for region detection and manipulation.

    Provides algorithms for finding connected components, tracing boundaries,
    forming masks from boundaries, and computing region unions.
    """

    @staticmethod
    def form_regions(
        mask: NDArray[np.bool_],
        min_points: int = 20,
        fixed_width: bool = False,
    ) -> Tuple[List["Region"], bool]:
        """
        Find 4-way connected regions in a binary mask.

        Args:
            mask: Binary mask image
            min_points: Minimum number of points for a valid region
            fixed_width: If True, all regions have noderange width = mask width

        Returns:
            Tuple of (list of Region objects, removed flag)
        """
        from ..core.roi import Region

        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=np.bool_)
        regions = []
        removed = False

        for start_y in range(h):
            for start_x in range(w):
                if mask[start_y, start_x] and not visited[start_y, start_x]:
                    # BFS to find connected component
                    region_mask = np.zeros_like(mask, dtype=np.bool_)
                    queue = deque([(start_x, start_y)])
                    visited[start_y, start_x] = True
                    region_mask[start_y, start_x] = True

                    min_x, max_x = start_x, start_x
                    min_y, max_y = start_y, start_y
                    point_count = 1

                    while queue:
                        cx, cy = queue.popleft()

                        # 4-way neighbors
                        for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                            if 0 <= nx < w and 0 <= ny < h:
                                if mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    region_mask[ny, nx] = True
                                    queue.append((nx, ny))
                                    point_count += 1
                                    min_x = min(min_x, nx)
                                    max_x = max(max_x, nx)
                                    min_y = min(min_y, ny)
                                    max_y = max(max_y, ny)

                    # Check minimum size
                    if point_count < min_points:
                        removed = True
                        continue

                    # Build region structure
                    region = RegionProcessor._build_region_from_mask(
                        region_mask, min_x, max_x, min_y, max_y, point_count, fixed_width
                    )
                    regions.append(region)

        return regions, removed

    @staticmethod
    def _build_region_from_mask(
        region_mask: NDArray[np.bool_],
        left: int,
        right: int,
        top: int,
        bottom: int,
        total: int,
        fixed_width: bool,
    ) -> "Region":
        """Build Region structure from a binary mask."""
        from ..core.roi import Region

        if fixed_width:
            width = region_mask.shape[1]
            left_bound = 0
        else:
            width = right - left + 1
            left_bound = left

        # Count max node pairs per column
        max_pairs = 0
        for x in range(left, right + 1):
            in_segment = False
            pairs = 0
            for y in range(region_mask.shape[0]):
                if region_mask[y, x] and not in_segment:
                    in_segment = True
                elif not region_mask[y, x] and in_segment:
                    in_segment = False
                    pairs += 1
            if in_segment:
                pairs += 1
            max_pairs = max(max_pairs, pairs)

        # Allocate arrays
        nodelist = -np.ones((width, max_pairs * 2), dtype=np.int32)
        noderange = np.zeros(width, dtype=np.int32)

        # Fill arrays
        for i, x in enumerate(range(left, right + 1)):
            if fixed_width:
                i = x

            pair_idx = 0
            in_segment = False
            start_y = 0

            for y in range(region_mask.shape[0]):
                if region_mask[y, x] and not in_segment:
                    in_segment = True
                    start_y = y
                elif not region_mask[y, x] and in_segment:
                    in_segment = False
                    nodelist[i, pair_idx * 2] = start_y
                    nodelist[i, pair_idx * 2 + 1] = y - 1
                    pair_idx += 1

            if in_segment:
                nodelist[i, pair_idx * 2] = start_y
                nodelist[i, pair_idx * 2 + 1] = region_mask.shape[0] - 1
                pair_idx += 1

            noderange[i] = pair_idx * 2

        return Region(
            nodelist=nodelist,
            noderange=noderange,
            leftbound=left_bound,
            rightbound=right if not fixed_width else region_mask.shape[1] - 1,
            upperbound=top,
            lowerbound=bottom,
            totalpoints=total,
        )

    @staticmethod
    def form_boundary(
        start_point: Tuple[int, int],
        start_direction: int,
        mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """
        Trace 8-connected boundary starting from a point.

        Uses Moore-Neighbor tracing algorithm.

        Args:
            start_point: Starting (x, y) coordinate on the boundary
            start_direction: Initial direction (0-7, 0=right, counter-clockwise)
            mask: Binary mask

        Returns:
            Nx2 array of boundary coordinates
        """
        # Direction offsets (8-connected, counter-clockwise from right)
        dx = [1, 1, 0, -1, -1, -1, 0, 1]
        dy = [0, -1, -1, -1, 0, 1, 1, 1]

        h, w = mask.shape
        x, y = start_point
        direction = start_direction

        boundary = [(x, y)]
        visited_states = {(x, y, direction)}

        while True:
            # Search for next boundary pixel
            found = False
            for i in range(8):
                check_dir = (direction + i) % 8
                nx = x + dx[check_dir]
                ny = y + dy[check_dir]

                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                    # Found next pixel
                    x, y = nx, ny
                    # Update direction to come from the opposite side
                    direction = (check_dir + 5) % 8
                    found = True
                    break

            if not found:
                break

            # Check if we've completed the loop
            state = (x, y, direction)
            if state in visited_states:
                break

            visited_states.add(state)
            boundary.append((x, y))

            # Safety limit
            if len(boundary) > w * h:
                break

        return np.array(boundary, dtype=np.float64)

    @staticmethod
    def form_boundary_from_mask(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        """
        Find and trace the outer boundary of a mask.

        Args:
            mask: Binary mask

        Returns:
            Boundary coordinates
        """
        # Find first point (top-left)
        for x in range(mask.shape[1]):
            for y in range(mask.shape[0]):
                if mask[y, x]:
                    return RegionProcessor.form_boundary((x, y), 7, mask)

        return np.array([], dtype=np.float64).reshape(0, 2)

    @staticmethod
    def form_mask(
        draw_objects: List[dict],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """
        Form mask from drawing objects (polygons).

        Args:
            draw_objects: List of drawing objects with keys:
                - 'pos_imroi': Nx2 array of polygon vertices
                - 'type': 'poly' for polygon
                - 'addorsub': 'add' or 'sub'
            mask: Initial mask to modify

        Returns:
            Modified mask
        """
        for obj in draw_objects:
            polygon = obj["pos_imroi"]
            add_or_sub = obj.get("addorsub", "add")

            if len(polygon) < 3:
                continue

            poly_mask = RegionProcessor.fill_polygon(polygon, mask.shape)

            if add_or_sub == "add":
                mask = mask | poly_mask
            else:
                mask = mask & ~poly_mask

        return mask

    @staticmethod
    def form_mask_from_boundary(
        boundary: "Boundary",
        size: Tuple[int, int],
    ) -> NDArray[np.bool_]:
        """
        Create mask from boundary object.

        Args:
            boundary: Boundary object with add and sub polygons
            size: Output mask size (height, width)

        Returns:
            Binary mask
        """
        mask = np.zeros(size, dtype=np.bool_)

        # Add outer boundary
        if len(boundary.add) > 2 and not np.all(boundary.add == -1):
            mask = RegionProcessor.fill_polygon(boundary.add, size)

        # Subtract holes
        for sub in boundary.sub:
            if len(sub) > 2:
                sub_mask = RegionProcessor.fill_polygon(sub, size)
                mask = mask & ~sub_mask

        return mask

    @staticmethod
    def fill_polygon(
        vertices: NDArray[np.float64],
        size: Tuple[int, int],
    ) -> NDArray[np.bool_]:
        """
        Fill a polygon using scanline algorithm.

        Args:
            vertices: Nx2 array of (x, y) polygon vertices
            size: Output size (height, width)

        Returns:
            Filled polygon mask
        """
        if len(vertices) < 3:
            return np.zeros(size, dtype=np.bool_)

        h, w = size
        mask = np.zeros(size, dtype=np.bool_)

        # Get bounds
        min_y = max(0, int(np.floor(vertices[:, 1].min())))
        max_y = min(h - 1, int(np.ceil(vertices[:, 1].max())))

        n = len(vertices)

        for y in range(min_y, max_y + 1):
            # Find intersections with polygon edges
            intersections = []

            for i in range(n):
                j = (i + 1) % n
                y1, y2 = vertices[i, 1], vertices[j, 1]
                x1, x2 = vertices[i, 0], vertices[j, 0]

                if y1 == y2:
                    continue

                if min(y1, y2) <= y < max(y1, y2):
                    # Calculate x intersection
                    t = (y - y1) / (y2 - y1)
                    x_int = x1 + t * (x2 - x1)
                    intersections.append(x_int)

            # Sort and fill pairs
            intersections.sort()

            for i in range(0, len(intersections) - 1, 2):
                x_start = max(0, int(np.ceil(intersections[i])))
                x_end = min(w - 1, int(np.floor(intersections[i + 1])))
                if x_start <= x_end:
                    mask[y, x_start:x_end + 1] = True

        return mask

    @staticmethod
    def form_union(
        roi: "NcorrROI",
        mask: NDArray[np.bool_],
    ) -> List["Region"]:
        """
        Compute union of ROI regions with a mask.

        Args:
            roi: NcorrROI object
            mask: Binary mask to intersect with

        Returns:
            List of unioned Region objects
        """
        from ..core.roi import Region

        union_regions = []

        for region in roi.regions:
            if region.is_empty():
                union_regions.append(Region.empty_placeholder())
                continue

            # Create mask for this region
            region_mask = np.zeros(mask.shape, dtype=np.bool_)
            for j in range(region.noderange.shape[0]):
                x = j + region.leftbound
                if 0 <= x < mask.shape[1]:
                    for k in range(0, region.noderange[j], 2):
                        y_start = max(0, region.nodelist[j, k])
                        y_end = min(mask.shape[0] - 1, region.nodelist[j, k + 1])
                        if y_start <= y_end:
                            region_mask[y_start:y_end + 1, x] = True

            # Intersect with input mask
            intersected = region_mask & mask

            if not np.any(intersected):
                union_regions.append(Region.empty_placeholder())
                continue

            # Build new region from intersection
            # Find bounds
            rows, cols = np.where(intersected)
            left = cols.min()
            right = cols.max()
            top = rows.min()
            bottom = rows.max()
            total = len(rows)

            new_region = RegionProcessor._build_region_from_mask(
                intersected, left, right, top, bottom, total, False
            )
            union_regions.append(new_region)

        return union_regions

    @staticmethod
    def extrapolate_data(
        data: NDArray[np.float64],
        roi: "NcorrROI",
        border: int = 20,
    ) -> List[NDArray[np.float64]]:
        """
        Extrapolate data outside ROI boundaries for smoother interpolation.

        Equivalent to ncorr_alg_extrapdata.cpp

        Args:
            data: Data array to extrapolate
            roi: ROI defining valid data region
            border: Border size for extrapolation

        Returns:
            List of extrapolated data arrays (one per region)
        """
        from ..core.roi import Region

        result = []

        for region in roi.regions:
            if region.is_empty():
                result.append(np.array([]))
                continue

            # Calculate output size with border
            width = region.rightbound - region.leftbound + 1 + 2 * border
            height = region.lowerbound - region.upperbound + 1 + 2 * border

            extrapolated = np.zeros((height, width), dtype=np.float64)

            # Copy valid data
            for j in range(region.noderange.shape[0]):
                x_data = j + region.leftbound
                x_out = j + border

                for k in range(0, region.noderange[j], 2):
                    y_start = region.nodelist[j, k]
                    y_end = region.nodelist[j, k + 1]

                    for y_data in range(y_start, y_end + 1):
                        y_out = y_data - region.upperbound + border
                        if (0 <= y_out < height and
                            0 <= x_out < width and
                            0 <= y_data < data.shape[0] and
                            0 <= x_data < data.shape[1]):
                            extrapolated[y_out, x_out] = data[y_data, x_data]

            # Extrapolate by replicating boundary values
            # This is a simplified version - full implementation would use
            # gradient-based extrapolation

            # Fill with nearest neighbor extrapolation
            from scipy.ndimage import distance_transform_edt

            # Create mask of valid pixels
            valid = extrapolated != 0
            if np.any(valid):
                # Use distance transform for nearest neighbor
                _, indices = distance_transform_edt(~valid, return_indices=True)
                extrapolated = extrapolated[indices[0], indices[1]]

            result.append(extrapolated)

        return result

    @staticmethod
    def form_thread_diagram(
        roi: "NcorrROI",
        num_threads: int,
    ) -> List[NDArray[np.int32]]:
        """
        Create thread assignment diagram for parallel processing.

        Divides regions into approximately equal work units for threads.

        Args:
            roi: ROI to divide
            num_threads: Number of threads

        Returns:
            List of region point assignments per thread
        """
        total_points = sum(r.totalpoints for r in roi.regions if not r.is_empty())

        if total_points == 0:
            return [np.array([], dtype=np.int32) for _ in range(num_threads)]

        points_per_thread = total_points // num_threads
        thread_assignments = [[] for _ in range(num_threads)]

        current_thread = 0
        current_count = 0

        for region_idx, region in enumerate(roi.regions):
            if region.is_empty():
                continue

            for j in range(region.noderange.shape[0]):
                x = j + region.leftbound

                for k in range(0, region.noderange[j], 2):
                    y_start = region.nodelist[j, k]
                    y_end = region.nodelist[j, k + 1]

                    for y in range(y_start, y_end + 1):
                        thread_assignments[current_thread].append((region_idx, x, y))
                        current_count += 1

                        if (current_count >= points_per_thread and
                            current_thread < num_threads - 1):
                            current_thread += 1
                            current_count = 0

        return [np.array(ta, dtype=np.int32) for ta in thread_assignments]
