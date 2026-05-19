"""
kdtree.py — Custom 2D KD-Tree for Spatial Property Indexing
============================================================
Implemented from scratch (no third-party spatial libraries).
Axes: axis-0 = latitude, axis-1 = longitude
Distance metric: Haversine (great-circle distance in km)

Schema aligned with the King County House Sales dataset:
  price, bedrooms, bathrooms, sqft_living, sqft_lot, floors,
  yr_built, lat, long
"""

import math
import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class Property:
    """Represents a single real-estate property from the King County House Sales dataset."""
    id: str
    price: float
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    yr_built: float
    lat: float
    long: float


# ---------------------------------------------------------------------------
# KD-Tree Node
# ---------------------------------------------------------------------------

class KDNode:
    """A node in the KD-Tree holding one Property and child pointers."""

    __slots__ = ("property", "axis", "left", "right")

    def __init__(self, property: Property, axis: int):
        self.property: Property = property
        self.axis: int = axis          # 0 → split on lat, 1 → split on long
        self.left:  Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


# ---------------------------------------------------------------------------
# KD-Tree
# ---------------------------------------------------------------------------

class KDTree:
    """
    A 2-dimensional KD-Tree keyed on (lat, long).

    Public API
    ----------
    build(properties)                              → builds a balanced tree O(n log n)
    insert(property)                               → insert one node O(log n) average
    search_within_radius(lat, lon, radius_km)      → all points ≤ radius  O(√n + k)
    k_nearest_neighbors(lat, lon, k)               → k closest points     O(k log n)
    """

    # Earth's mean radius in kilometres (WGS-84 approximation)
    _EARTH_RADIUS_KM: float = 6371.0

    def __init__(self) -> None:
        self.root: Optional[KDNode] = None
        self.size: int = 0

    # ------------------------------------------------------------------
    # Haversine distance (pure Python, no external deps)
    # ------------------------------------------------------------------

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Returns the great-circle distance (km) between two geographic points.

        Formula:
            a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
            d = 2R · atan2(√a, √(1−a))
        """
        R = KDTree._EARTH_RADIUS_KM
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi    = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (math.sin(dphi / 2) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)

        return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    # ------------------------------------------------------------------
    # Coordinate accessor
    # ------------------------------------------------------------------

    @staticmethod
    def _coord(prop: Property, axis: int) -> float:
        """Return the relevant coordinate for the given splitting axis."""
        return prop.lat if axis == 0 else prop.long

    # ------------------------------------------------------------------
    # Build — O(n log n)
    # ------------------------------------------------------------------

    def build(self, properties: List[Property]) -> None:
        """
        Build a balanced KD-Tree from a list of Property objects.
        Uses median-of-sorted-list strategy for balance.
        Any previous tree is discarded.
        """
        self.root = self._build_recursive(list(properties), depth=0)
        self.size = len(properties)

    def _build_recursive(
        self, props: List[Property], depth: int
    ) -> Optional[KDNode]:
        if not props:
            return None

        axis = depth % 2
        # Sort by current axis and pick the median as pivot
        props.sort(key=lambda p: self._coord(p, axis))
        median = len(props) // 2

        node = KDNode(props[median], axis)
        node.left  = self._build_recursive(props[:median],       depth + 1)
        node.right = self._build_recursive(props[median + 1:],   depth + 1)
        return node

    # ------------------------------------------------------------------
    # Insert — O(log n) average (tree may become unbalanced over time)
    # ------------------------------------------------------------------

    def insert(self, prop: Property) -> None:
        """Insert a single property. Use `build()` for bulk loading."""
        self.root = self._insert_recursive(self.root, prop, depth=0)
        self.size += 1

    def _insert_recursive(
        self, node: Optional[KDNode], prop: Property, depth: int
    ) -> KDNode:
        if node is None:
            return KDNode(prop, depth % 2)

        axis = node.axis
        if self._coord(prop, axis) < self._coord(node.property, axis):
            node.left  = self._insert_recursive(node.left,  prop, depth + 1)
        else:
            node.right = self._insert_recursive(node.right, prop, depth + 1)
        return node

    # ------------------------------------------------------------------
    # Radius Search — O(√n + k) for uniform distributions
    # ------------------------------------------------------------------

    def search_within_radius(
        self, lat: float, lon: float, radius_km: float
    ) -> List[Property]:
        """
        Return all properties whose Haversine distance from (lat, lon)
        is ≤ radius_km.

        Pruning strategy
        ----------------
        At each node, compute the *minimum possible* Haversine distance
        from the query point to the splitting hyperplane (converted to km).
        If that distance already exceeds the radius, the entire subtree on
        the far side is pruned.
        """
        results: List[Property] = []
        self._radius_search(self.root, lat, lon, radius_km, results)
        return results

    def _radius_search(
        self,
        node: Optional[KDNode],
        lat: float,
        lon: float,
        radius_km: float,
        results: List[Property],
    ) -> None:
        if node is None:
            return

        # Check current node
        dist = self.haversine(lat, lon, node.property.lat, node.property.long)
        if dist <= radius_km:
            results.append(node.property)

        # Compute signed difference along splitting axis
        axis = node.axis
        if axis == 0:
            diff    = lat - node.property.lat
            # 1° latitude ≈ 111.32 km (constant everywhere)
            diff_km = abs(diff) * 111.32
        else:
            diff    = lon - node.property.long
            # 1° longitude ≈ 111.32 · cos(lat) km
            diff_km = abs(diff) * 111.32 * math.cos(math.radians(lat))

        # Visit the "near" subtree first, then conditionally the "far" subtree
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)

        self._radius_search(near, lat, lon, radius_km, results)

        # Prune the far subtree if the splitting plane is farther than radius
        if diff_km <= radius_km:
            self._radius_search(far, lat, lon, radius_km, results)

    # ------------------------------------------------------------------
    # k-Nearest Neighbors — O(k log k · log n) average
    # ------------------------------------------------------------------

    def k_nearest_neighbors(
        self, lat: float, lon: float, k: int
    ) -> List[Tuple[float, Property]]:
        """
        Return the k nearest properties as a list of (distance_km, Property)
        tuples, sorted ascending by distance.

        Uses a max-heap of size k to efficiently prune the search space.
        """
        if k <= 0:
            return []

        # Max-heap: store (−distance, tie-breaker id, property)
        # Python's heapq is a min-heap, so we negate distance for max-heap behaviour.
        heap: List[Tuple[float, int, Property]] = []

        def _search(node: Optional[KDNode]) -> None:
            if node is None:
                return

            dist = self.haversine(lat, lon, node.property.lat, node.property.long)

            if len(heap) < k:
                heapq.heappush(heap, (-dist, id(node), node.property))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, id(node), node.property))

            # Splitting-plane distance (same calculation as radius search)
            axis = node.axis
            if axis == 0:
                diff    = lat - node.property.lat
                diff_km = abs(diff) * 111.32
            else:
                diff    = lon - node.property.long
                diff_km = abs(diff) * 111.32 * math.cos(math.radians(lat))

            near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)

            _search(near)

            # Only explore the far side if it could contain a closer neighbour
            worst_in_heap = -heap[0][0] if heap else float("inf")
            if len(heap) < k or diff_km < worst_in_heap:
                _search(far)

        _search(self.root)

        # Sort ascending by distance and strip the heap bookkeeping fields
        return sorted(
            [(-neg_dist, prop) for neg_dist, _, prop in heap],
            key=lambda t: t[0],
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def height(self) -> int:
        """Return the height of the tree (useful for debugging balance)."""
        def _height(node: Optional[KDNode]) -> int:
            if node is None:
                return 0
            return 1 + max(_height(node.left), _height(node.right))
        return _height(self.root)

    def __repr__(self) -> str:
        return f"KDTree(size={self.size}, height={self.height()})"
