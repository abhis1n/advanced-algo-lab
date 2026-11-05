"""
Dijkstra's Algorithm
=====================
Standard SSSP implementation using a binary min-heap (heapq).

Time  : O((m + n) log n)
Space : O(n + m)
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from .graph import Graph


def dijkstra(
    graph: Graph,
    source: int,
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """
    Compute shortest-path distances from *source* to every reachable vertex.

    Parameters
    ----------
    graph : Graph
        Directed weighted graph with non-negative edge weights.
    source : int
        Source vertex.

    Returns
    -------
    dist : dict[int, float]
        Shortest-path distance from source to each vertex (math.inf if unreachable).
    prev : dict[int, int | None]
        Predecessor map for reconstructing shortest-path tree.
    """
    n = graph.n
    dist: Dict[int, float] = {v: math.inf for v in range(n)}
    prev: Dict[int, Optional[int]] = {v: None for v in range(n)}
    dist[source] = 0.0

    # Min-heap: (distance, vertex)
    heap: List[Tuple[float, int]] = [(0.0, source)]

    while heap:
        d_u, u = heapq.heappop(heap)

        # Skip stale entries
        if d_u > dist[u]:
            continue

        for v, w in graph.neighbors(u):
            new_dist = d_u + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, prev


def reconstruct_path(
    prev: Dict[int, Optional[int]],
    source: int,
    target: int,
) -> List[int]:
    """Reconstruct the shortest path from source to target using the prev map."""
    path: List[int] = []
    current: Optional[int] = target
    while current is not None:
        path.append(current)
        if current == source:
            break
        current = prev[current]
    path.reverse()
    if not path or path[0] != source:
        return []  # target unreachable
    return path
