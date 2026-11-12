"""
Dijkstra Stepper
=================
Generator-based Dijkstra's algorithm that yields execution snapshots after
each major step (initialization, vertex extraction, and completion) to power
step-by-step visualization.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .graph import Graph


def dijkstra_steps(
    graph: Graph,
    source: int,
) -> Iterator[Dict[str, Any]]:
    """
    Yield step-by-step snapshots of Dijkstra's algorithm from *source*.

    Each yielded dictionary contains:
      - step: int
      - phase: str ("Initialize" | "Extract-Min" | "Complete")
      - description: str
      - settled: set[int]
      - frontier: list[tuple[float, int]]  (active heap entries sorted by dist)
      - frontier_set: set[int]
      - current_vertex: int | None
      - relaxed_edges: list[tuple[int, int]]
      - dist: dict[int, float]
      - prev: dict[int, int | None]
      - operations: int
    """
    n = graph.n
    dist: Dict[int, float] = {v: math.inf for v in range(n)}
    prev: Dict[int, Optional[int]] = {v: None for v in range(n)}
    dist[source] = 0.0

    settled: Set[int] = set()
    heap: List[Tuple[float, int]] = [(0.0, source)]
    in_frontier: Set[int] = {source}
    step = 0
    operations = 0

    def _active_frontier() -> List[Tuple[float, int]]:
        return sorted((dist[v], v) for v in in_frontier if v not in settled)

    yield {
        "step": step,
        "phase": "Initialize",
        "description": f"Initialized source vertex {source} with distance 0; pushed to min-heap.",
        "settled": set(settled),
        "frontier": _active_frontier(),
        "frontier_set": set(in_frontier),
        "current_vertex": source,
        "relaxed_edges": [],
        "dist": dict(dist),
        "prev": dict(prev),
        "operations": operations,
    }

    while heap:
        d_u, u = heapq.heappop(heap)
        operations += 1

        # Skip stale heap entries
        if d_u > dist[u] or u in settled:
            continue

        settled.add(u)
        in_frontier.discard(u)
        relaxed_edges: List[Tuple[int, int]] = []

        for v, w in graph.neighbors(u):
            operations += 1
            new_dist = d_u + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))
                operations += 1
                if v not in settled:
                    in_frontier.add(v)
                relaxed_edges.append((u, v))

        step += 1
        relaxed_str = (
            ", ".join(f"{a}->{b} ({dist[b]:.0f})" for a, b in relaxed_edges)
            if relaxed_edges
            else "none"
        )
        yield {
            "step": step,
            "phase": "Extract-Min",
            "description": (
                f"Extracted vertex {u} (dist={d_u:.0f}). "
                f"Relaxed edges: {relaxed_str}."
            ),
            "settled": set(settled),
            "frontier": _active_frontier(),
            "frontier_set": set(in_frontier),
            "current_vertex": u,
            "relaxed_edges": relaxed_edges,
            "dist": dict(dist),
            "prev": dict(prev),
            "operations": operations,
        }

    step += 1
    reachable = sum(1 for d in dist.values() if d < math.inf)
    yield {
        "step": step,
        "phase": "Complete",
        "description": (
            f"Priority queue empty. Settled {reachable}/{n} reachable vertices "
            f"in {operations} heap/relax operations."
        ),
        "settled": set(settled),
        "frontier": [],
        "frontier_set": set(),
        "current_vertex": None,
        "relaxed_edges": [],
        "dist": dict(dist),
        "prev": dict(prev),
        "operations": operations,
    }
