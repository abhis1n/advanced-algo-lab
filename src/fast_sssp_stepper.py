"""
FastSSSP Stepper
=================
Generator-based FastSSSP algorithm that yields execution snapshots across its
conceptual stages (Pivot Selection, Bucket Partitioning, BMSSP bounded
relaxations, and Completion) to power step-by-step visualization.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .fast_sssp import _dijkstra_partial, _select_pivots
from .graph import Graph


def fast_sssp_steps(
    graph: Graph,
    source: int,
    seed: int = 0,
) -> Iterator[Dict[str, Any]]:
    """
    Yield step-by-step snapshots of FastSSSP from *source*.

    Each yielded dictionary contains:
      - step: int
      - phase: str
      - description: str
      - settled: set[int]
      - frontier: list[int]         (active FIFO queue vertices in BMSSP)
      - frontier_set: set[int]
      - current_vertex: int | None
      - relaxed_edges: list[tuple[int, int]]
      - pivots: set[int]
      - buckets: list[set[int]]
      - boundaries: list[float]
      - current_bucket_idx: int | None
      - dist: dict[int, float]
      - prev: dict[int, int | None]
      - operations: int
    """
    n = graph.n
    dist: Dict[int, float] = {v: math.inf for v in range(n)}
    prev: Dict[int, Optional[int]] = {v: None for v in range(n)}

    if n == 0:
        yield {
            "step": 0,
            "phase": "Complete",
            "description": "Empty graph.",
            "settled": set(),
            "frontier": [],
            "frontier_set": set(),
            "current_vertex": None,
            "relaxed_edges": [],
            "pivots": set(),
            "buckets": [],
            "boundaries": [],
            "current_bucket_idx": None,
            "dist": dist,
            "prev": prev,
            "operations": 0,
        }
        return

    dist[source] = 0.0
    rng = random.Random(seed)
    step = 0
    operations = 0

    def _make_snapshot(
        phase: str,
        description: str,
        settled: Set[int],
        frontier: List[int],
        current_vertex: Optional[int],
        relaxed_edges: List[Tuple[int, int]],
        pivots: Set[int],
        buckets: List[Set[int]],
        boundaries: List[float],
        current_bucket_idx: Optional[int],
    ) -> Dict[str, Any]:
        return {
            "step": step,
            "phase": phase,
            "description": description,
            "settled": set(settled),
            "frontier": list(frontier),
            "frontier_set": set(frontier),
            "current_vertex": current_vertex,
            "relaxed_edges": list(relaxed_edges),
            "pivots": set(pivots),
            "buckets": [set(b) for b in buckets],
            "boundaries": list(boundaries),
            "current_bucket_idx": current_bucket_idx,
            "dist": dict(dist),
            "prev": dict(prev),
            "operations": operations,
        }

    yield _make_snapshot(
        phase="Initialize",
        description=f"Initialized FastSSSP at source vertex {source} (dist=0).",
        settled={source},
        frontier=[],
        current_vertex=source,
        relaxed_edges=[],
        pivots=set(),
        buckets=[],
        boundaries=[],
        current_bucket_idx=None,
    )

    # ------------------------------------------------------------------
    # Phase 1: Pivot Selection & Distance Estimation
    # ------------------------------------------------------------------
    num_pivots = max(2, int(round(n ** (1.0 / 3.0))))
    num_pivots = min(num_pivots, n)
    pivots_list = _select_pivots(n, num_pivots, rng)
    pilot_dist = _dijkstra_partial(graph, source)

    pivot_dists = sorted((pilot_dist.get(p, math.inf), p) for p in pivots_list)
    pivot_dists = [(d, p) for d, p in pivot_dists if d < math.inf]

    if not pivot_dists:
        max_reachable = max(pilot_dist.values(), default=0.0)
        if max_reachable == 0.0 and len(pilot_dist) <= 1:
            step += 1
            yield _make_snapshot(
                phase="Complete",
                description=f"Source vertex {source} has no outgoing reachable vertices.",
                settled={source},
                frontier=[],
                current_vertex=None,
                relaxed_edges=[],
                pivots=set(pivots_list),
                buckets=[],
                boundaries=[],
                current_bucket_idx=None,
            )
            return
        pivot_dists = [(0.0, source)]

    pivot_set = {p for _, p in pivot_dists}
    step += 1
    yield _make_snapshot(
        phase="Phase 1: Pivot Selection",
        description=(
            f"Selected k={len(pivot_set)} reachable pivot(s): "
            f"{', '.join(f'{p} (d={d:.0f})' for d, p in pivot_dists)}."
        ),
        settled={source},
        frontier=[],
        current_vertex=None,
        relaxed_edges=[],
        pivots=pivot_set,
        buckets=[],
        boundaries=[],
        current_bucket_idx=None,
    )

    # ------------------------------------------------------------------
    # Phase 2: Bucket Partitioning
    # ------------------------------------------------------------------
    boundaries: List[float] = [0.0]
    for i in range(len(pivot_dists) - 1):
        mid = (pivot_dists[i][0] + pivot_dists[i + 1][0]) / 2.0
        boundaries.append(mid)
    max_dist = max(
        (pilot_dist.get(v, 0.0) for v in range(n) if pilot_dist.get(v, math.inf) < math.inf),
        default=0.0,
    )
    boundaries.append(max_dist + 1.0)

    num_buckets = len(boundaries) - 1
    buckets: List[Set[int]] = [set() for _ in range(num_buckets)]
    for v in range(n):
        d = pilot_dist.get(v, math.inf)
        if d == math.inf:
            continue
        placed = False
        for b in range(num_buckets):
            if boundaries[b] <= d < boundaries[b + 1]:
                buckets[b].add(v)
                placed = True
                break
        if not placed:
            buckets[-1].add(v)

    step += 1
    yield _make_snapshot(
        phase="Phase 2: Bucket Partitioning",
        description=(
            f"Partitioned reachable vertices into {num_buckets} distance bucket(s) "
            f"with boundaries [{', '.join(f'{b:.0f}' for b in boundaries)}]."
        ),
        settled={source},
        frontier=[],
        current_vertex=None,
        relaxed_edges=[],
        pivots=pivot_set,
        buckets=buckets,
        boundaries=boundaries,
        current_bucket_idx=None,
    )

    # ------------------------------------------------------------------
    # Phase 3: BMSSP within each bucket (FIFO bounded relaxation)
    # ------------------------------------------------------------------
    settled: Set[int] = set()
    frontier_sources: Dict[int, float] = {source: 0.0}

    for b_idx in range(num_buckets):
        if not frontier_sources:
            break

        bucket_lo = boundaries[b_idx]
        bucket_hi = boundaries[b_idx + 1]

        # Initialize BMSSP FIFO queue for this bucket
        bmssp_dist: Dict[int, float] = {}
        for v, d in frontier_sources.items():
            if d <= bucket_hi:
                bmssp_dist[v] = d
                if d < dist[v]:
                    dist[v] = d

        queue: deque[int] = deque(bmssp_dist.keys())
        in_queue: Set[int] = set(queue)

        step += 1
        yield _make_snapshot(
            phase=f"Phase 3: BMSSP (Bucket {b_idx + 1}/{num_buckets})",
            description=(
                f"Starting BMSSP in bucket {b_idx + 1} "
                f"[{bucket_lo:.0f}, {bucket_hi:.0f}) with {len(queue)} frontier source(s)."
            ),
            settled=settled,
            frontier=list(queue),
            current_vertex=None,
            relaxed_edges=[],
            pivots=pivot_set,
            buckets=buckets,
            boundaries=boundaries,
            current_bucket_idx=b_idx,
        )

        while queue:
            u = queue.popleft()
            in_queue.discard(u)
            d_u = bmssp_dist[u]
            operations += 1
            relaxed_edges: List[Tuple[int, int]] = []

            for v, w in graph.neighbors(u):
                operations += 1
                new_dist = d_u + w
                if new_dist > bucket_hi:
                    continue
                if v not in bmssp_dist or new_dist < bmssp_dist[v]:
                    bmssp_dist[v] = new_dist
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev[v] = u
                    if v not in in_queue:
                        queue.append(v)
                        in_queue.add(v)
                    relaxed_edges.append((u, v))

            if relaxed_edges or len(queue) == 0:
                step += 1
                relaxed_str = (
                    ", ".join(f"{a}->{b} ({dist[b]:.0f})" for a, b in relaxed_edges)
                    if relaxed_edges
                    else "none"
                )
                yield _make_snapshot(
                    phase=f"Phase 3: BMSSP (Bucket {b_idx + 1}/{num_buckets})",
                    description=(
                        f"Popped vertex {u} (dist={d_u:.0f}) from FIFO queue "
                        f"[bound <= {bucket_hi:.0f}]. Relaxed: {relaxed_str}."
                    ),
                    settled=settled,
                    frontier=list(queue),
                    current_vertex=u,
                    relaxed_edges=relaxed_edges,
                    pivots=pivot_set,
                    buckets=buckets,
                    boundaries=boundaries,
                    current_bucket_idx=b_idx,
                )

        # Merge BMSSP distances and mark bucket vertices settled
        for v, d in bmssp_dist.items():
            if d < dist[v]:
                dist[v] = d
            if d < bucket_hi:
                settled.add(v)

        frontier_sources = {v: d for v, d in bmssp_dist.items() if d < math.inf}

    # Finalize predecessor map for consistency with fast_sssp
    prev_final: Dict[int, Optional[int]] = {v: None for v in range(n)}
    for u in range(n):
        if dist[u] == math.inf:
            continue
        for v, w in graph.neighbors(u):
            candidate = dist[u] + w
            if abs(candidate - dist[v]) < 1e-9:
                if prev_final[v] is None or dist[u] < dist.get(prev_final[v], math.inf):
                    prev_final[v] = u
    prev = prev_final
    settled = {v for v in range(n) if dist[v] < math.inf}

    step += 1
    yield _make_snapshot(
        phase="Complete",
        description=(
            f"All {len(settled)}/{n} reachable vertices settled via BMSSP "
            f"in {operations} queue/relax operations."
        ),
        settled=settled,
        frontier=[],
        current_vertex=None,
        relaxed_edges=[],
        pivots=pivot_set,
        buckets=buckets,
        boundaries=boundaries,
        current_bucket_idx=None,
    )
