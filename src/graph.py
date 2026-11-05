"""
Graph Generation Module
========================
Generates directed weighted graphs for SSSP algorithm benchmarking.

Supported graph families:
  - Sparse directed (m ≈ 2n)
  - Medium-density directed (m ≈ 5n)
  - Dense directed (m ≈ n^1.5)
  - Random directed / Erdős–Rényi (variable p)
  - Grid / lattice graphs (k × k)

All edge weights are non-negative integers drawn uniformly from [1, 1000].
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Type alias: adjacency list mapping vertex -> list of (neighbor, weight)
AdjList = Dict[int, List[Tuple[int, float]]]


@dataclass
class Graph:
    """Directed weighted graph stored as an adjacency list."""

    n: int  # number of vertices (labelled 0 … n-1)
    adj: AdjList = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def m(self) -> int:
        """Return the number of edges."""
        return sum(len(neighbors) for neighbors in self.adj.values())

    def add_edge(self, u: int, v: int, w: float) -> None:
        self.adj.setdefault(u, []).append((v, w))

    def vertices(self) -> range:
        return range(self.n)

    def neighbors(self, u: int) -> List[Tuple[int, float]]:
        return self.adj.get(u, [])

    def __repr__(self) -> str:
        return f"Graph(n={self.n}, m={self.m})"


# ====================================================================== #
#  Graph generators
# ====================================================================== #

def _rand_weight(rng: random.Random, lo: int = 1, hi: int = 1000) -> int:
    """Return a random edge weight in [lo, hi]."""
    return rng.randint(lo, hi)


def generate_sparse(n: int, seed: int = 42) -> Graph:
    """Sparse directed graph with m ≈ 2n edges."""
    return _generate_random_edges(n, target_m=2 * n, seed=seed)


def generate_medium(n: int, seed: int = 42) -> Graph:
    """Medium-density directed graph with m ≈ 5n edges."""
    return _generate_random_edges(n, target_m=5 * n, seed=seed)


def generate_dense(n: int, seed: int = 42) -> Graph:
    """Dense directed graph with m ≈ n^1.5 edges."""
    target_m = int(n ** 1.5)
    return _generate_random_edges(n, target_m=target_m, seed=seed)


def generate_erdos_renyi(n: int, p: float = 0.05, seed: int = 42) -> Graph:
    """Erdős–Rényi random directed graph G(n, p)."""
    rng = random.Random(seed)
    g = Graph(n=n)
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                g.add_edge(u, v, _rand_weight(rng))
    return g


def generate_grid(k: int, seed: int = 42) -> Graph:
    """
    k × k grid / lattice graph (directed, 4-connected).
    Vertex (r, c) is labelled r*k + c.
    Edges go right and down (and left and up) — all four directions.
    """
    rng = random.Random(seed)
    n = k * k
    g = Graph(n=n)

    def idx(r: int, c: int) -> int:
        return r * k + c

    for r in range(k):
        for c in range(k):
            u = idx(r, c)
            # right
            if c + 1 < k:
                g.add_edge(u, idx(r, c + 1), _rand_weight(rng))
            # left
            if c - 1 >= 0:
                g.add_edge(u, idx(r, c - 1), _rand_weight(rng))
            # down
            if r + 1 < k:
                g.add_edge(u, idx(r + 1, c), _rand_weight(rng))
            # up
            if r - 1 >= 0:
                g.add_edge(u, idx(r - 1, c), _rand_weight(rng))
    return g


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _generate_random_edges(n: int, target_m: int, seed: int) -> Graph:
    """
    Generate a directed graph with *approximately* target_m random edges.
    Ensures the graph is weakly connected by first building a random spanning
    arborescence (tree), then adding remaining edges randomly.
    """
    rng = random.Random(seed)
    g = Graph(n=n)
    max_possible = n * (n - 1)
    target_m = min(target_m, max_possible)

    # Phase 1: random spanning tree to guarantee connectivity
    vertices = list(range(n))
    rng.shuffle(vertices)
    edge_set: set[Tuple[int, int]] = set()
    for i in range(1, n):
        u = vertices[rng.randint(0, i - 1)]
        v = vertices[i]
        g.add_edge(u, v, _rand_weight(rng))
        edge_set.add((u, v))

    # Phase 2: add remaining random edges
    remaining = target_m - (n - 1)
    attempts = 0
    max_attempts = remaining * 10
    while remaining > 0 and attempts < max_attempts:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (u, v) not in edge_set:
            g.add_edge(u, v, _rand_weight(rng))
            edge_set.add((u, v))
            remaining -= 1
        attempts += 1

    return g


# ====================================================================== #
#  Convenience: get all graph configurations for benchmarking
# ====================================================================== #

GRAPH_CONFIGS = [
    # (label, generator_func, kwargs_list)
    ("Sparse (m~2n)", generate_sparse, [
        {"n": 500}, {"n": 1000}, {"n": 5000}, {"n": 10000},
    ]),
    ("Medium (m~5n)", generate_medium, [
        {"n": 500}, {"n": 1000}, {"n": 5000}, {"n": 10000},
    ]),
    ("Dense (m~n^1.5)", generate_dense, [
        {"n": 500}, {"n": 1000}, {"n": 5000},
    ]),
    ("Erdos-Renyi (p=0.05)", generate_erdos_renyi, [
        {"n": 500, "p": 0.05}, {"n": 1000, "p": 0.05}, {"n": 5000, "p": 0.01},
    ]),
    ("Grid", generate_grid, [
        {"k": 20}, {"k": 50},
    ]),
]


def iter_benchmark_graphs(seeds: range = range(42, 47)):
    """
    Yield (label, graph) tuples for benchmarking.
    Generates `len(seeds)` instances per configuration.
    """
    for label, gen_fn, kwargs_list in GRAPH_CONFIGS:
        for kwargs in kwargs_list:
            for seed in seeds:
                g = gen_fn(**kwargs, seed=seed)
                size_tag = f"n={kwargs.get('n', kwargs.get('k', '?'))}"
                yield f"{label} {size_tag}", g
