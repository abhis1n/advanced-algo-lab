"""
Correctness Tests
==================
Verify that both Dijkstra and FastSSSP produce identical shortest-path
distances on a variety of graph configurations.
"""

from __future__ import annotations

import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import (
    Graph,
    generate_sparse,
    generate_medium,
    generate_dense,
    generate_erdos_renyi,
    generate_grid,
)
from src.dijkstra import dijkstra, reconstruct_path as dij_path
from src.fast_sssp import fast_sssp, reconstruct_path as fast_path
from src.benchmark import compare_distances


def _assert_distances_match(
    graph: Graph,
    source: int,
    label: str,
    tolerance: float = 1e-6,
):
    """Run both algorithms and assert distances match."""
    dij_dist, dij_prev = dijkstra(graph, source)
    fast_dist, fast_prev = fast_sssp(graph, source, seed=0)

    match, max_err = compare_distances(dij_dist, fast_dist, tolerance)

    if not match:
        # Print detailed mismatch info
        mismatches = []
        for v in sorted(dij_dist.keys()):
            dd = dij_dist.get(v, math.inf)
            fd = fast_dist.get(v, math.inf)
            if abs(dd - fd) > tolerance and not (dd == math.inf and fd == math.inf):
                mismatches.append((v, dd, fd))

        print(f"  FAIL: {label} — {len(mismatches)} mismatches (max_err={max_err:.2e})")
        for v, dd, fd in mismatches[:5]:
            print(f"    vertex {v}: Dijkstra={dd:.2f}, FastSSSP={fd:.2f}")
        return False
    else:
        print(f"  PASS: {label} (max_err={max_err:.2e}, "
              f"reachable={sum(1 for d in dij_dist.values() if d < math.inf)})")
        return True


def test_small_sparse():
    """Test on small sparse graphs."""
    for seed in range(5):
        g = generate_sparse(50, seed=seed)
        yield _assert_distances_match(g, 0, f"Sparse n=50 seed={seed}")


def test_small_medium():
    """Test on small medium-density graphs."""
    for seed in range(5):
        g = generate_medium(50, seed=seed)
        yield _assert_distances_match(g, 0, f"Medium n=50 seed={seed}")


def test_small_dense():
    """Test on small dense graphs."""
    for seed in range(3):
        g = generate_dense(30, seed=seed)
        yield _assert_distances_match(g, 0, f"Dense n=30 seed={seed}")


def test_erdos_renyi():
    """Test on Erdos-Renyi random graphs."""
    for seed in range(3):
        g = generate_erdos_renyi(40, p=0.1, seed=seed)
        yield _assert_distances_match(g, 0, f"ER n=40 p=0.1 seed={seed}")


def test_grid():
    """Test on grid graphs."""
    for k in [5, 10]:
        g = generate_grid(k, seed=42)
        yield _assert_distances_match(g, 0, f"Grid {k}×{k}")


def test_medium_graphs():
    """Test on larger graphs to check scaling."""
    for n in [200, 500]:
        g = generate_sparse(n, seed=42)
        yield _assert_distances_match(g, 0, f"Sparse n={n}")
    for n in [200, 500]:
        g = generate_medium(n, seed=42)
        yield _assert_distances_match(g, 0, f"Medium n={n}")


def test_different_sources():
    """Test with different source vertices."""
    g = generate_medium(100, seed=42)
    for source in [0, 10, 50, 99]:
        yield _assert_distances_match(g, source, f"Medium n=100 source={source}")


def test_single_vertex():
    """Edge case: single-vertex graph."""
    g = Graph(n=1)
    dij_dist, _ = dijkstra(g, 0)
    fast_dist, _ = fast_sssp(g, 0)
    match = dij_dist[0] == 0 and fast_dist[0] == 0
    print(f"  {'PASS' if match else 'FAIL'}: Single vertex")
    yield match


def test_disconnected():
    """Test with a partially disconnected graph."""
    g = Graph(n=5)
    g.add_edge(0, 1, 10)
    g.add_edge(1, 2, 20)
    # vertices 3 and 4 are unreachable
    yield _assert_distances_match(g, 0, "Disconnected n=5")


def main():
    """Run all tests."""
    print("=" * 60)
    print("  SSSP Algorithm Correctness Tests")
    print("=" * 60)

    all_tests = [
        ("Small Sparse Graphs", test_small_sparse),
        ("Small Medium-Density Graphs", test_small_medium),
        ("Small Dense Graphs", test_small_dense),
        ("Erdos-Renyi Random Graphs", test_erdos_renyi),
        ("Grid / Lattice Graphs", test_grid),
        ("Medium-Size Graphs", test_medium_graphs),
        ("Different Source Vertices", test_different_sources),
        ("Single Vertex (edge case)", test_single_vertex),
        ("Disconnected Graph (edge case)", test_disconnected),
    ]

    total_pass = 0
    total_fail = 0

    for section_name, test_fn in all_tests:
        print(f"\n{section_name}:")
        for result in test_fn():
            if result:
                total_pass += 1
            else:
                total_fail += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {total_pass} passed, {total_fail} failed, "
          f"{total_pass + total_fail} total")
    print(f"{'=' * 60}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
