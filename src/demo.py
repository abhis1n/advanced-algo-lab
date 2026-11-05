"""
Interactive Demo
=================
CLI-based interactive demonstration for the SSSP algorithm comparison.

Features:
  - Choose graph type, size, and density
  - Select source vertex
  - Run both Dijkstra and FastSSSP
  - Display distance comparison & correctness check
  - Show timing & memory results
  - Visualize shortest-path tree (small graphs)
  - Run full benchmark suite with plots
"""

from __future__ import annotations

import math
import os
import sys
import time

from .graph import (
    Graph,
    generate_sparse,
    generate_medium,
    generate_dense,
    generate_erdos_renyi,
    generate_grid,
)
from .dijkstra import dijkstra, reconstruct_path as dijkstra_path
from .fast_sssp import fast_sssp, reconstruct_path as fast_path
from .benchmark import (
    time_algorithm,
    compare_distances,
    run_comparison,
    run_benchmark_suite,
    aggregate_results,
    print_aggregate_table,
)
from .visualize import (
    generate_all_plots,
    plot_shortest_path_tree,
)


# ====================================================================== #
#  Helpers
# ====================================================================== #

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _header(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _prompt_int(msg: str, lo: int, hi: int) -> int:
    while True:
        try:
            val = int(input(f"  {msg} [{lo}-{hi}]: "))
            if lo <= val <= hi:
                return val
            print(f"    Please enter a value between {lo} and {hi}.")
        except (ValueError, EOFError):
            print(f"    Invalid input. Please enter an integer.")


def _prompt_choice(msg: str, options: list[str]) -> int:
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    return _prompt_int(msg, 1, len(options)) - 1


# ====================================================================== #
#  Menu options
# ====================================================================== #

def _generate_graph() -> tuple[Graph, str]:
    """Interactive graph generation."""
    _header("Graph Generation")

    graph_types = [
        "Sparse Directed (m ~ 2n)",
        "Medium-Density Directed (m ~ 5n)",
        "Dense Directed (m ~ n^1.5)",
        "Erdos-Renyi Random (variable p)",
        "Grid / Lattice (k x k)",
    ]
    print("\n  Available graph types:")
    idx = _prompt_choice("Select graph type", graph_types)

    if idx == 4:  # Grid
        k = _prompt_int("Grid dimension k (k*k vertices)", 3, 200)
        seed = _prompt_int("Random seed", 0, 99999)
        g = generate_grid(k, seed=seed)
        label = f"Grid {k}x{k}"
    else:
        n = _prompt_int("Number of vertices n", 10, 100000)
        seed = _prompt_int("Random seed", 0, 99999)
        if idx == 0:
            g = generate_sparse(n, seed=seed)
            label = f"Sparse n={n}"
        elif idx == 1:
            g = generate_medium(n, seed=seed)
            label = f"Medium n={n}"
        elif idx == 2:
            g = generate_dense(n, seed=seed)
            label = f"Dense n={n}"
        else:  # Erdos-Renyi
            p_choices = ["0.01 (very sparse)", "0.05 (moderate)", "0.1 (fairly dense)", "0.2 (dense)"]
            p_idx = _prompt_choice("Select edge probability p", p_choices)
            p = [0.01, 0.05, 0.1, 0.2][p_idx]
            g = generate_erdos_renyi(n, p=p, seed=seed)
            label = f"Erdos-Renyi n={n} p={p}"

    print(f"\n  [OK] Generated: {label}")
    print(f"    Vertices: {g.n:,}")
    print(f"    Edges:    {g.m:,}")
    if g.n > 1:
        print(f"    Density:  {g.m / (g.n * (g.n - 1)) * 100:.4f}%")

    return g, label


def _run_single_comparison(graph: Graph, label: str):
    """Run both algorithms on a single graph and display results."""
    _header(f"Running SSSP on: {label}")

    source = _prompt_int(f"Source vertex", 0, graph.n - 1)

    print(f"\n  Running Dijkstra from vertex {source}...")
    dij_result, dij_dist = time_algorithm(
        dijkstra, graph, source, "Dijkstra", label,
    )
    dij_prev = dijkstra(graph, source)[1]
    print(f"    Time:   {dij_result.time_seconds:.6f} s")
    print(f"    Memory: {dij_result.peak_memory_bytes / 1024:.1f} KB")
    print(f"    Reachable vertices: {dij_result.num_reachable:,}")

    print(f"\n  Running FastSSSP from vertex {source}...")
    fast_result, fast_dist = time_algorithm(
        fast_sssp, graph, source, "FastSSSP", label,
    )
    fast_prev = fast_sssp(graph, source)[1]
    print(f"    Time:   {fast_result.time_seconds:.6f} s")
    print(f"    Memory: {fast_result.peak_memory_bytes / 1024:.1f} KB")
    print(f"    Reachable vertices: {fast_result.num_reachable:,}")

    # Correctness check
    match, max_err = compare_distances(dij_dist, fast_dist)
    print(f"\n  Correctness Check:")
    if match:
        print(f"    [PASS] Distances MATCH (max error: {max_err:.2e})")
    else:
        print(f"    [FAIL] Distances MISMATCH (max error: {max_err:.2e})")
        # Show first few mismatches
        mismatch_count = 0
        for v in sorted(dij_dist.keys()):
            dd = dij_dist.get(v, math.inf)
            fd = fast_dist.get(v, math.inf)
            if abs(dd - fd) > 1e-6 and not (dd == math.inf and fd == math.inf):
                print(f"      vertex {v}: Dijkstra={dd:.2f}  FastSSSP={fd:.2f}")
                mismatch_count += 1
                if mismatch_count >= 5:
                    print(f"      ... and more")
                    break

    # Speedup
    speedup = dij_result.time_seconds / fast_result.time_seconds if fast_result.time_seconds > 0 else float("inf")
    print(f"\n  Speedup (Dijkstra / FastSSSP): {speedup:.3f}x")
    if speedup > 1:
        print(f"    -> FastSSSP was {speedup:.2f}x faster")
    else:
        print(f"    -> Dijkstra was {1/speedup:.2f}x faster")

    # Distance summary
    print(f"\n  Distance Summary (first 10 vertices):")
    print(f"    {'Vertex':<10} {'Dijkstra':<15} {'FastSSSP':<15} {'Match':<6}")
    print(f"    {'-'*46}")
    for v in range(min(10, graph.n)):
        dd = dij_dist.get(v, math.inf)
        fd = fast_dist.get(v, math.inf)
        m = "OK" if abs(dd - fd) < 1e-6 or (dd == math.inf and fd == math.inf) else "X"
        dd_s = f"{dd:.1f}" if dd < math.inf else "INF"
        fd_s = f"{fd:.1f}" if fd < math.inf else "INF"
        print(f"    {v:<10} {dd_s:<15} {fd_s:<15} {m:<6}")

    # Offer visualization for small graphs
    if graph.n <= 100:
        print(f"\n  Graph is small enough for visualization ({graph.n} vertices).")
        try:
            viz_input = input("  Generate shortest-path tree plot? [y/N]: ").strip().lower()
        except EOFError:
            viz_input = "n"
        if viz_input == "y":
            target = _prompt_int("Target vertex (for path highlight)", 0, graph.n - 1)
            os.makedirs("plots", exist_ok=True)
            plot_shortest_path_tree(
                graph, dij_dist, dij_prev, source, target,
                output_path="plots/shortest_path_tree.png",
                title=f"Shortest-Path Tree: {label} (source={source})",
            )
            print("    [OK] Plot saved to plots/shortest_path_tree.png")


def _run_quick_benchmark():
    """Run a quick benchmark across several graph sizes."""
    _header("Quick Benchmark Suite")

    print("\n  This will benchmark both algorithms across multiple graph configs.")
    print("  Using reduced settings for speed (3 runs per config).\n")

    # Build graph set
    graphs = []
    configs = [
        ("Sparse n=500", generate_sparse, {"n": 500}),
        ("Sparse n=1000", generate_sparse, {"n": 1000}),
        ("Sparse n=5000", generate_sparse, {"n": 5000}),
        ("Medium n=500", generate_medium, {"n": 500}),
        ("Medium n=1000", generate_medium, {"n": 1000}),
        ("Medium n=5000", generate_medium, {"n": 5000}),
        ("Dense n=500", generate_dense, {"n": 500}),
        ("Dense n=1000", generate_dense, {"n": 1000}),
        ("Grid 20x20", generate_grid, {"k": 20}),
        ("Grid 50x50", generate_grid, {"k": 50}),
    ]

    print("  Generating graphs...")
    for label, gen_fn, kwargs in configs:
        g = gen_fn(**kwargs, seed=42)
        graphs.append((label, g))
        print(f"    {label}: n={g.n:,}, m={g.m:,}")

    print()

    results = run_benchmark_suite(
        graphs, dijkstra, fast_sssp,
        source=0, num_runs=3, verbose=True,
    )

    agg = aggregate_results(results)
    print_aggregate_table(agg)

    # Generate plots
    os.makedirs("plots", exist_ok=True)
    generate_all_plots(agg, output_dir="plots")
    print("\n  [OK] All plots saved to plots/ directory")


def _run_full_benchmark():
    """Run the full benchmark suite from the synopsis."""
    _header("Full Benchmark Suite (Synopsis Protocol)")

    print("\n  Running full benchmark protocol:")
    print("    * 5 graph families x multiple sizes")
    print("    * 3 runs per config (reduced for demo)")
    print("    * Correctness verification on every run")
    print()

    from .graph import GRAPH_CONFIGS

    graphs = []
    for family_label, gen_fn, kwargs_list in GRAPH_CONFIGS:
        for kwargs in kwargs_list:
            g = gen_fn(**kwargs, seed=42)
            size_tag = f"n={kwargs.get('n', kwargs.get('k', '?'))}"
            label = f"{family_label} {size_tag}"
            graphs.append((label, g))
            print(f"    {label}: n={g.n:,}, m={g.m:,}")

    print()

    results = run_benchmark_suite(
        graphs, dijkstra, fast_sssp,
        source=0, num_runs=3, verbose=True,
    )

    agg = aggregate_results(results)
    print_aggregate_table(agg)

    os.makedirs("plots", exist_ok=True)
    generate_all_plots(agg, output_dir="plots")
    print("\n  [OK] All plots saved to plots/ directory")


# ====================================================================== #
#  Main menu
# ====================================================================== #

def main():
    """Main entry point for the interactive demo."""
    print("""
    +==============================================================+
    |    SSSP Algorithm Comparison -- Interactive Demo              |
    |    Dijkstra vs FastSSSP (Duan et al., STOC 2025)             |
    +==============================================================+
    """)

    while True:
        _header("Main Menu")
        options = [
            "Single Graph Comparison (interactive)",
            "Quick Benchmark (10 configs, 3 runs each)",
            "Full Benchmark Suite (all synopsis configs)",
            "Exit",
        ]
        choice = _prompt_choice("Select option", options)

        if choice == 0:
            graph, label = _generate_graph()
            _run_single_comparison(graph, label)
        elif choice == 1:
            _run_quick_benchmark()
        elif choice == 2:
            _run_full_benchmark()
        elif choice == 3:
            print("\n  Goodbye!")
            break

        print("\n  Press Enter to continue...")
        try:
            input()
        except EOFError:
            break


if __name__ == "__main__":
    main()
