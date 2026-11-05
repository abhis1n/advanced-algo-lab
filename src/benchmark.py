"""
Benchmarking Framework
=======================
Measures execution time, memory usage, and correctness for SSSP algorithms.

Protocol (from synopsis):
  - 10 runs per configuration, report mean ± std
  - 5 independent graph instances per config
  - Graph generation time excluded from algorithm timing
  - Correctness verified by comparing distances from both algorithms
"""

from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .graph import Graph


@dataclass
class BenchmarkResult:
    """Results from a single algorithm run."""

    algorithm: str
    graph_label: str
    n: int
    m: int
    time_seconds: float
    peak_memory_bytes: int
    num_reachable: int  # vertices with finite distance


@dataclass
class ComparisonResult:
    """Results from comparing two algorithms on the same graph."""

    graph_label: str
    n: int
    m: int
    dijkstra_result: BenchmarkResult
    fast_sssp_result: BenchmarkResult
    distances_match: bool
    max_distance_error: float
    speedup: float  # dijkstra_time / fast_sssp_time


@dataclass
class AggregateResult:
    """Aggregated results across multiple runs."""

    graph_label: str
    n: int
    m: int
    algorithm: str
    mean_time: float
    std_time: float
    mean_memory: float
    std_memory: float
    num_runs: int
    all_correct: bool


# Type alias for an SSSP algorithm function
SSSPFunc = Callable[[Graph, int], Tuple[Dict[int, float], Dict[int, Optional[int]]]]


def time_algorithm(
    algorithm: SSSPFunc,
    graph: Graph,
    source: int,
    algo_name: str = "unknown",
    graph_label: str = "unknown",
) -> Tuple[BenchmarkResult, Dict[int, float]]:
    """
    Run an SSSP algorithm once, measuring wall-clock time and peak memory.

    Returns the BenchmarkResult and the distance dictionary.
    """
    # Start memory tracking
    tracemalloc.start()

    # Time the algorithm
    t_start = time.perf_counter()
    dist, prev = algorithm(graph, source)
    t_end = time.perf_counter()

    # Capture peak memory
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = t_end - t_start
    num_reachable = sum(1 for d in dist.values() if d < math.inf)

    result = BenchmarkResult(
        algorithm=algo_name,
        graph_label=graph_label,
        n=graph.n,
        m=graph.m,
        time_seconds=elapsed,
        peak_memory_bytes=peak_mem,
        num_reachable=num_reachable,
    )

    return result, dist


def compare_distances(
    dist_a: Dict[int, float],
    dist_b: Dict[int, float],
    tolerance: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Compare two distance dictionaries.

    Returns (match, max_error).
    """
    all_keys = set(dist_a.keys()) | set(dist_b.keys())
    max_error = 0.0

    for k in all_keys:
        da = dist_a.get(k, math.inf)
        db = dist_b.get(k, math.inf)
        if da == math.inf and db == math.inf:
            continue
        if da == math.inf or db == math.inf:
            return False, math.inf
        err = abs(da - db)
        max_error = max(max_error, err)

    return max_error <= tolerance, max_error


def run_comparison(
    graph: Graph,
    source: int,
    dijkstra_func: SSSPFunc,
    fast_sssp_func: SSSPFunc,
    graph_label: str = "unknown",
) -> ComparisonResult:
    """
    Run both algorithms on the same graph and compare results.
    """
    dij_result, dij_dist = time_algorithm(
        dijkstra_func, graph, source,
        algo_name="Dijkstra", graph_label=graph_label,
    )
    fast_result, fast_dist = time_algorithm(
        fast_sssp_func, graph, source,
        algo_name="FastSSSP", graph_label=graph_label,
    )

    match, max_err = compare_distances(dij_dist, fast_dist)

    speedup = dij_result.time_seconds / fast_result.time_seconds if fast_result.time_seconds > 0 else math.inf

    return ComparisonResult(
        graph_label=graph_label,
        n=graph.n,
        m=graph.m,
        dijkstra_result=dij_result,
        fast_sssp_result=fast_result,
        distances_match=match,
        max_distance_error=max_err,
        speedup=speedup,
    )


def run_benchmark_suite(
    graphs: List[Tuple[str, Graph]],
    dijkstra_func: SSSPFunc,
    fast_sssp_func: SSSPFunc,
    source: int = 0,
    num_runs: int = 3,
    verbose: bool = True,
) -> List[ComparisonResult]:
    """
    Run the full benchmark suite across all graph configurations.

    Parameters
    ----------
    graphs : list of (label, Graph)
    dijkstra_func : Dijkstra SSSP function
    fast_sssp_func : FastSSSP function
    source : source vertex
    num_runs : repetitions per graph (reduced from 10 for demo speed)
    verbose : print progress

    Returns
    -------
    list of ComparisonResult
    """
    results: List[ComparisonResult] = []

    for i, (label, graph) in enumerate(graphs):
        if verbose:
            print(f"\n[{i+1}/{len(graphs)}] {label}  (n={graph.n}, m={graph.m})")

        for run in range(num_runs):
            comp = run_comparison(
                graph, source, dijkstra_func, fast_sssp_func,
                graph_label=label,
            )
            results.append(comp)

            if verbose:
                status = "PASS" if comp.distances_match else "FAIL MISMATCH"
                print(
                    f"  Run {run+1}/{num_runs}: "
                    f"Dijkstra={comp.dijkstra_result.time_seconds:.4f}s  "
                    f"FastSSSP={comp.fast_sssp_result.time_seconds:.4f}s  "
                    f"Speedup={comp.speedup:.2f}x  "
                    f"{status}"
                )

    return results


def aggregate_results(
    results: List[ComparisonResult],
) -> Dict[str, Dict[str, AggregateResult]]:
    """
    Aggregate ComparisonResults by graph label and algorithm.

    Returns a nested dict: {graph_label: {algorithm: AggregateResult}}.
    """
    import statistics

    # Group by graph label
    groups: Dict[str, List[ComparisonResult]] = {}
    for r in results:
        groups.setdefault(r.graph_label, []).append(r)

    aggregated: Dict[str, Dict[str, AggregateResult]] = {}

    for label, comps in groups.items():
        dij_times = [c.dijkstra_result.time_seconds for c in comps]
        dij_mems = [c.dijkstra_result.peak_memory_bytes for c in comps]
        fast_times = [c.fast_sssp_result.time_seconds for c in comps]
        fast_mems = [c.fast_sssp_result.peak_memory_bytes for c in comps]

        n = comps[0].n
        m = comps[0].m
        all_correct = all(c.distances_match for c in comps)

        aggregated[label] = {
            "Dijkstra": AggregateResult(
                graph_label=label, n=n, m=m, algorithm="Dijkstra",
                mean_time=statistics.mean(dij_times),
                std_time=statistics.stdev(dij_times) if len(dij_times) > 1 else 0.0,
                mean_memory=statistics.mean(dij_mems),
                std_memory=statistics.stdev(dij_mems) if len(dij_mems) > 1 else 0.0,
                num_runs=len(comps),
                all_correct=True,  # Dijkstra is the reference
            ),
            "FastSSSP": AggregateResult(
                graph_label=label, n=n, m=m, algorithm="FastSSSP",
                mean_time=statistics.mean(fast_times),
                std_time=statistics.stdev(fast_times) if len(fast_times) > 1 else 0.0,
                mean_memory=statistics.mean(fast_mems),
                std_memory=statistics.stdev(fast_mems) if len(fast_mems) > 1 else 0.0,
                num_runs=len(comps),
                all_correct=all_correct,
            ),
        }

    return aggregated


def print_aggregate_table(agg: Dict[str, Dict[str, AggregateResult]]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Graph':<30} {'Algorithm':<12} {'Mean Time (s)':<16} {'Std (s)':<12} "
          f"{'Memory (KB)':<14} {'Correct':<8}")
    print("=" * 100)

    for label, algos in agg.items():
        for algo_name in ["Dijkstra", "FastSSSP"]:
            a = algos[algo_name]
            print(
                f"{label:<30} {algo_name:<12} {a.mean_time:<16.6f} {a.std_time:<12.6f} "
                f"{a.mean_memory / 1024:<14.1f} {'PASS' if a.all_correct else 'FAIL':<8}"
            )
        print("-" * 100)
