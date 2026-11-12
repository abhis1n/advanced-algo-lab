"""
Visualization Module
=====================
Generates charts and graph drawings for the SSSP comparison project.

Charts produced:
  1. Runtime comparison bar chart (Dijkstra vs FastSSSP)
  2. Scalability line plot (runtime vs n)
  3. Memory usage comparison
  4. Speedup bar chart
  5. Shortest-path tree visualization (small graphs)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx

from .graph import Graph
from .benchmark import AggregateResult, ComparisonResult


# ====================================================================== #
#  Styling
# ====================================================================== #

DIJKSTRA_COLOR = "#2196F3"  # blue
FASTSSP_COLOR = "#FF9800"   # orange
CORRECT_COLOR = "#4CAF50"   # green
ERROR_COLOR = "#F44336"     # red

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


# ====================================================================== #
#  1. Runtime Comparison Bar Chart
# ====================================================================== #

def plot_runtime_comparison(
    agg: Dict[str, Dict[str, AggregateResult]],
    output_path: str = "plots/runtime_comparison.png",
) -> str:
    """Bar chart comparing mean runtime of Dijkstra vs FastSSSP."""
    _ensure_dir(output_path)

    labels = list(agg.keys())
    dij_times = [agg[l]["Dijkstra"].mean_time for l in labels]
    fast_times = [agg[l]["FastSSSP"].mean_time for l in labels]
    dij_stds = [agg[l]["Dijkstra"].std_time for l in labels]
    fast_stds = [agg[l]["FastSSSP"].std_time for l in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    bars1 = ax.bar([i - width/2 for i in x], dij_times, width,
                   yerr=dij_stds, label="Dijkstra", color=DIJKSTRA_COLOR,
                   capsize=3, alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], fast_times, width,
                   yerr=fast_stds, label="FastSSSP", color=FASTSSP_COLOR,
                   capsize=3, alpha=0.85)

    ax.set_xlabel("Graph Configuration")
    ax.set_ylabel("Mean Runtime (seconds)")
    ax.set_title("Runtime Comparison: Dijkstra vs FastSSSP")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


# ====================================================================== #
#  2. Scalability Line Plot
# ====================================================================== #

def plot_scalability(
    agg: Dict[str, Dict[str, AggregateResult]],
    output_path: str = "plots/scalability.png",
) -> str:
    """Line plot: runtime vs graph size (n) for each algorithm."""
    _ensure_dir(output_path)

    # Extract (n, time) pairs grouped by algorithm
    dij_points: List[Tuple[int, float]] = []
    fast_points: List[Tuple[int, float]] = []

    for label, algos in agg.items():
        n = algos["Dijkstra"].n
        dij_points.append((n, algos["Dijkstra"].mean_time))
        fast_points.append((n, algos["FastSSSP"].mean_time))

    # Sort by n
    dij_points.sort()
    fast_points.sort()

    fig, ax = plt.subplots()
    ax.plot([p[0] for p in dij_points], [p[1] for p in dij_points],
            "o-", color=DIJKSTRA_COLOR, label="Dijkstra", markersize=6)
    ax.plot([p[0] for p in fast_points], [p[1] for p in fast_points],
            "s-", color=FASTSSP_COLOR, label="FastSSSP", markersize=6)

    ax.set_xlabel("Number of Vertices (n)")
    ax.set_ylabel("Mean Runtime (seconds)")
    ax.set_title("Scalability: Runtime vs Graph Size")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


# ====================================================================== #
#  3. Memory Usage Comparison
# ====================================================================== #

def plot_memory_comparison(
    agg: Dict[str, Dict[str, AggregateResult]],
    output_path: str = "plots/memory_comparison.png",
) -> str:
    """Bar chart comparing peak memory usage."""
    _ensure_dir(output_path)

    labels = list(agg.keys())
    dij_mem = [agg[l]["Dijkstra"].mean_memory / 1024 for l in labels]  # KB
    fast_mem = [agg[l]["FastSSSP"].mean_memory / 1024 for l in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    ax.bar([i - width/2 for i in x], dij_mem, width,
           label="Dijkstra", color=DIJKSTRA_COLOR, alpha=0.85)
    ax.bar([i + width/2 for i in x], fast_mem, width,
           label="FastSSSP", color=FASTSSP_COLOR, alpha=0.85)

    ax.set_xlabel("Graph Configuration")
    ax.set_ylabel("Peak Memory (KB)")
    ax.set_title("Memory Usage Comparison: Dijkstra vs FastSSSP")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


# ====================================================================== #
#  4. Speedup Bar Chart
# ====================================================================== #

def plot_speedup(
    agg: Dict[str, Dict[str, AggregateResult]],
    output_path: str = "plots/speedup.png",
) -> str:
    """Bar chart showing speedup (Dijkstra_time / FastSSSP_time)."""
    _ensure_dir(output_path)

    labels = list(agg.keys())
    speedups = []
    for l in labels:
        dt = agg[l]["Dijkstra"].mean_time
        ft = agg[l]["FastSSSP"].mean_time
        speedups.append(dt / ft if ft > 0 else 0)

    colors = [CORRECT_COLOR if s >= 1.0 else ERROR_COLOR for s in speedups]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    ax.bar(range(len(labels)), speedups, color=colors, alpha=0.85)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Graph Configuration")
    ax.set_ylabel("Speedup (Dijkstra / FastSSSP)")
    ax.set_title("Speedup: Values > 1 mean FastSSSP is faster")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


# ====================================================================== #
#  5. Shortest-Path Tree Visualization
# ====================================================================== #

def plot_shortest_path_tree(
    graph: Graph,
    dist: Dict[int, float],
    prev: Dict[int, Optional[int]],
    source: int,
    target: Optional[int] = None,
    output_path: str = "plots/shortest_path_tree.png",
    title: str = "Shortest-Path Tree",
) -> str:
    """
    Draw the graph with the shortest-path tree highlighted.
    Only practical for small graphs (n ≤ ~100).
    """
    _ensure_dir(output_path)

    G = nx.DiGraph()
    for u in range(graph.n):
        G.add_node(u)
        for v, w in graph.neighbors(u):
            G.add_edge(u, v, weight=w)

    # Build tree edges from prev
    tree_edges = set()
    for v, u in prev.items():
        if u is not None:
            tree_edges.add((u, v))

    # Build shortest path to target if specified
    path_edges = set()
    if target is not None and prev.get(target) is not None:
        curr = target
        while curr is not None and curr != source:
            p = prev.get(curr)
            if p is not None:
                path_edges.add((p, curr))
            curr = p

    pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, graph.n ** 0.5))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw non-tree edges (light gray)
    non_tree = [(u, v) for u, v in G.edges() if (u, v) not in tree_edges]
    nx.draw_networkx_edges(G, pos, edgelist=non_tree, ax=ax,
                           edge_color="#CCCCCC", alpha=0.3, arrows=True,
                           arrowsize=8, width=0.5)

    # Draw tree edges (blue)
    tree_list = [(u, v) for u, v in G.edges() if (u, v) in tree_edges and (u, v) not in path_edges]
    nx.draw_networkx_edges(G, pos, edgelist=tree_list, ax=ax,
                           edge_color=DIJKSTRA_COLOR, alpha=0.7, arrows=True,
                           arrowsize=12, width=1.5)

    # Draw path edges (orange, bold)
    if path_edges:
        path_list = list(path_edges)
        nx.draw_networkx_edges(G, pos, edgelist=path_list, ax=ax,
                               edge_color=FASTSSP_COLOR, alpha=1.0, arrows=True,
                               arrowsize=15, width=3.0)

    # Node colors
    node_colors = []
    for v in G.nodes():
        if v == source:
            node_colors.append(CORRECT_COLOR)
        elif target is not None and v == target:
            node_colors.append(FASTSSP_COLOR)
        elif dist.get(v, math.inf) < math.inf:
            node_colors.append(DIJKSTRA_COLOR)
        else:
            node_colors.append("#CCCCCC")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=300, alpha=0.9, edgecolors="white", linewidths=1.5)

    # Labels: show vertex id and distance
    labels = {}
    for v in G.nodes():
        d = dist.get(v, math.inf)
        if d < math.inf:
            labels[v] = f"{v}\n({d:.0f})"
        else:
            labels[v] = f"{v}\n(∞)"
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


# ====================================================================== #
#  6. Theoretical Scaling Analysis
# ====================================================================== #

def plot_theoretical_scaling(
    output_dir: str = "plots",
) -> List[str]:
    """
    Generate theoretical scaling plots that project operation counts
    at graph sizes far beyond what fits in RAM, demonstrating the
    asymptotic regime where FastSSSP outperforms Dijkstra.

    Produces three plots:
      - theoretical_ops_sparse.png  (m = 2n, sparse / road-network-like)
      - theoretical_ops_medium.png  (m = 5n, moderate density)
      - theoretical_ops_dense.png   (m = n^1.5, dense)
    """
    import numpy as np

    _ensure_dir(f"{output_dir}/x")
    paths: List[str] = []

    # Range of n values: from 10^3 to 10^15
    n_values = np.logspace(3, 15, num=500)
    log2_n = np.log2(n_values)

    density_configs = [
        ("Sparse (m = 2n)",  lambda n: 2 * n,       "sparse"),
        ("Medium (m = 5n)",  lambda n: 5 * n,       "medium"),
        ("Dense (m = n^1.5)", lambda n: n ** 1.5,    "dense"),
    ]

    for density_label, m_fn, tag in density_configs:
        m_values = np.array([m_fn(n) for n in n_values])

        # Operation counts (proportional to asymptotic bound)
        dijkstra_binary = (m_values + n_values) * log2_n        # O((m+n) log n)
        dijkstra_fibo   = m_values + n_values * log2_n          # O(m + n log n)
        fast_sssp_ops   = m_values * (log2_n ** (2.0 / 3.0))   # O(m log^{2/3} n)

        # ----- Plot 1: Absolute operation counts -----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.loglog(n_values, dijkstra_binary, "-",
                   color="#1565C0", linewidth=2, label="Dijkstra (binary heap)")
        ax1.loglog(n_values, dijkstra_fibo, "--",
                   color=DIJKSTRA_COLOR, linewidth=2, label="Dijkstra (Fibonacci heap)")
        ax1.loglog(n_values, fast_sssp_ops, "-",
                   color=FASTSSP_COLOR, linewidth=2.5, label="FastSSSP")

        ax1.set_xlabel("Number of Vertices (n)", fontsize=12)
        ax1.set_ylabel("Theoretical Operation Count", fontsize=12)
        ax1.set_title(f"Theoretical Scaling: {density_label}", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, which="both", alpha=0.3)

        # Mark crossover region
        ratio = dijkstra_fibo / fast_sssp_ops
        crossover_indices = np.where(ratio > 1.0)[0]
        if len(crossover_indices) > 0:
            cross_n = n_values[crossover_indices[0]]
            ax1.axvline(x=cross_n, color=CORRECT_COLOR, linestyle=":", linewidth=1.5, alpha=0.7)
            ax1.text(cross_n * 1.5, ax1.get_ylim()[0] * 10,
                     f"Crossover\nn ~ {cross_n:.0e}", fontsize=9,
                     color=CORRECT_COLOR, fontweight="bold")

        # ----- Plot 2: Speedup ratio (Dijkstra / FastSSSP) -----
        speedup_binary = dijkstra_binary / fast_sssp_ops
        speedup_fibo = dijkstra_fibo / fast_sssp_ops

        ax2.semilogx(n_values, speedup_binary, "-",
                     color="#1565C0", linewidth=2, label="vs Dijkstra (binary heap)")
        ax2.semilogx(n_values, speedup_fibo, "--",
                     color=DIJKSTRA_COLOR, linewidth=2, label="vs Dijkstra (Fibonacci heap)")
        ax2.axhline(y=1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax2.fill_between(n_values, 1.0, speedup_fibo,
                         where=speedup_fibo > 1.0,
                         alpha=0.15, color=FASTSSP_COLOR,
                         label="FastSSSP advantage")

        ax2.set_xlabel("Number of Vertices (n)", fontsize=12)
        ax2.set_ylabel("Speedup Ratio (Dijkstra ops / FastSSSP ops)", fontsize=12)
        ax2.set_title(f"Theoretical Speedup of FastSSSP: {density_label}",
                      fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10, loc="upper left")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.set_ylim(bottom=0)

        fig.tight_layout()
        out_path = f"{output_dir}/theoretical_ops_{tag}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")
        paths.append(out_path)

    # ----- Summary comparison table plot -----
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_title("Theoretical Operation Count Comparison (Sparse Graphs, m = 2n)",
                 fontsize=14, fontweight="bold", pad=20)

    col_labels = ["n (vertices)", "Dijkstra\n(binary heap)\n(m+n) log n",
                  "Dijkstra\n(Fibonacci)\nm + n log n",
                  "FastSSSP\nm log^(2/3) n",
                  "Speedup\n(Fibo / FastSSSP)"]

    table_n_values = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12, 1e15]
    table_data = []
    for n in table_n_values:
        m = 2 * n
        lg = math.log2(n)
        dij_bin = (m + n) * lg
        dij_fib = m + n * lg
        fast = m * (lg ** (2.0 / 3.0))
        speedup = dij_fib / fast
        table_data.append([
            f"{n:.0e}",
            f"{dij_bin:.2e}",
            f"{dij_fib:.2e}",
            f"{fast:.2e}",
            f"{speedup:.2f}x",
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#D9E2F3")
        cell.set_text_props(fontweight="bold", fontsize=9)

    # Highlight rows where FastSSSP wins
    for i, row in enumerate(table_data):
        speedup_val = float(row[-1].replace("x", ""))
        if speedup_val > 1.5:
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor("#E8F5E9")  # light green
        # Highlight the speedup cell
        if speedup_val > 2.0:
            table[i + 1, len(col_labels) - 1].set_text_props(
                fontweight="bold", color="#2E7D32")

    fig.tight_layout()
    out_path = f"{output_dir}/theoretical_comparison_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    paths.append(out_path)

    return paths


# ====================================================================== #
#  Generate all plots from benchmark results
# ====================================================================== #

def generate_all_plots(
    agg: Dict[str, Dict[str, AggregateResult]],
    output_dir: str = "plots",
) -> List[str]:
    """Generate all standard charts and return list of output paths."""
    paths = []
    print("\nGenerating plots...")
    paths.append(plot_runtime_comparison(agg, f"{output_dir}/runtime_comparison.png"))
    paths.append(plot_scalability(agg, f"{output_dir}/scalability.png"))
    paths.append(plot_memory_comparison(agg, f"{output_dir}/memory_comparison.png"))
    paths.append(plot_speedup(agg, f"{output_dir}/speedup.png"))
    print("\nGenerating theoretical scaling analysis...")
    paths.extend(plot_theoretical_scaling(output_dir))
    return paths
