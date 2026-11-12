"""
UI Helpers for Streamlit Frontend
==================================
Shared session state management and Matplotlib/NetworkX graph rendering
functions for the multi-page Streamlit visualization app.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import streamlit as st

from .graph import (
    Graph,
    generate_sparse,
    generate_medium,
    generate_dense,
    generate_erdos_renyi,
    generate_grid,
)

# Color palette
COLOR_DIJKSTRA = "#2196F3"   # Blue
COLOR_FASTSSSP = "#FF9800"   # Orange
COLOR_SOURCE = "#4CAF50"     # Green
COLOR_FRONTIER = "#FFC107"   # Amber / Yellow
COLOR_UNVISITED = "#CBD5E1"  # Cool Gray
COLOR_CURRENT = "#E91E63"    # Pink / Magenta
COLOR_PIVOT = "#D32F2F"      # Red

BUCKET_COLORS = [
    "#BBDEFB",  # Light Blue
    "#C8E6C9",  # Light Green
    "#FFE0B2",  # Light Orange
    "#E1BEE7",  # Light Purple
    "#B2DFDB",  # Teal
    "#F8BBD0",  # Pink
    "#FFF9C4",  # Pale Yellow
    "#D7CCC8",  # Warm Gray
]


def graph_to_nx(graph: Graph) -> nx.DiGraph:
    """Convert internal Graph to a NetworkX DiGraph."""
    G = nx.DiGraph()
    for u in range(graph.n):
        G.add_node(u)
        for v, w in graph.neighbors(u):
            G.add_edge(u, v, weight=w)
    return G


def compute_layout(
    nx_graph: nx.DiGraph,
    graph_type: str = "Sparse",
    grid_k: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """Compute stable node positions for rendering."""
    n = nx_graph.number_of_nodes()
    if n == 0:
        return {}
    if graph_type == "Grid" and grid_k is not None and grid_k * grid_k == n:
        return {r * grid_k + c: (float(c), float(-r)) for r in range(grid_k) for c in range(grid_k)}
    return nx.spring_layout(nx_graph, seed=42, k=2.2 / max(1.0, math.sqrt(n)), iterations=80)


def set_active_graph(
    graph: Graph,
    label: str,
    graph_type: str = "Sparse",
    grid_k: Optional[int] = None,
) -> None:
    """Store a newly generated graph and its cached layout in session_state."""
    nx_g = graph_to_nx(graph)
    pos = compute_layout(nx_g, graph_type=graph_type, grid_k=grid_k)

    st.session_state["graph"] = graph
    st.session_state["graph_label"] = label
    st.session_state["graph_type"] = graph_type
    st.session_state["grid_k"] = grid_k
    st.session_state["nx_graph"] = nx_g
    st.session_state["pos"] = pos
    # Clear cached race & animation steps when graph changes
    st.session_state.pop("race_results", None)
    st.session_state.pop("anim_cache", None)
    st.session_state["anim_step"] = 0
    st.session_state["anim_playing"] = False


def init_session_state() -> None:
    """Ensure a default graph exists in session_state so any page works immediately."""
    if "graph" not in st.session_state:
        default_g = generate_medium(20, seed=42)
        set_active_graph(default_g, "Medium (m~5n) n=20", graph_type="Medium")


# ====================================================================== #
#  Page 1: Graph Builder Figure
# ====================================================================== #

def draw_graph_preview(
    graph: Graph,
    nx_graph: nx.DiGraph,
    pos: Dict[int, Tuple[float, float]],
    title: str,
) -> plt.Figure:
    """Draw graph with edge-weight color gradient."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=130)
    n = graph.n
    edges = list(nx_graph.edges(data=True))

    if edges:
        weights = [d["weight"] for _, _, d in edges]
        vmin, vmax = min(weights), max(weights)
        if vmin == vmax:
            vmax = vmin + 1
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis

        edge_colors = [cmap(norm(w)) for w in weights]
        node_size = 420 if n <= 35 else (200 if n <= 120 else 60)

        nx.draw_networkx_edges(
            nx_graph,
            pos,
            ax=ax,
            edge_color=edge_colors,
            alpha=0.75 if n <= 100 else 0.45,
            arrows=n <= 150,
            arrowsize=12 if n <= 40 else 8,
            width=1.4 if n <= 60 else 0.7,
            connectionstyle="arc3,rad=0.06",
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Edge Weight", fontsize=9)
    else:
        node_size = 420

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        ax=ax,
        node_color=COLOR_DIJKSTRA,
        node_size=node_size,
        edgecolors="white",
        linewidths=1.2,
    )

    if n <= 60:
        nx.draw_networkx_labels(
            nx_graph,
            pos,
            ax=ax,
            font_size=8 if n <= 30 else 6,
            font_color="white",
            font_weight="bold",
        )

    if n <= 25 and len(edges) <= 80:
        edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(
            nx_graph,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=6.5,
            label_pos=0.35,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ====================================================================== #
#  Page 2: Shortest-Path Tree Figure
# ====================================================================== #

def draw_spt_figure(
    nx_graph: nx.DiGraph,
    pos: Dict[int, Tuple[float, float]],
    dist: Dict[int, float],
    prev: Dict[int, Optional[int]],
    source: int,
    target: Optional[int],
    title: str,
    tree_color: str = COLOR_DIJKSTRA,
) -> plt.Figure:
    """Draw shortest-path tree and optional source->target highlighted path."""
    n = nx_graph.number_of_nodes()
    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=130)

    tree_edges = {(u, v) for v, u in prev.items() if u is not None}

    path_edges: Set[Tuple[int, int]] = set()
    path_nodes: Set[int] = set()
    if target is not None and dist.get(target, math.inf) < math.inf:
        curr: Optional[int] = target
        visited: Set[int] = set()
        while curr is not None and curr != source and curr not in visited:
            visited.add(curr)
            path_nodes.add(curr)
            p = prev.get(curr)
            if p is not None:
                path_edges.add((p, curr))
            curr = p
        path_nodes.add(source)

    non_tree = [(u, v) for u, v in nx_graph.edges() if (u, v) not in tree_edges]
    tree_only = [(u, v) for u, v in tree_edges if (u, v) not in path_edges]

    nx.draw_networkx_edges(
        nx_graph, pos, edgelist=non_tree, ax=ax,
        edge_color="#CBD5E1", alpha=0.28, arrows=n <= 120,
        arrowsize=8, width=0.6,
    )
    if tree_only:
        nx.draw_networkx_edges(
            nx_graph, pos, edgelist=tree_only, ax=ax,
            edge_color=tree_color, alpha=0.8, arrows=n <= 120,
            arrowsize=11, width=1.6,
        )
    if path_edges:
        nx.draw_networkx_edges(
            nx_graph, pos, edgelist=list(path_edges), ax=ax,
            edge_color="#EF4444", alpha=1.0, arrows=n <= 120,
            arrowsize=14, width=3.0,
        )

    node_colors = []
    for v in nx_graph.nodes():
        if v == source:
            node_colors.append(COLOR_SOURCE)
        elif target is not None and v == target:
            node_colors.append("#EF4444")
        elif v in path_nodes:
            node_colors.append("#F59E0B")
        elif dist.get(v, math.inf) < math.inf:
            node_colors.append(tree_color)
        else:
            node_colors.append(COLOR_UNVISITED)

    node_size = 380 if n <= 35 else (180 if n <= 120 else 55)
    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax, node_color=node_colors,
        node_size=node_size, edgecolors="white", linewidths=1.2,
    )

    if n <= 45:
        labels = {
            v: f"{v}\n({dist[v]:.0f})" if dist.get(v, math.inf) < math.inf else f"{v}\n(INF)"
            for v in nx_graph.nodes()
        }
        nx.draw_networkx_labels(
            nx_graph, pos, labels=labels, ax=ax,
            font_size=6.5 if n <= 25 else 5.5, font_weight="bold",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ====================================================================== #
#  Page 3: Step-by-Step Animation Frames
# ====================================================================== #

def draw_dijkstra_step(
    nx_graph: nx.DiGraph,
    pos: Dict[int, Tuple[float, float]],
    snapshot: Dict[str, Any],
    source: int,
) -> plt.Figure:
    """Render a single step snapshot of Dijkstra's algorithm."""
    n = nx_graph.number_of_nodes()
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=130)

    settled: Set[int] = snapshot["settled"]
    frontier_set: Set[int] = snapshot["frontier_set"]
    current_v: Optional[int] = snapshot["current_vertex"]
    relaxed_edges: Set[Tuple[int, int]] = set(snapshot["relaxed_edges"])
    dist: Dict[int, float] = snapshot["dist"]
    prev: Dict[int, Optional[int]] = snapshot["prev"]

    tree_edges = {(u, v) for v, u in prev.items() if u is not None}
    other_edges = [e for e in nx_graph.edges() if e not in tree_edges and e not in relaxed_edges]
    tree_only = [e for e in tree_edges if e not in relaxed_edges]

    nx.draw_networkx_edges(
        nx_graph, pos, edgelist=other_edges, ax=ax,
        edge_color="#CBD5E1", alpha=0.35, arrows=True, arrowsize=9, width=0.8,
        connectionstyle="arc3,rad=0.05",
    )
    if tree_only:
        nx.draw_networkx_edges(
            nx_graph, pos, edgelist=tree_only, ax=ax,
            edge_color=COLOR_DIJKSTRA, alpha=0.8, arrows=True, arrowsize=12, width=1.8,
            connectionstyle="arc3,rad=0.05",
        )
    if relaxed_edges:
        nx.draw_networkx_edges(
            nx_graph, pos, edgelist=list(relaxed_edges), ax=ax,
            edge_color=COLOR_FASTSSSP, alpha=1.0, arrows=True, arrowsize=15, width=3.0,
            style="dashed", connectionstyle="arc3,rad=0.05",
        )

    node_colors = []
    edge_rings = []
    ring_widths = []
    for v in nx_graph.nodes():
        if v == source:
            node_colors.append(COLOR_SOURCE)
        elif v in settled:
            node_colors.append(COLOR_DIJKSTRA)
        elif v in frontier_set:
            node_colors.append(COLOR_FRONTIER)
        else:
            node_colors.append(COLOR_UNVISITED)

        if v == current_v:
            edge_rings.append(COLOR_CURRENT)
            ring_widths.append(3.2)
        else:
            edge_rings.append("white")
            ring_widths.append(1.2)

    node_size = 520 if n <= 30 else 280
    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax, node_color=node_colors,
        node_size=node_size, edgecolors=edge_rings, linewidths=ring_widths,
    )

    labels = {
        v: f"{v}\n({dist[v]:.0f})" if dist.get(v, math.inf) < math.inf else f"{v}\n(INF)"
        for v in nx_graph.nodes()
    }
    nx.draw_networkx_labels(
        nx_graph, pos, labels=labels, ax=ax,
        font_size=7.5 if n <= 30 else 6.0, font_weight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=COLOR_SOURCE, label="Source"),
        mpatches.Patch(color=COLOR_DIJKSTRA, label="Settled"),
        mpatches.Patch(color=COLOR_FRONTIER, label="In Priority Queue"),
        mpatches.Patch(color=COLOR_UNVISITED, label="Unvisited"),
        mpatches.Patch(facecolor="none", edgecolor=COLOR_CURRENT, linewidth=2, label="Current Vertex"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Dijkstra — Step {snapshot['step']} ({snapshot['phase']})",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_fast_sssp_step(
    nx_graph: nx.DiGraph,
    pos: Dict[int, Tuple[float, float]],
    snapshot: Dict[str, Any],
    source: int,
) -> plt.Figure:
    """Render a single step snapshot of FastSSSP."""
    n = nx_graph.number_of_nodes()
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=130)

    settled: Set[int] = snapshot["settled"]
    frontier_set: Set[int] = snapshot["frontier_set"]
    current_v: Optional[int] = snapshot["current_vertex"]
    relaxed_edges: Set[Tuple[int, int]] = set(snapshot["relaxed_edges"])
    pivots: Set[int] = snapshot["pivots"]
    buckets: List[Set[int]] = snapshot["buckets"]
    dist: Dict[int, float] = snapshot["dist"]
    prev: Dict[int, Optional[int]] = snapshot["prev"]

    # Map vertex -> bucket index
    vertex_bucket: Dict[int, int] = {}
    for b_idx, b_set in enumerate(buckets):
        for v in b_set:
            vertex_bucket[v] = b_idx

    tree_edges = {(u, v) for v, u in prev.items() if u is not None}
    other_edges = [e for e in nx_graph.edges() if e not in tree_edges and e not in relaxed_edges]
    tree_only = [e for e in tree_edges if e not in relaxed_edges]

    nx.draw_networkx_edges(
        nx_graph, pos, edgelist=other_edges, ax=ax,
        edge_color="#CBD5E1", alpha=0.35, arrows=True, arrowsize=9, width=0.8,
        connectionstyle="arc3,rad=0.05",
    )
    if tree_only:
        nx.draw_networkx_edges(
            nx_graph, pos, edgelist=tree_only, ax=ax,
            edge_color=COLOR_FASTSSSP, alpha=0.8, arrows=True, arrowsize=12, width=1.8,
            connectionstyle="arc3,rad=0.05",
        )
    if relaxed_edges:
        nx.draw_networkx_edges(
            nx_graph, pos, edgelist=list(relaxed_edges), ax=ax,
            edge_color="#E91E63", alpha=1.0, arrows=True, arrowsize=15, width=3.0,
            style="dashed", connectionstyle="arc3,rad=0.05",
        )

    node_colors = []
    edge_rings = []
    ring_widths = []
    for v in nx_graph.nodes():
        if v == source:
            node_colors.append(COLOR_SOURCE)
        elif v in frontier_set:
            node_colors.append(COLOR_FRONTIER)
        elif v in settled:
            node_colors.append(COLOR_DIJKSTRA)
        elif v in vertex_bucket:
            b_idx = vertex_bucket[v]
            node_colors.append(BUCKET_COLORS[b_idx % len(BUCKET_COLORS)])
        else:
            node_colors.append(COLOR_UNVISITED)

        if v == current_v:
            edge_rings.append(COLOR_CURRENT)
            ring_widths.append(3.2)
        elif v in pivots:
            edge_rings.append(COLOR_PIVOT)
            ring_widths.append(2.8)
        else:
            edge_rings.append("white")
            ring_widths.append(1.2)

    node_size = 520 if n <= 30 else 280
    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax, node_color=node_colors,
        node_size=node_size, edgecolors=edge_rings, linewidths=ring_widths,
    )

    labels = {
        v: (
            f"{v}*\n({dist[v]:.0f})" if v in pivots and dist.get(v, math.inf) < math.inf
            else (f"{v}*\n(INF)" if v in pivots
                  else (f"{v}\n({dist[v]:.0f})" if dist.get(v, math.inf) < math.inf else f"{v}\n(INF)"))
        )
        for v in nx_graph.nodes()
    }
    nx.draw_networkx_labels(
        nx_graph, pos, labels=labels, ax=ax,
        font_size=7.5 if n <= 30 else 6.0, font_weight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=COLOR_SOURCE, label="Source"),
        mpatches.Patch(color=COLOR_DIJKSTRA, label="Settled"),
        mpatches.Patch(color=COLOR_FRONTIER, label="In BMSSP Queue"),
        mpatches.Patch(color=BUCKET_COLORS[0], label="Bucketed"),
        mpatches.Patch(facecolor="none", edgecolor=COLOR_PIVOT, linewidth=2, label="Pivot (* )"),
        mpatches.Patch(facecolor="none", edgecolor=COLOR_CURRENT, linewidth=2, label="Current Vertex"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"FastSSSP — Step {snapshot['step']} ({snapshot['phase']})",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    return fig
