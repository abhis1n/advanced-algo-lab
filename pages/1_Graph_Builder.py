"""
Page 1: Graph Builder & Visualizer
"""

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from src.graph import (
    generate_dense,
    generate_erdos_renyi,
    generate_grid,
    generate_medium,
    generate_sparse,
)
from src.ui_helpers import (
    draw_graph_preview,
    init_session_state,
    set_active_graph,
)

st.set_page_config(page_title="Graph Builder | SSSP Visualizer", page_icon="🕸️", layout="wide")
init_session_state()

st.title("1. Graph Builder & Visualizer")
st.markdown("Generate a directed weighted graph (non-negative weights in `[1, 1000]`) and inspect its properties.")

# Controls in sidebar
with st.sidebar:
    st.header("Graph Configuration")
    family = st.selectbox(
        "Graph Family",
        [
            "Sparse (m ≈ 2n)",
            "Medium (m ≈ 5n)",
            "Dense (m ≈ n^1.5)",
            "Erdős–Rényi G(n, p)",
            "Grid / Lattice (k × k)",
        ],
    )

    if family == "Grid / Lattice (k × k)":
        k = st.slider("Grid dimension (k)", min_value=3, max_value=22, value=5)
        n_val = k * k
        st.caption(f"Total vertices: n = {k} × {k} = {n_val}")
    else:
        n_val = st.slider("Number of vertices (n)", min_value=10, max_value=500, value=25, step=5)
        k = None

    p_val = 0.1
    if family == "Erdős–Rényi G(n, p)":
        p_val = st.slider("Edge probability (p)", min_value=0.01, max_value=0.30, value=0.10, step=0.01)

    seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42, step=1)

    if st.button("Generate Graph", type="primary", width="stretch"):
        if family.startswith("Sparse"):
            g = generate_sparse(n_val, seed=int(seed))
            g_type = "Sparse"
            label = f"Sparse (m~2n) n={n_val}"
        elif family.startswith("Medium"):
            g = generate_medium(n_val, seed=int(seed))
            g_type = "Medium"
            label = f"Medium (m~5n) n={n_val}"
        elif family.startswith("Dense"):
            g = generate_dense(n_val, seed=int(seed))
            g_type = "Dense"
            label = f"Dense (m~n^1.5) n={n_val}"
        elif family.startswith("Erdős"):
            g = generate_erdos_renyi(n_val, p=float(p_val), seed=int(seed))
            g_type = "Erdos-Renyi"
            label = f"Erdos-Renyi n={n_val} p={p_val:.2f}"
        else:
            g = generate_grid(int(k), seed=int(seed))
            g_type = "Grid"
            label = f"Grid {k}x{k}"

        set_active_graph(g, label=label, graph_type=g_type, grid_k=k)
        st.success(f"Generated {label}")

# Active graph stats
graph = st.session_state["graph"]
nx_graph = st.session_state["nx_graph"]
pos = st.session_state["pos"]
label = st.session_state["graph_label"]

max_edges = graph.n * (graph.n - 1) if graph.n > 1 else 1
density_pct = (graph.m / max_edges) * 100.0 if graph.n > 1 else 0.0
avg_deg = graph.m / graph.n if graph.n > 0 else 0.0
weak_cc = nx.number_weakly_connected_components(nx_graph) if graph.n > 0 else 0
strong_cc = nx.number_strongly_connected_components(nx_graph) if graph.n > 0 else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Vertices (n)", f"{graph.n:,}")
m2.metric("Edges (m)", f"{graph.m:,}")
m3.metric("Density", f"{density_pct:.2f}%")
m4.metric("Avg Out-Degree", f"{avg_deg:.2f}")
m5.metric("Weak / Strong CC", f"{weak_cc} / {strong_cc}")

# Render preview
if graph.n > 200:
    st.warning("Graph has > 200 vertices; node/edge labels are hidden to keep the visualization legible.")

fig = draw_graph_preview(graph, nx_graph, pos, title=f"Active Graph: {label}")
st.pyplot(fig, width="stretch")
plt.close(fig)

with st.expander("Inspect Edge List (first 100 edges)"):
    edge_rows = []
    for u in range(graph.n):
        for v, w in graph.neighbors(u):
            edge_rows.append({"From (u)": u, "To (v)": v, "Weight": w})
            if len(edge_rows) >= 100:
                break
        if len(edge_rows) >= 100:
            break
    st.dataframe(edge_rows, width="stretch")
