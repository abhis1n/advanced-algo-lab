"""
Page 2: Algorithm Race (Side-by-Side Comparison)
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.benchmark import compare_distances, time_algorithm
from src.dijkstra import dijkstra, reconstruct_path as dijkstra_path
from src.fast_sssp import fast_sssp, reconstruct_path as fast_path
from src.ui_helpers import (
    COLOR_DIJKSTRA,
    COLOR_FASTSSSP,
    draw_spt_figure,
    init_session_state,
)

st.set_page_config(page_title="Algorithm Race | SSSP Visualizer", page_icon="🏁", layout="wide")
init_session_state()

graph = st.session_state["graph"]
nx_graph = st.session_state["nx_graph"]
pos = st.session_state["pos"]
label = st.session_state["graph_label"]

st.title("2. Algorithm Race: Dijkstra vs. FastSSSP")
st.markdown(f"Running both algorithms head-to-head on **{label}** ($n={graph.n:,}$, $m={graph.m:,}$).")

with st.sidebar:
    st.header("Race Settings")
    source = st.number_input("Source Vertex", min_value=0, max_value=max(0, graph.n - 1), value=0, step=1)
    use_target = st.checkbox("Highlight Shortest Path to Target", value=True)
    target = None
    if use_target:
        default_target = min(graph.n - 1, max(1, graph.n // 2))
        target = st.number_input("Target Vertex", min_value=0, max_value=max(0, graph.n - 1), value=default_target, step=1)

    run_clicked = st.button("Run Algorithm Race", type="primary", width="stretch")

# Auto-run on initial load or when button clicked or parameters changed
cached = st.session_state.get("race_results")
needs_run = (
    run_clicked
    or cached is None
    or cached.get("source") != source
    or cached.get("graph_label") != label
    or cached.get("n") != graph.n
    or cached.get("m") != graph.m
)

if needs_run:
    dij_res, dij_dist = time_algorithm(dijkstra, graph, int(source), "Dijkstra", label)
    _, dij_prev = dijkstra(graph, int(source))

    fast_res, fast_dist = time_algorithm(fast_sssp, graph, int(source), "FastSSSP", label)
    _, fast_prev = fast_sssp(graph, int(source))

    match, max_err = compare_distances(dij_dist, fast_dist)
    speedup = dij_res.time_seconds / fast_res.time_seconds if fast_res.time_seconds > 0 else math.inf

    st.session_state["race_results"] = {
        "source": int(source),
        "graph_label": label,
        "n": graph.n,
        "m": graph.m,
        "dij_res": dij_res,
        "dij_dist": dij_dist,
        "dij_prev": dij_prev,
        "fast_res": fast_res,
        "fast_dist": fast_dist,
        "fast_prev": fast_prev,
        "match": match,
        "max_err": max_err,
        "speedup": speedup,
    }
    cached = st.session_state["race_results"]

dij_res = cached["dij_res"]
fast_res = cached["fast_res"]
dij_dist = cached["dij_dist"]
fast_dist = cached["fast_dist"]
dij_prev = cached["dij_prev"]
fast_prev = cached["fast_prev"]
match = cached["match"]
max_err = cached["max_err"]
speedup = cached["speedup"]

# Correctness banner + Speedup summary
top_c1, top_c2 = st.columns([2, 1])
with top_c1:
    if match:
        st.success(f"**DISTANCES MATCH** — All {graph.n:,} vertices verified identical (max error: `{max_err:.2e}`).")
    else:
        st.error(f"**MISMATCH DETECTED** — Max distance error: `{max_err:.2e}`.")

with top_c2:
    winner_text = f"FastSSSP {speedup:.2f}x faster" if speedup > 1.0 else f"Dijkstra {1/speedup:.2f}x faster"
    st.metric(
        "Speedup (Dijkstra / FastSSSP)",
        f"{speedup:.3f}x",
        delta=winner_text,
        delta_color="normal" if speedup >= 1.0 else "inverse",
    )

# Side-by-side columns
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Dijkstra (Binary Min-Heap)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Runtime", f"{dij_res.time_seconds * 1000:.3f} ms")
    m2.metric("Peak Memory", f"{dij_res.peak_memory_bytes / 1024:.1f} KB")
    m3.metric("Reachable", f"{dij_res.num_reachable:,} / {graph.n:,}")

    fig_dij = draw_spt_figure(
        nx_graph, pos, dij_dist, dij_prev, int(source), target,
        title="Dijkstra Shortest-Path Tree", tree_color=COLOR_DIJKSTRA,
    )
    st.pyplot(fig_dij, width="stretch")
    plt.close(fig_dij)

    if target is not None:
        path = dijkstra_path(dij_prev, int(source), int(target))
        if path:
            st.caption(f"**Path ({source} → {target}):** {' → '.join(map(str, path))} (dist = {dij_dist[target]:.1f})")
        else:
            st.caption(f"Target vertex {target} is unreachable from {source}.")

with col_right:
    st.subheader("FastSSSP (BMSSP + Pivots)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Runtime", f"{fast_res.time_seconds * 1000:.3f} ms")
    m2.metric("Peak Memory", f"{fast_res.peak_memory_bytes / 1024:.1f} KB")
    m3.metric("Reachable", f"{fast_res.num_reachable:,} / {graph.n:,}")

    fig_fast = draw_spt_figure(
        nx_graph, pos, fast_dist, fast_prev, int(source), target,
        title="FastSSSP Shortest-Path Tree", tree_color=COLOR_FASTSSSP,
    )
    st.pyplot(fig_fast, width="stretch")
    plt.close(fig_fast)

    if target is not None:
        path = fast_path(fast_prev, int(source), int(target))
        if path:
            st.caption(f"**Path ({source} → {target}):** {' → '.join(map(str, path))} (dist = {fast_dist[target]:.1f})")
        else:
            st.caption(f"Target vertex {target} is unreachable from {source}.")

# Distance Comparison Table
st.markdown("---")
st.subheader("Vertex-by-Vertex Distance Comparison")

rows = []
for v in range(graph.n):
    dd = dij_dist.get(v, math.inf)
    fd = fast_dist.get(v, math.inf)
    is_eq = (dd == math.inf and fd == math.inf) or abs(dd - fd) <= 1e-6
    rows.append({
        "Vertex": v,
        "Dijkstra Dist": "∞" if dd == math.inf else round(dd, 2),
        "FastSSSP Dist": "∞" if fd == math.inf else round(fd, 2),
        "Match": "✅" if is_eq else "❌",
    })

df = pd.DataFrame(rows)
st.dataframe(df, width="stretch", height=320)
