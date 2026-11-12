"""
SSSP Algorithm Visualizer — Streamlit Entry Point
==================================================
Multi-page interactive web frontend comparing Dijkstra's algorithm and
FastSSSP (Duan et al., STOC 2025).
"""

import streamlit as st

from src.ui_helpers import init_session_state

st.set_page_config(
    page_title="SSSP Algorithm Visualizer",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# Sidebar active graph summary
with st.sidebar:
    st.markdown("### Active Graph")
    g = st.session_state["graph"]
    st.info(
        f"**{st.session_state['graph_label']}**\n\n"
        f"- Vertices ($n$): **{g.n:,}**\n"
        f"- Edges ($m$): **{g.m:,}**"
    )

# Main landing page
st.title("SSSP Algorithm Visualizer")
st.markdown(
    "**Experimental Analysis of the Sorting-Barrier-Breaking Algorithm "
    "for Directed Single-Source Shortest Paths**"
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. Graph Builder")
    st.write(
        "Create directed weighted graphs across 5 families "
        "(Sparse, Medium, Dense, Erdős–Rényi, Grid) and inspect their structure."
    )
    if st.button("Open Graph Builder →", width="stretch"):
        st.switch_page("pages/1_Graph_Builder.py")

with col2:
    st.markdown("### 2. Algorithm Race")
    st.write(
        "Run Dijkstra and FastSSSP side-by-side on the active graph. "
        "Compare shortest-path trees, runtime, memory, and distance accuracy."
    )
    if st.button("Open Algorithm Race →", width="stretch"):
        st.switch_page("pages/2_Algorithm_Race.py")

with col3:
    st.markdown("### 3. Step Animation")
    st.write(
        "Watch both algorithms explore the graph node-by-node: priority queue "
        "extractions vs. pivot selection, bucketing, and BMSSP relaxation."
    )
    if st.button("Open Step Animation →", width="stretch"):
        st.switch_page("pages/3_Step_Animation.py")

col4, col5 = st.columns(2)

with col4:
    st.markdown("### 4. Benchmark Dashboard")
    st.write(
        "Run configurable multi-run benchmarks across graph families and sizes "
        "to visualize empirical runtime, memory usage, and speedup."
    )
    if st.button("Open Benchmark Dashboard →", width="stretch"):
        st.switch_page("pages/4_Benchmark.py")

with col5:
    st.markdown("### 5. Theoretical Scaling")
    st.write(
        "Explore projected operation counts from $n = 10^3$ to $10^{15}$ "
        "and adjust constant-factor overhead to see where FastSSSP breaks the sorting barrier."
    )
    if st.button("Open Theoretical Scaling →", width="stretch"):
        st.switch_page("pages/5_Theoretical.py")

st.markdown("---")

st.subheader("Algorithms Compared")
st.markdown(
    """
| Algorithm | Time Complexity | Space | Core Mechanism |
|---|---|---|---|
| **Dijkstra (Binary Heap)** | $O((m + n) \\log n)$ | $O(n + m)$ | Global min-heap priority queue (greedy extraction) |
| **Dijkstra (Fibonacci Heap)** | $O(m + n \\log n)$ | $O(n + m)$ | Amortized $O(1)$ decrease-key priority queue |
| **FastSSSP (Duan et al., STOC 2025)** | $O(m \\log^{2/3} n)$ | $O(n + m)$ | BMSSP + recursive pivot partitioning (avoids global sorting) |
"""
)

st.info(
    "**Research Questions:** This dashboard lets you interactively explore "
    "**RQ1** (execution time vs. graph size), **RQ2** (effect of graph density), "
    "and **RQ3** (runtime/memory trade-offs vs. asymptotic theory)."
)
