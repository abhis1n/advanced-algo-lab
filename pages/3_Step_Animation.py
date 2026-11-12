"""
Page 3: Step-by-Step Algorithm Animation
"""

import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.dijkstra_stepper import dijkstra_steps
from src.fast_sssp_stepper import fast_sssp_steps
from src.graph import generate_medium
from src.ui_helpers import (
    draw_dijkstra_step,
    draw_fast_sssp_step,
    init_session_state,
    set_active_graph,
)

st.set_page_config(page_title="Step Animation | SSSP Visualizer", page_icon="🎬", layout="wide")
init_session_state()

graph = st.session_state["graph"]
nx_graph = st.session_state["nx_graph"]
pos = st.session_state["pos"]
label = st.session_state["graph_label"]

st.title("3. Step-by-Step Algorithm Animation")
st.markdown(
    "Watch how **Dijkstra** (global priority queue) and **FastSSSP** "
    "(pivot selection, bucket partitioning, and BMSSP FIFO relaxation) explore the graph."
)

if graph.n > 60:
    st.warning(
        f"Current graph (**{label}**, n={graph.n}) is large for node-by-node animation. "
        "Animation is clearest on graphs with 10–40 vertices."
    )
    if st.button("Load 20-Vertex Animation Sample Graph", type="primary"):
        sample_g = generate_medium(20, seed=42)
        set_active_graph(sample_g, "Medium (m~5n) n=20", graph_type="Medium")
        st.rerun()

with st.sidebar:
    st.header("Animation Settings")
    mode = st.radio("View Mode", ["Side-by-Side", "Dijkstra Only", "FastSSSP Only"])
    source = st.number_input("Source Vertex", min_value=0, max_value=max(0, graph.n - 1), value=0, step=1)
    speed_ms = st.slider("Playback Speed (ms/step)", min_value=100, max_value=1500, value=450, step=50)

# Compute & cache step traces
cache_key = (label, graph.n, graph.m, int(source))
anim_cache = st.session_state.get("anim_cache")
if anim_cache is None or anim_cache.get("key") != cache_key:
    dij_trace = list(dijkstra_steps(graph, int(source)))
    fast_trace = list(fast_sssp_steps(graph, int(source), seed=0))
    st.session_state["anim_cache"] = {
        "key": cache_key,
        "dijkstra": dij_trace,
        "fast_sssp": fast_trace,
    }
    st.session_state["anim_step"] = 0
    st.session_state["anim_playing"] = False

dij_trace = st.session_state["anim_cache"]["dijkstra"]
fast_trace = st.session_state["anim_cache"]["fast_sssp"]

if mode == "Dijkstra Only":
    max_step = len(dij_trace) - 1
elif mode == "FastSSSP Only":
    max_step = len(fast_trace) - 1
else:
    max_step = max(len(dij_trace), len(fast_trace)) - 1

if "anim_step" not in st.session_state:
    st.session_state["anim_step"] = 0
if "anim_playing" not in st.session_state:
    st.session_state["anim_playing"] = False

st.session_state["anim_step"] = max(0, min(st.session_state["anim_step"], max_step))

# Playback controls bar
c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1.2, 1, 1, 3])
with c1:
    if st.button("⏮ Reset", width="stretch"):
        st.session_state["anim_step"] = 0
        st.session_state["anim_playing"] = False
        st.rerun()
with c2:
    if st.button("◀ Back", width="stretch", disabled=st.session_state["anim_step"] <= 0):
        st.session_state["anim_step"] -= 1
        st.session_state["anim_playing"] = False
        st.rerun()
with c3:
    play_label = "⏸ Pause" if st.session_state["anim_playing"] else "▶ Play"
    if st.button(play_label, type="primary", width="stretch"):
        if not st.session_state["anim_playing"] and st.session_state["anim_step"] >= max_step:
            st.session_state["anim_step"] = 0
        st.session_state["anim_playing"] = not st.session_state["anim_playing"]
        st.rerun()
with c4:
    if st.button("Next ▶", width="stretch", disabled=st.session_state["anim_step"] >= max_step):
        st.session_state["anim_step"] += 1
        st.session_state["anim_playing"] = False
        st.rerun()
with c5:
    if st.button("⏭ End", width="stretch"):
        st.session_state["anim_step"] = max_step
        st.session_state["anim_playing"] = False
        st.rerun()
with c6:
    chosen = st.slider(
        "Step",
        min_value=0,
        max_value=max(1, max_step),
        value=st.session_state["anim_step"],
        label_visibility="collapsed",
    )
    if chosen != st.session_state["anim_step"]:
        st.session_state["anim_step"] = chosen
        st.session_state["anim_playing"] = False

step_idx = st.session_state["anim_step"]


def _render_dijkstra_panel(container, snap):
    with container:
        st.info(f"**Step {snap['step']} / {len(dij_trace)-1} — {snap['phase']}:** {snap['description']}")
        fig = draw_dijkstra_step(nx_graph, pos, snap, int(source))
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        col_a, col_b = st.columns(2)
        col_a.metric("Settled Vertices", f"{len(snap['settled'])} / {graph.n}")
        col_b.metric("Heap / Relax Ops", snap["operations"])

        with st.expander("Priority Queue (Min-Heap) Contents", expanded=True):
            if snap["frontier"]:
                pq_df = pd.DataFrame(
                    [{"Vertex": v, "Tentative Dist": round(d, 1)} for d, v in snap["frontier"][:15]]
                )
                st.dataframe(pq_df, width="stretch", hide_index=True)
            else:
                st.caption("Priority queue is empty.")


def _render_fast_panel(container, snap):
    with container:
        st.warning(f"**Step {snap['step']} / {len(fast_trace)-1} — {snap['phase']}:** {snap['description']}")
        fig = draw_fast_sssp_step(nx_graph, pos, snap, int(source))
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Settled Vertices", f"{len(snap['settled'])} / {graph.n}")
        col_b.metric("Pivots", len(snap["pivots"]))
        col_c.metric("BMSSP Ops", snap["operations"])

        with st.expander("FastSSSP Buckets & BMSSP FIFO Queue", expanded=True):
            if snap["boundaries"]:
                b_rows = []
                for idx, b_set in enumerate(snap["buckets"]):
                    lo = snap["boundaries"][idx]
                    hi = snap["boundaries"][idx + 1]
                    active_mark = "👉 Active" if snap["current_bucket_idx"] == idx else ""
                    b_rows.append({
                        "Bucket": idx + 1,
                        "Range": f"[{lo:.0f}, {hi:.0f})",
                        "Vertices": len(b_set),
                        "Status": active_mark,
                    })
                st.dataframe(pd.DataFrame(b_rows), width="stretch", hide_index=True)
            if snap["frontier"]:
                q_str = ", ".join(
                    f"{v} (d={snap['dist'][v]:.0f})" if snap["dist"].get(v, math.inf) < math.inf else str(v)
                    for v in snap["frontier"][:15]
                )
                st.caption(f"**BMSSP FIFO Queue:** {q_str}")
            else:
                st.caption("BMSSP FIFO queue is empty.")


if mode == "Dijkstra Only":
    _render_dijkstra_panel(st.container(), dij_trace[min(step_idx, len(dij_trace) - 1)])
elif mode == "FastSSSP Only":
    _render_fast_panel(st.container(), fast_trace[min(step_idx, len(fast_trace) - 1)])
else:
    col_l, col_r = st.columns(2)
    _render_dijkstra_panel(col_l, dij_trace[min(step_idx, len(dij_trace) - 1)])
    _render_fast_panel(col_r, fast_trace[min(step_idx, len(fast_trace) - 1)])

# Auto-advance when playing
if st.session_state["anim_playing"]:
    if st.session_state["anim_step"] < max_step:
        time.sleep(speed_ms / 1000.0)
        st.session_state["anim_step"] += 1
        st.rerun()
    else:
        st.session_state["anim_playing"] = False
        st.rerun()
