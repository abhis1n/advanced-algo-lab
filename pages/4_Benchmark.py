"""
Page 4: Benchmark Dashboard
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from src.benchmark import aggregate_results, run_benchmark_suite
from src.dijkstra import dijkstra
from src.fast_sssp import fast_sssp
from src.graph import (
    generate_dense,
    generate_erdos_renyi,
    generate_grid,
    generate_medium,
    generate_sparse,
)
from src.ui_helpers import init_session_state
from src.visualize import (
    plot_memory_comparison,
    plot_runtime_comparison,
    plot_scalability,
    plot_speedup,
)

st.set_page_config(page_title="Benchmark Dashboard | SSSP Visualizer", page_icon="📊", layout="wide")
init_session_state()

st.title("4. Empirical Benchmark Dashboard")
st.markdown(
    "Run controlled benchmarks across multiple graph families and sizes, "
    "measuring wall-clock runtime (`time.perf_counter`), peak memory (`tracemalloc`), "
    "correctness, and speedup."
)

with st.sidebar:
    st.header("Benchmark Parameters")
    selected_families = st.multiselect(
        "Graph Families",
        ["Sparse (m~2n)", "Medium (m~5n)", "Dense (m~n^1.5)", "Erdos-Renyi (p=0.05)", "Grid (k x k)"],
        default=["Sparse (m~2n)", "Medium (m~5n)", "Grid (k x k)"],
    )

    size_options = st.multiselect(
        "Graph Sizes (n)",
        [250, 500, 1000, 2500, 5000],
        default=[500, 1000],
    )

    num_runs = st.slider("Runs per Configuration", min_value=1, max_value=10, value=3)
    seed = st.number_input("Graph Generation Seed", min_value=0, max_value=99999, value=42)

    run_bench = st.button("Run Benchmark Suite", type="primary", width="stretch")

if run_bench:
    if not selected_families or not size_options:
        st.error("Please select at least one graph family and one size.")
    else:
        graphs = []
        with st.spinner("Generating benchmark graph instances..."):
            for fam in selected_families:
                if fam.startswith("Grid"):
                    for n_val in sorted(size_options):
                        k = max(5, int(round(n_val ** 0.5)))
                        g = generate_grid(k, seed=int(seed))
                        graphs.append((f"Grid {k}x{k}", g))
                elif fam.startswith("Sparse"):
                    for n_val in sorted(size_options):
                        graphs.append((f"Sparse n={n_val}", generate_sparse(n_val, seed=int(seed))))
                elif fam.startswith("Medium"):
                    for n_val in sorted(size_options):
                        graphs.append((f"Medium n={n_val}", generate_medium(n_val, seed=int(seed))))
                elif fam.startswith("Dense"):
                    for n_val in sorted(size_options):
                        if n_val <= 2500:
                            graphs.append((f"Dense n={n_val}", generate_dense(n_val, seed=int(seed))))
                elif fam.startswith("Erdos"):
                    for n_val in sorted(size_options):
                        p = 0.05 if n_val <= 1000 else 0.01
                        graphs.append((f"ER n={n_val} p={p}", generate_erdos_renyi(n_val, p=p, seed=int(seed))))

        progress_bar = st.progress(0.0, text="Running benchmarks...")
        results = []
        total = len(graphs)
        for idx, (g_label, g) in enumerate(graphs):
            progress_bar.progress((idx) / max(1, total), text=f"Benchmarking {g_label} (n={g.n}, m={g.m})...")
            res = run_benchmark_suite(
                [(g_label, g)], dijkstra, fast_sssp,
                source=0, num_runs=int(num_runs), verbose=False,
            )
            results.extend(res)
        progress_bar.progress(1.0, text="Generating plots...")

        agg = aggregate_results(results)

        tmp_dir = tempfile.mkdtemp(prefix="sssp_plots_")
        plot_paths = {
            "runtime": plot_runtime_comparison(agg, os.path.join(tmp_dir, "runtime.png")),
            "scalability": plot_scalability(agg, os.path.join(tmp_dir, "scalability.png")),
            "memory": plot_memory_comparison(agg, os.path.join(tmp_dir, "memory.png")),
            "speedup": plot_speedup(agg, os.path.join(tmp_dir, "speedup.png")),
        }

        st.session_state["bench_agg"] = agg
        st.session_state["bench_plots"] = plot_paths
        progress_bar.empty()
        st.success(f"Completed benchmarking across {len(graphs)} configurations ({len(results)} total comparisons).")

if "bench_agg" in st.session_state:
    agg = st.session_state["bench_agg"]
    plot_paths = st.session_state["bench_plots"]

    st.subheader("Aggregated Results")
    table_rows = []
    for label, algos in agg.items():
        dij = algos["Dijkstra"]
        fast = algos["FastSSSP"]
        speedup = dij.mean_time / fast.mean_time if fast.mean_time > 0 else 0.0
        table_rows.append({
            "Graph": label,
            "n": dij.n,
            "m": dij.m,
            "Dijkstra Mean (ms)": round(dij.mean_time * 1000, 3),
            "Dijkstra Std (ms)": round(dij.std_time * 1000, 3),
            "FastSSSP Mean (ms)": round(fast.mean_time * 1000, 3),
            "FastSSSP Std (ms)": round(fast.std_time * 1000, 3),
            "Dijkstra Mem (KB)": round(dij.mean_memory / 1024, 1),
            "FastSSSP Mem (KB)": round(fast.mean_memory / 1024, 1),
            "Speedup (Dij/Fast)": round(speedup, 3),
            "Correct": "✅" if fast.all_correct else "❌",
        })

    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Benchmark Visualizations")

    c1, c2 = st.columns(2)
    with c1:
        st.image(plot_paths["runtime"], caption="Runtime Comparison (Log Scale)", width="stretch")
        st.image(plot_paths["memory"], caption="Peak Memory Usage (KB)", width="stretch")
    with c2:
        st.image(plot_paths["scalability"], caption="Scalability: Runtime vs n", width="stretch")
        st.image(plot_paths["speedup"], caption="Speedup Ratio (Dijkstra / FastSSSP)", width="stretch")
else:
    st.info("Configure your graph families and sizes in the sidebar, then click **Run Benchmark Suite**.")
