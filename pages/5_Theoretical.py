"""
Page 5: Theoretical Scaling Analysis
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.ui_helpers import COLOR_DIJKSTRA, COLOR_FASTSSSP, init_session_state

st.set_page_config(page_title="Theoretical Scaling | SSSP Visualizer", page_icon="📈", layout="wide")
init_session_state()

st.title("5. Theoretical Scaling Analysis: Where FastSSSP Wins")
st.markdown(
    "Explore projected operation counts from $n = 10^3$ up to $n = 10^{20}$ "
    "to visualize how **FastSSSP** ($O(m \\log^{2/3} n)$) asymptotically breaks "
    "the $\\Omega(n \\log n)$ sorting barrier of **Dijkstra's algorithm**."
)

with st.sidebar:
    st.header("Theoretical Parameters")
    density = st.selectbox(
        "Graph Density Model",
        ["Sparse (m = 2n)", "Medium (m = 5n)", "Dense (m = n^1.5)"],
    )
    max_exp = st.slider("Max Graph Size Exponent (10^x)", min_value=6, max_value=20, value=15)
    const_factor = st.slider(
        "FastSSSP Constant Overhead Factor (C_FastSSSP / C_Dijkstra)",
        min_value=1.0,
        max_value=20.0,
        value=1.0,
        step=0.5,
        help="1.0 = pure asymptotic Big-O comparison. Increase to simulate real implementation overhead.",
    )

n_values = np.logspace(3, max_exp, num=500)
log2_n = np.log2(n_values)

if density.startswith("Sparse"):
    m_values = 2.0 * n_values
elif density.startswith("Medium"):
    m_values = 5.0 * n_values
else:
    m_values = n_values ** 1.5

dijkstra_binary = (m_values + n_values) * log2_n
dijkstra_fibo = m_values + n_values * log2_n
fast_sssp_ops = const_factor * m_values * (log2_n ** (2.0 / 3.0))

speedup_binary = dijkstra_binary / fast_sssp_ops
speedup_fibo = dijkstra_fibo / fast_sssp_ops

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=130)

ax1.loglog(n_values, dijkstra_binary, "-", color="#1565C0", linewidth=2, label="Dijkstra (binary heap)")
ax1.loglog(n_values, dijkstra_fibo, "--", color=COLOR_DIJKSTRA, linewidth=2, label="Dijkstra (Fibonacci heap)")
ax1.loglog(n_values, fast_sssp_ops, "-", color=COLOR_FASTSSSP, linewidth=2.5,
           label=f"FastSSSP (C = {const_factor:.1f}x)")

ax1.set_xlabel("Number of Vertices (n)", fontsize=11)
ax1.set_ylabel("Projected Cost (Operations)", fontsize=11)
ax1.set_title(f"Asymptotic Operation Scaling — {density}", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)

crossover_idx = np.where(speedup_fibo > 1.0)[0]
if len(crossover_idx) > 0:
    cross_n = n_values[crossover_idx[0]]
    ax1.axvline(x=cross_n, color="#4CAF50", linestyle=":", linewidth=1.8)
    ax2.axvline(x=cross_n, color="#4CAF50", linestyle=":", linewidth=1.8)

ax2.semilogx(n_values, speedup_binary, "-", color="#1565C0", linewidth=2, label="vs Dijkstra (binary heap)")
ax2.semilogx(n_values, speedup_fibo, "--", color=COLOR_DIJKSTRA, linewidth=2, label="vs Dijkstra (Fibonacci heap)")
ax2.axhline(y=1.0, color="black", linestyle="-", linewidth=0.9, alpha=0.6)
ax2.fill_between(
    n_values, 1.0, speedup_fibo,
    where=speedup_fibo > 1.0,
    alpha=0.18, color=COLOR_FASTSSSP, label="FastSSSP wins (vs Fibonacci)",
)
ax2.set_xlabel("Number of Vertices (n)", fontsize=11)
ax2.set_ylabel("Speedup Ratio (Dijkstra / FastSSSP)", fontsize=11)
ax2.set_title(f"Theoretical Speedup (> 1 means FastSSSP wins)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, which="both", alpha=0.3)
ax2.set_ylim(bottom=0)

fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# Crossover callout
if len(crossover_idx) > 0:
    cross_n = n_values[crossover_idx[0]]
    if cross_n <= 1005:
        st.success(
            f"At **C = {const_factor:.1f}x**, FastSSSP already beats Fibonacci-heap Dijkstra "
            f"across the entire plotted range ($n \\ge 10^3$)."
        )
    else:
        st.info(
            f"At constant overhead **C = {const_factor:.1f}x**, the crossover against "
            f"Fibonacci-heap Dijkstra occurs around **$n \\approx {cross_n:.2e}$** vertices."
        )
else:
    st.warning(
        f"At constant overhead **C = {const_factor:.1f}x**, the crossover against "
        f"Fibonacci-heap Dijkstra exceeds $n = 10^{{{max_exp}}}$ — try increasing the max exponent or lowering C."
    )

# Table
st.subheader(f"Projected Operation Counts ({density}, C = {const_factor:.1f}x)")
table_exps = [e for e in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20] if e <= max_exp]
rows = []
for e in table_exps:
    n = 10.0 ** e
    m = 2 * n if density.startswith("Sparse") else (5 * n if density.startswith("Medium") else n ** 1.5)
    lg = math.log2(n)
    dij_bin = (m + n) * lg
    dij_fib = m + n * lg
    fast = const_factor * m * (lg ** (2.0 / 3.0))
    rows.append({
        "n (vertices)": f"10^{e}",
        "Dijkstra (Binary)": f"{dij_bin:.2e}",
        "Dijkstra (Fibonacci)": f"{dij_fib:.2e}",
        "FastSSSP": f"{fast:.2e}",
        "Speedup (vs Binary)": f"{dij_bin / fast:.2f}x",
        "Speedup (vs Fibo)": f"{dij_fib / fast:.2f}x",
    })

st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.info(
    "**Why empirical benchmarks don't show this crossover yet (RQ3):**\n\n"
    "Because $\\log^{1/3} n$ grows extremely slowly (only $\\approx 3.1\\times$ at $n = 10^9$), "
    "even a modest constant-factor overhead ($C \\ge 4\\times$) pushes the practical crossover point "
    "past billions of vertices. FastSSSP is a landmark theoretical breakthrough proving SSSP is strictly "
    "easier than sorting, even though Dijkstra's tight heap loop remains faster at practical RAM scales."
)
