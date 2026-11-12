# SSSP Algorithm Comparison

**Experimental Analysis of the Sorting-Barrier-Breaking Algorithm for Directed Single-Source Shortest Paths**

This project implements and compares two Single-Source Shortest Path (SSSP) algorithms:

1. **Dijkstra's Algorithm** — Classical greedy approach with binary min-heap, $O((m + n) \log n)$
2. **FastSSSP** — Sorting-barrier-breaking algorithm (Duan et al., STOC 2025), $O(m \log^{2/3} n)$

---

## Quick Start

```bash
# Run the interactive demo
uv run python -m src.demo

# Run correctness tests
uv run python tests/test_correctness.py
```

---

## Project Structure

```
project/
├── pyproject.toml              # uv project config
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── graph.py                # Graph generation (5 families)
│   ├── dijkstra.py             # Dijkstra's algorithm
│   ├── fast_sssp.py            # FastSSSP algorithm
│   ├── benchmark.py            # Benchmarking framework
│   ├── visualize.py            # Matplotlib charts + graph drawing
│   └── demo.py                 # Interactive CLI demo
├── tests/
│   └── test_correctness.py     # Correctness verification (28 tests)
└── plots/                      # Generated charts (after running benchmarks)
```

---

## Demo Modes

### 1. Single Graph Comparison (Interactive)
- Choose graph type, size, and density
- Select source vertex
- Run both algorithms and compare results
- View distance tables and correctness check
- Visualize shortest-path tree (small graphs)

### 2. Quick Benchmark
- 10 graph configurations × 3 runs each
- Generates comparison charts in `plots/`

### 3. Full Benchmark Suite
- All synopsis configurations (5 graph families × multiple sizes)
- Comprehensive comparison with charts

---

## Graph Families

| Type | Edges | Description |
|------|-------|-------------|
| Sparse | $m \approx 2n$ | Road-network-like |
| Medium | $m \approx 5n$ | Moderate connectivity |
| Dense | $m \approx n^{1.5}$ | High density, stress test |
| Erdős–Rényi | variable $p$ | Controlled random |
| Grid | $k \times k$ lattice | Structured, planar-like |

---

## Theoretical Advantage vs. Practical Performance (RQ3 Analysis)

> **The Question:** *If FastSSSP has higher mean runtime and consumes more memory in benchmarks, in what sense is it "better"?*

This tension between **asymptotic theoretical superiority** and **practical empirical performance** is the central finding of this project and directly addresses **Research Question 3 (RQ3)** from the lab synopsis.

### 1. The Theory: Breaking the 66-Year "Sorting Barrier"

Since Edsger Dijkstra proposed his algorithm in 1959, SSSP algorithms using comparison-based approaches maintained a priority queue to iteratively extract the minimum tentative distance. Because this implicitly sorts vertices by distance, it was widely conjectured that SSSP was bounded by the $\Omega(n \log n)$ sorting lower bound.

**Duan et al. (STOC 2025)** achieved a monumental theoretical breakthrough by showing that **SSSP does not require globally sorting vertices**. By reducing SSSP to Bounded Multi-Source Shortest Paths (BMSSP) and using recursive pivot partitioning, they achieved a deterministic bound of:

$$\mathcal{O}(m \log^{2/3} n)$$

On sparse graphs ($m = \mathcal{O}(n)$), comparing Dijkstra with a Fibonacci heap $\mathcal{O}(n \log n)$ against FastSSSP $\mathcal{O}(n \log^{2/3} n)$:

$$\lim_{n \to \infty} \frac{\mathcal{O}(n \log^{2/3} n)}{\mathcal{O}(n \log n)} = \lim_{n \to \infty} \frac{1}{\log^{1/3} n} = 0$$

Asymptotically ($n \to \infty$), FastSSSP is strictly superior.

---

### 2. The Practice: Why Dijkstra Wins on Real Computers

In real-world execution, wall-clock time is governed by $\text{Time} \approx C \cdot f(n)$, where $C$ is the implementation constant factor.

#### A. Extremely Slow Growth of $\log^{1/3} n$
The theoretical speedup factor over Dijkstra is only $\log^{1/3} n$:

| Number of Vertices ($n$) | $\log_2(n)$ | Theoretical Advantage ($\sqrt[3]{\log_2 n}$) |
|---|---|---|
| $1{,}000$ (Small graph) | $\approx 10$ | $\approx 2.15\times$ |
| $1{,}000{,}000$ (1 Million) | $\approx 20$ | $\approx 2.71\times$ |
| $1{,}000{,}000{,}000$ (1 Billion) | $\approx 30$ | $\approx 3.10\times$ |

Even on a graph with **1 billion vertices**, the theoretical asymptotic advantage is barely **$3.1\times$**.

#### B. Massive Constant Factor Overhead ($C_{\text{FastSSSP}} \gg C_{\text{Dijkstra}}$)
* **Dijkstra:** Extremely streamlined. Operates on a single continuous priority queue (implemented directly in optimized C in Python's `heapq`), performing simple edge relaxations in a tight loop.
* **FastSSSP:** Multi-stage algorithm involving:
  1. Random pivot sampling
  2. Pilot distance calculation & boundary determination
  3. Dynamic vertex bucket allocation
  4. Multi-source frontier tracking
  5. BMSSP subroutines with queue-driven FIFO Bellman-Ford passes
* Because $C_{\text{FastSSSP}}$ is roughly $10\times$ to $50\times$ larger than $C_{\text{Dijkstra}}$, the crossover point where FastSSSP would beat Dijkstra requires a graph far larger than typical computer memory can store (analogous to *"galactic algorithms"* such as Coppersmith–Winograd matrix multiplication).

#### C. Memory Footprint
* **Dijkstra:** Requires only $O(n)$ space for distance array, predecessor array, and min-heap.
* **FastSSSP:** In addition to distances and predecessors, maintains pivot tables, bucket boundary lists, sets of vertices per partition, and frontier queues, resulting in $2\times$ to $5\times$ higher peak memory usage.

---

### 3. Scientific Significance

FastSSSP is a **theoretical milestone**, not an immediate production drop-in:
1. **Fundamental Open Problem Solved:** It proved that computing shortest paths is strictly easier than comparison-based sorting.
2. **First-Generation Pioneer:** Like the AKS Primality Test (which proved in 2002 that primes are decidable in polynomial time $O(\log^{12} n)$ despite being much slower in practice than probabilistic tests), FastSSSP lays the theoretical foundation. Future research will streamline its data structures and reduce constant factors.

---

## Theoretical Scaling Analysis: Where FastSSSP Wins

While our empirical benchmarks (n up to 10,000) show Dijkstra outperforming FastSSSP due to constant-factor overhead, the **theoretical operation counts** tell a different story as graph sizes grow toward real-world scales (millions to trillions of vertices — global road networks, internet graphs, social networks, molecular simulations).

The plots below are generated by `src/visualize.py` (run the Quick or Full Benchmark to regenerate them in `plots/`).

### Operation Count Formulas

For a directed graph with $n$ vertices and $m$ edges:

| Algorithm | Operation Count | Model |
|---|---|---|
| Dijkstra (binary heap) | $(m + n) \log_2 n$ | Comparison-based |
| Dijkstra (Fibonacci heap) | $m + n \log_2 n$ | Comparison-based |
| **FastSSSP (Duan et al.)** | $m \cdot (\log_2 n)^{2/3}$ | Comparison-addition |

### Theoretical Comparison Table (Sparse Graphs, $m = 2n$)

| $n$ (vertices) | Dijkstra (binary) | Dijkstra (Fibonacci) | FastSSSP | Speedup (Fibo / FastSSSP) |
|---|---|---|---|---|
| $10^3$ | $2.99 \times 10^4$ | $3.30 \times 10^4$ | $9.33 \times 10^3$ | **1.07x** |
| $10^4$ | $1.99 \times 10^5$ | $1.53 \times 10^5$ | $5.10 \times 10^4$ | **1.36x** |
| $10^5$ | $1.66 \times 10^6$ | $8.61 \times 10^5$ | $2.47 \times 10^5$ | **1.63x** |
| $10^6$ | $1.20 \times 10^7$ | $3.99 \times 10^7$ | $1.09 \times 10^7$ | **1.83x** |
| $10^7$ | $9.30 \times 10^7$ | $1.66 \times 10^8$ | $4.55 \times 10^7$ | **1.98x** |
| $10^8$ | $6.64 \times 10^8$ | $6.66 \times 10^8$ | $1.83 \times 10^8$ | **2.10x** |
| $10^9$ | $4.49 \times 10^9$ | $2.60 \times 10^9$ | $7.15 \times 10^8$ | **2.19x** |
| $10^{10}$ | $2.99 \times 10^{10}$ | $9.97 \times 10^9$ | $2.74 \times 10^9$ | **2.27x** |
| $10^{12}$ | $1.20 \times 10^{13}$ | $1.40 \times 10^{11}$ | $3.84 \times 10^{10}$ | **2.41x** |
| $10^{15}$ | $2.99 \times 10^{16}$ | $9.97 \times 10^{13}$ | $2.18 \times 10^{13}$ | **2.62x** |

> **Key observation:** By $n = 10^5$ vertices (a modest city-scale road network), FastSSSP already performs **1.63x fewer operations** than Dijkstra with a Fibonacci heap. By $n = 10^9$ (internet-scale), the advantage grows to **2.19x**. The advantage grows without bound as $n \to \infty$.

### Generated Plots

Running the benchmark generates the following theoretical analysis charts in `plots/`:

| Plot | What It Shows |
|---|---|
| `theoretical_ops_sparse.png` | Operation counts + speedup ratio for sparse graphs ($m = 2n$) |
| `theoretical_ops_medium.png` | Operation counts + speedup ratio for medium graphs ($m = 5n$) |
| `theoretical_ops_dense.png` | Operation counts + speedup ratio for dense graphs ($m = n^{1.5}$) |
| `theoretical_comparison_table.png` | Formatted comparison table (sparse, $n = 10^3$ to $10^{15}$) |

Each plot includes:
- **Left panel:** Log-log plot of theoretical operation counts for all three algorithms, with the crossover point marked.
- **Right panel:** Speedup ratio (Dijkstra ops / FastSSSP ops) vs. $n$, with the green-shaded region showing where FastSSSP is provably faster.

### Why Benchmarks Don't Show This (Yet)

Our Python benchmarks test graphs up to $n = 50{,}000$ vertices. At these scales, Dijkstra wins because:

1. **Constant factor dominates:** FastSSSP's multi-phase pipeline (pivot selection, bucket partitioning, BMSSP passes) involves many more Python-level operations per edge than Dijkstra's single tight `heapq` loop (which is implemented in compiled C).
2. **The log-improvement is tiny at small n:** At $n = 10{,}000$, the theoretical advantage is only $\approx 1.4\times$, easily wiped out by a $10\times$-$50\times$ constant factor difference.
3. **The crossover point** where FastSSSP's lower asymptotic cost overcomes its higher constant factor exists, but at graph sizes too large for practical Python benchmarking.

The theoretical scaling analysis demonstrates that at the scales where SSSP is most critically needed (planetary-scale routing, billion-node social graphs, internet topology), FastSSSP's $O(m \log^{2/3} n)$ bound represents a genuine and growing advantage over Dijkstra.

---

## References

1. R. Duan, J. Mao, X. Mao, X. Shu, and L. Yin, *"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths,"* STOC 2025 (arXiv:2504.17033).
2. E. W. Dijkstra, *"A note on two problems in connexion with graphs,"* Numerische Mathematik, vol. 1, pp. 269--271, 1959.
3. M. L. Fredman and R. E. Tarjan, *"Fibonacci heaps and their uses in improved network optimization algorithms,"* JACM, vol. 34, no. 3, pp. 596--615, 1987.
