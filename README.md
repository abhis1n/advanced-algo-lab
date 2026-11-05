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

## References

1. R. Duan, J. Mao, X. Mao, X. Shu, and L. Yin, *"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths,"* STOC 2025 (arXiv:2504.17033).
2. E. W. Dijkstra, *"A note on two problems in connexion with graphs,"* Numerische Mathematik, vol. 1, pp. 269–271, 1959.
3. M. L. Fredman and R. E. Tarjan, *"Fibonacci heaps and their uses in improved network optimization algorithms,"* JACM, vol. 34, no. 3, pp. 596–615, 1987.
