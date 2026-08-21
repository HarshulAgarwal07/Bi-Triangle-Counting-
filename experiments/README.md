# Performance experiments

Two optimizations were prototyped here, measured, correctness-checked, and are
now **landed in both `signed_rswj_count.cpp` and `signed_rswj_count_omp.cpp`**.
This directory keeps the measurement record, the transformation script, and the
test-data generator.

| File | What it is |
|---|---|
| `patch_land.py` | Applies both landed optimizations to either main source. Re-runnable, asserts every substitution. |
| `gen_graph.py` | Seeded synthetic signed bipartite graph generator (test data) |
| `v_packed.cpp` | Prototype 1 in isolation — sign packed into the adjacency entry. Superseded by the landed code; kept only as the intermediate measurement point. |
| `v_st.cpp` | Prototype 2 — prototype 1 **plus** the S/T merge. Code-identical to what landed. |
| `patch_packed.py`, `patch_st.py` | The original two-stage patch scripts that produced the prototypes. Superseded by `patch_land.py`, which handles both files. |

`patch_land.py` handles the serial file (where `S`/`H` are function locals) and
the OMP file (where they live in `VertexScratch`) from the same substitution
list, which is what makes "both files got identical treatment in the counting
path" a checkable claim rather than a hope. It asserts on every substitution, so
if a source changes shape it fails loudly instead of silently producing
something subtly wrong.

```sh
python3 patch_land.py ../signed_rswj_count.cpp     /tmp/out.cpp
python3 patch_land.py ../signed_rswj_count_omp.cpp /tmp/out.cpp
```

Running it against the landed sources now fails its own assertions — that is the
expected outcome, and is the check that the landing is complete.

## Optimization 1 — pack the sign into the adjacency entry

**The problem.** `RankedCoreGraph::sign()` probed an `unordered_map<ll,int>`
keyed by `l*(nU+1)+u`, and it sat in the innermost loop of both traversals.
Every call was a random hash probe — a cache miss per edge visited.

**Why it was removable.** Every `sign()` call site in the counting path asks for
the sign of an edge that is *already being traversed*: `R.sign(y,x)` inside
`for (y : psiU(x,rv))` is the sign of the very adjacency entry just read. So the
sign is now stored with the entry: `id | (sign << 31)`, ids being < 2^31. One
sequential load yields both, no second array, no extra memory.

**Cost of the change.** `adjL`/`adjU` hold packed entries, so the `Phi`/`Psi`
binary searches compare `rankOfU(ID(lst[mid]))`, the rank sorts unpack, and the
traversal loops read `ID(e)`/`SG(e)`. `signOf` stays on the struct — the
brute-force validator still uses it, and the core build reads it once per edge
to do the packing — but the hot path never touches it.

## Optimization 2 — merge the S and T maps

The S-build probed `S[y]` and `T[y]` separately, then the correction loop probed
`T.at(y)` a third time — three hash lookups per `y`, on identical keys. One map
with a 4-field value (`struct Wedge { ll se, so, te, to; }`) does it in one.

## Measurements

Machine: 13th Gen i7-13650HX (20 logical threads, heterogeneous P/E cores),
g++ 13.3 under WSL, `-O2 -std=c++17 -fopenmp`. Graph `big` below.

| Build | before | after |
|---|---:|---:|
| `signed_rswj_count.cpp` (serial) | 23.5s | **7.4s** |
| `signed_rswj_count_omp.cpp`, 1 thread | 23.3s | **5.2s** |
| `signed_rswj_count_omp.cpp`, 20 threads | 3.81s | **0.73s** |

5.2x over the previous parallel version; 32x over the original serial code.
The serial file stays slower than the OMP file at one thread because it still
constructs and destroys its `S`/`H` maps per vertex — `VertexScratch` is
deliberately OMP-only plumbing.

Intermediate measurement, prototype 1 alone (`v_packed.cpp`): 6.83s serial,
1.27s at 20 threads. So roughly 3.4x from the sign packing and a further 1.8x
(parallel) from the S/T merge.

Baseline thread scaling before either change, for reference: 1→23.4s, 2→12.4s,
4→6.70s, 8→5.44s, 16→4.06s, 20→3.81s. Near-linear to ~4 threads, then
memory-latency bound.

### How the bottleneck was identified

Replacing `sign()`'s body with `return 0` (wrong answers, right timing) took
`big` from 23.4s to 6.1s serial — i.e. **~74% of runtime was that one hash map**,
far more than the per-vertex scratch maps. That measurement is what optimization
1 was based on; it is an upper bound (it also perfects branch prediction), and
the real fix landed at 3.4x against it.

### What the bottleneck is now

`gprof` on the landed code (serial, `big`): `unordered_map<int,pair<ll,ll>>::operator[]`
at **301,220,591 calls**, 34% of self time. That is the `H` map, built per (y,z)
in the 3-hop expansion — the next thing worth attacking. Options, in the order
I'd try them:

1. **Per-thread dense array + touched-list** for `H`, resetting only what was
   written. Fastest, but costs O(|V|) per thread — trivial at 35k vertices,
   ~3GB at 10M vertices x 20 threads, so it needs a size guard with a hash-map
   fallback.
2. **Open-addressing flat map** (~60 lines, no dependency). Smaller win than the
   dense array, but memory stays bounded.
3. **Remove the `rank[]` indirection**: every `Phi`/`Psi` binary-search probe
   does `rankOfU(ID(lst[mid]))`, a random lookup into `rank[]`. Relabelling
   vertices so that id *is* rank after DegRank makes the search compare stored
   ids directly, and lets dense scratch arrays be indexed by rank.
4. **CSR the adjacency** — `vector<vector<int>>` is one heap allocation per
   vertex. Natural to do together with (3).

Tested and **not** worth doing: thread pinning (`OMP_PROC_BIND=close`,
`OMP_PLACES=cores`) changed nothing measurable — 0.68s vs 0.71s at 20 threads.

## Correctness

The landed code passes, in both files: `--verify` against brute force on
`small1`/`small2`/`small3`, `--cross-check` (parallel == serial exactly) on
`small*`, `mid` and `big`, and it reproduces the pre-landing totals on every
graph exactly. `signed_swj_count.cpp` — a structurally different algorithm,
untouched by this work — independently agrees on `mid` (9,230,653,370).

## Reproducing the test graphs

`gen_graph.py out.txt nL nU nE seed [neg_prob]` — seeded, so these are exact.
Degrees are skewed (`index = n * random()^2.2`) to mimic power-law structure.

```sh
python3 gen_graph.py small1.txt   40    30    260  1
python3 gen_graph.py small2.txt   60    45    500  7
python3 gen_graph.py small3.txt   25    25    180 13 0.5
python3 gen_graph.py mid.txt    4000  3000  90000 99
python3 gen_graph.py big.txt   20000 15000 600000  5
```

Expected counts (balanced / unbalanced / total):

| Graph | Balanced | Unbalanced | Total |
|---|---:|---:|---:|
| small1 | 32,173 | 31,941 | 64,114 |
| small2 | 220,447 | 221,068 | 441,515 |
| small3 | 20,634 | 20,947 | 41,581 |
| mid | 4,618,874,346 | 4,611,779,024 | 9,230,653,370 |
| big | 104,393,721,583 | 104,239,780,105 | 208,633,501,688 |

The graph files themselves are not committed (`big.txt` is 7MB); regenerate them.

**Known gap in this test set:** these synthetic graphs are dense enough that
(2,2)-core peeling removes almost nothing, so the core-peeling path is barely
exercised, and no graph here produces counts anywhere near the 64-bit ceiling
(largest is 2.1e11 against 9.2e18). Real KONECT/SNAP datasets via
`Python_Scripts/` would cover both.
