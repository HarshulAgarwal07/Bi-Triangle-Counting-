# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research implementation of **balanced/unbalanced bi-triangle counting in signed bipartite graphs**. A bi-triangle is a 6-cycle; it is *balanced* when it contains an even number of negative edges. Two papers drive the design:

- Yang et al., VLDB 2021, *Efficient Bi-triangle Counting for Large Bipartite Networks* — the unsigned WJ-Count → SWJ-Count → RSWJ-Count progression (Algorithms 2 and 3).
- Mekala et al. 2026, *Multi-core & GPU-based Balanced Butterfly Counting in Signed Bipartite Graphs* — BB2K sign-parity bucketing, M-BBC multi-core pattern.

Each `.cpp` file is a **standalone single-file program** (graph loading, algorithm, brute-force validator, `main`). There is no build system, no shared header, and no test harness — code is deliberately duplicated across files rather than factored into a library.

## Build and run

Requires GCC or Clang: the sources use `#include <bits/stdc++.h>` and `__int128`, neither of which MSVC supports. The checked-in binaries are macOS arm64.

**On the Windows dev box there is no compiler on PATH, but WSL has g++ 13.3 with libgomp** — build and run there, with the repo visible at `/mnt/d/...`:

```sh
wsl -e bash -lc "cd /mnt/d/Coding/harshulProject/Bi-Triangle-Counting- && g++ -O2 -std=c++17 -fopenmp -o /tmp/omp signed_rswj_count_omp.cpp"
```

Build into `/tmp` rather than the repo so the committed macOS binaries aren't clobbered.

```sh
g++ -O2 -std=c++17          -o signed_swj_count      signed_swj_count.cpp
g++ -O2 -std=c++17          -o rswj_phase_a          rswj_phase_a.cpp
g++ -O2 -std=c++17          -o signed_rswj_count     signed_rswj_count.cpp
g++ -O2 -std=c++17 -fopenmp -o signed_rswj_count_omp signed_rswj_count_omp.cpp

./signed_swj_count       graph.txt [--verify]
./rswj_phase_a           graph.txt          # structural diagnostics + self-checks only
./signed_rswj_count      graph.txt [--verify]
./signed_rswj_count_omp  graph.txt [--threads=N] [--serial] [--cross-check] [--verify]
```

On macOS, Apple Clang needs libomp for the last one: `brew install libomp`, then `clang++ -O2 -std=c++17 -Xpreprocessor -fopenmp -lomp`. Building the OMP file *without* `-fopenmp` is supported and useful — the pragmas are ignored, `omp.h` is behind `#ifdef _OPENMP`, and it runs single-threaded with identical output.

Two correctness tests, both in-file:

- `--verify` runs the brute-force 6-cycle enumerator and diffs it. Cost grows with the *number of bi-triangles*, so keep these graphs small (roughly |L| in the hundreds).
- `--cross-check` (OMP file only) runs serial and parallel and diffs them. Every accumulator is an exact integer, so the two must agree **bit-for-bit** at any thread count — a difference is a parallelization bug, never a rounding artifact. This scales to any graph the serial version can finish, which is why it matters: `--verify` cannot reach the sizes where parallel bugs live.

## Input format

All three programs read the same plain-text format, so test graphs are interchangeable:

```
nL nU nE
l u s      (l in [0,nL), u in [0,nU), s: 0 = positive, 1 = negative)
...
```

No duplicate edges. Multi-edges with conflicting signs are undefined behaviour — dedupe first.

`Python_Scripts/KONECT_Converter.py` and `SNAP_Converter.py` produce this format from KONECT (`out.*`, plain or `.bz2`) and SNAP (edge-list or `SRC:`/`TGT:`/`VOT:` record files, plain or `.gz`) datasets. Both have hardcoded input/output paths in their `if __name__ == "__main__"` block — edit that block to point at a dataset, then `python Python_Scripts/SNAP_Converter.py`. `KONECT_Converter.py` can synthesize negative edges (`inject_neg_prob`) for signed stress tests on unsigned datasets.

## The signed extension (the one idea the whole codebase rests on)

**Central lemma:** a cycle formed by joining two edge-disjoint paths that share both endpoints is balanced iff the two paths have equal *weight* (XOR of their edge signs). The proof is length-independent, so it covers 2-edge wedges (butterflies) and 3-edge super-wedges (bi-triangles) alike.

Consequence: every scalar count in the unsigned algorithms becomes an `(even, odd)` parity pair, and every combinatorial term splits mechanically:

```
balanced   contribution: C(even,2) + C(odd,2)
unbalanced contribution: even * odd
```

All *other* multipliers (degrees, extension-edge counts, `phi`/`psi` sizes) carry over **unchanged** from the unsigned paper. This holds for the acyclic-correction terms too: the edge shared between the two halves being compared self-cancels under XOR, so each correction's balanced/unbalanced split depends only on relative wedge parity, never on the shared edge's own sign. When adding a new term, derive it in the unsigned algorithm first and then apply this substitution — do not re-derive signs from scratch.

## Overflow: where `__int128` is required and where it isn't

The final answers do not fit in 64 bits on real data — the source paper's own Reuters count is ~2.46e19 against `ll`'s ~9.2e18 ceiling — and SWJ-Count's pre-divide raw sums are ~3x the final answer, so they cross the line even earlier. The overflow is silent: it surfaces as a negative or nonsensical count, not a crash.

The rule the code follows, in all four programs:

- **`i128`** for every global accumulator (`Lam_*`, `Bar_*`), every `C(n,2)` (`C2()` takes `ll` and returns `i128`), and every product of two counts — `(i128)se * so`, not `se * so`. A 64-bit product overflows at n ≈ 3e9 per factor, well inside reach.
- **`ll`** for the per-vertex maps (`S`, `T`, `H`) and for `BruteResult`. These hold local wedge/super-wedge counts bounded by the neighbourhood, and cannot approach 9.2e18.

`i128` has no `operator<<`, so every print goes through `i128_to_string()`; brute-force comparisons cast the `ll` side up with `(i128)`. When adding a term, the thing to check is not the accumulator's type but the *product feeding into it* — `a * b` is evaluated in `ll` and truncated before the widening assignment ever happens.

## File-by-file architecture

**`signed_swj_count.cpp` — SWJ-Count (validated).** Two passes over the raw graph, one anchored at every `l in L`, one at every `u in U`. Each pass builds `S` (2-hop wedge counts, parity-split), applies its acyclic correction (Type-L / Type-U), builds `H` (3-hop super-wedge counts), applies the self-intersection fix, then accumulates into `Lam_*`. Result is `(Lam - Bar) / 3` — each bi-triangle has three swj-unit representations.

**`rswj_phase_a.cpp` — structural machinery, unsigned (validated).** Counts nothing. Builds the (2,2)-core by peeling (Lemma 5: every bi-triangle lives in the core, so this is lossless), assigns **DegRank** (a single unified rank over `L ∪ U` by core degree; rank 0 = highest degree = highest priority), and exposes rank-filtered neighbor access. Runs self-checks on the rank permutation, DegRank monotonicity, and the Phi/Psi partition property. Its logic was later copied verbatim into `signed_rswj_count.cpp`, so this file is now **historical rather than load-bearing** — changes to core/rank logic must be made in both files or the copy will drift.

**`signed_rswj_count.cpp` — RSWJ-Count, signed (validated on small and large graphs).** The main deliverable. A single unified pass over `V = L ∪ U`; each anchor `v` explores only rank-higher neighbors via `Psi`, thresholded at its own rank `sigma(v)`. Theorem 3 gives a **unique** representation per bi-triangle, so there is **no divide-by-3** here (unlike SWJ-Count) — a frequent source of confusion when comparing the two files.

### Key shared abstractions in `signed_rswj_count.cpp`

- `RawSignedBiGraph` → `computeCore22()` → `RankedCoreGraph` → `buildDegRankInPlace()` is the fixed pipeline order; `Phi`/`Psi` are only valid after the rank build, since they binary-search rank-sorted adjacency lists.
- `RankedCoreGraph` uses global ids (`gidL(l)=l`, `gidU(u)=nL+u`) so one `rank[]` array spans both layers.
- `phiL/psiL/phiU/psiU` return an `IdxRange` — a zero-copy `{vector*, lo, hi}` slice, *not* a container. `Psi(r,v)` = neighbors with rank **>** r, `Phi(r,v)` = rank **<** r. They partition each neighbor list exactly (asserted by Phase A's self-check).
- Signs live in an `unordered_map<ll,int> signOf` keyed by `l*(nU+1)+u`. Note `sign(l,u)` always takes the L-index first regardless of which layer the anchor is on.
- `i128` accumulators plus `i128_to_string()` for printing; `C2(n)` returns `i128`.
- `processVertexL` / `processVertexU` are mirror images of each other. Any change to one almost always needs the same change in the other.

### The four acyclic corrections (RSWJ)

Driven by `S_even/S_odd` (parity counts of the wedge `v-x-y`) and, for Lemma 3 only, a second degree-weighted split `T_even/T_odd` where `T[y] += phi(sigma(v),x)` is routed into the same parity bucket as the wedge that produced it:

- Type-1.1 (Lemma 2) and Type-1.3+Type-2 (Lemma 4) share a shape and are folded into one multiplier `phi(sigma(v),v) + (coreDeg(y) - 2)`.
- Type-1.2 (Lemma 3) is the highest-risk piece: `Bal += Te*(Se-1) + To*(So-1)`, `Unb += Te*So + To*Se`.
- The self-intersection fix reduces to the constant `sign(v,z)`, subtracted `psi(sigma(v),z)` times.

**Documented assumption:** the paper's Algorithm 3 line 11 reads `Psi(phi(v),y)`, which doesn't typecheck (`phi(v)` is a count, not a rank). This code assumes `Psi(sigma(v),y)`, consistent with every other call site. Brute-force validation has since confirmed it. If a structural (total-count) mismatch ever appears, check this first; if only the balanced/unbalanced *split* is wrong while the total matches, the bug is in parity bucketing — most likely Type-1.2's `T` map. `main()`'s failure output already prints this hint.

## The OpenMP variant

**`signed_rswj_count_omp.cpp` — RSWJ-Count, signed, OpenMP.** Created as a byte-exact copy of `signed_rswj_count.cpp`, with changes confined to scheduling and accumulation. The counting math is untouched, and that separation is the point: `diff signed_rswj_count.cpp signed_rswj_count_omp.cpp` should only ever show scratch/parallel plumbing. If a counting change is made, make it in the serial file first, validate with `--verify`, then port.

- `VertexScratch { S, T, H }` — the three per-vertex maps, allocated once per thread and `.clear()`ed per vertex instead of constructed and destroyed per vertex. `processVertexL`/`processVertexU` take it by reference and clear it themselves, so callers never reset it. It must never be shared between threads. `clear()` doesn't shrink buckets, and the work list hands out heavy vertices first, so peak memory (T × the largest single vertex's maps) is reached almost immediately and held — cap `--threads` if memory is tight.
- `WorkItem` + `buildWorkList()` — L and U merged into **one** list (two separate loops would impose a barrier between layers and idle threads on each tail), sorted by descending fanout = own `coreDeg` + Σ neighbours' `coreDeg`. Heaviest-first matters because a dynamic schedule can only balance what is still unclaimed: a huge vertex handed out last leaves every other thread waiting. The sort has a tiebreak on `(side, idx)` so the list is reproducible.
- `signedRswjCountParallel()` — one `#pragma omp parallel` region. `__int128` is not an OpenMP-reducible type, so there is no `reduction(+:)` here: each thread keeps private `i128` partials plus its own scratch, runs `#pragma omp for schedule(dynamic, 4) nowait`, and merges under **one** `#pragma omp critical` per thread at the end, not per vertex.
- `finalizeCounts()` — the shared `Lam - Bar` tail plus the negative-count consistency check, used by both drivers.

Parallel safety rests on `RankedCoreGraph` being const and read-only during counting (`unordered_map::find` on a const map is safe for concurrent readers), and each vertex writing only into thread-private state.

Measured scaling (synthetic skewed graph, |L|=20k, |U|=15k, |E|=600k, 20-thread box): 23.4s at 1 thread → 6.7s at 4 → 4.1s at 16 → 3.8s at 20. It scales near-linearly to about 4 threads and then flattens; the remaining cost is dominated by cache-missing hash lookups (`signOf`, and the S/T/H maps), not by anything OpenMP is doing. If more speedup is wanted, the next lever is the data layout — replacing `signOf`'s `unordered_map` with a CSR-parallel sign array, and the per-vertex maps with dense arrays indexed by vertex — not the schedule.

## `prompt.txt` is a stale session summary

It describes the chat that produced this code and is now wrong in both places it makes claims about file state — the `__int128` gap it reports in `signed_swj_count.cpp` and the mid-edit `VertexScratch` refactor it reports in `signed_rswj_count.cpp` have both been resolved. Read it for the *derivations* (the lemmas, the parity argument, the Algorithm 3 line-11 reasoning), which remain accurate and are not recorded anywhere else. Do not trust it for what any file currently contains.
