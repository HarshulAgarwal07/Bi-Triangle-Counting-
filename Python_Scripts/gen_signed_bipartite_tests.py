#!/usr/bin/env python3
"""
gen_signed_bipartite_tests.py -- small signed bipartite graphs + exact ground
truth, for demonstrating exactness of signed bi-triangle counters.

WHAT IT DOES
  1. Generates a suite of small synthetic signed bipartite graphs
     (default 40 graphs, every graph has nL + nU <= 49 vertices) covering
     different sizes, densities, degree skew and sign ratios.
  2. Computes the exact balanced / unbalanced bi-triangle counts of every graph
     with an independent Python oracle and writes them to manifest.csv.
  3. (--check) runs your counters on every graph and compares their output to
     the oracle.

OUTPUT FORMAT (same as the C++ programs):
      nL nU nE
      l u s        # nE lines, 0-indexed; s = 0 positive, 1 negative

ORACLE
  Bitmask brute force: for every triple of vertices on the smaller side, take
  the common-neighbour sets of the three pairs (split by wedge parity) and
  count distinct (x,y,z) triples by parity. It is cross-validated at start-up
  against a naive 6-cycle DFS and against closed-form known answers
  (K_{3,3}: 6 cycles, K_{4,4}: 96, K_{2,6}: 0, ...). Use --skip-selftest to
  skip that.

TEST SUITE CONTENT
  * 12 fixed graphs: empty graph, single edge, star, K33 (all +, all -, one
    negative edge: expected 2 balanced / 4 unbalanced), K44, K44 balanced by
    switching, K26 (no bi-triangle but many butterflies), single 6-cycle with
    1 and with 2 negative edges, and a 24x25 dense graph.
    Their hand-derived expected counts are asserted against the oracle.
  * the rest: random graphs cycling through 11 structure families x 11 sign
    models x 4 size classes (deterministic coverage, random parameters).
      families : er_p, er_m, powerlaw_both, powerlaw_L, powerlaw_U, hub_star,
                 lopsided, planted_blocks, sparse_treelike, near_regular,
                 complete
      signs    : iid with P(neg) in {0, .1, .25, .5, .75, .9, 1},
                 switching (balanced by construction, unbalanced must be 0),
                 near_balanced (switching + few flips), hub_neg, block
      sizes    : tiny 3-6, small 6-12, medium 12-24, large (total 38-49)
  * edge order inside each file is shuffled.
  * random graphs are re-drawn until they contain at least one bi-triangle
    (up to --retries attempts), so they actually exercise the counters.

USAGE
  python3 gen_signed_bipartite_tests.py --out tests                 # generate
  python3 gen_signed_bipartite_tests.py --out tests --count 50 --seed 7
  python3 gen_signed_bipartite_tests.py --out tests --check-only \\
      --check "./signed_rswj_count_omp {graph}" \\
      --check "./signed_swj_baseline {graph} --threads=4"
  (--check can be given several times; {graph} is replaced by the file path,
   or the path is appended if there is no placeholder. Generation and
   checking can be combined in one call.)

The checker reads the first line starting with "Balanced" and the first line
starting with "Unbalanced" in the program's stdout ("Balanced bi-triangles :
123"). Exit code is 1 if any graph disagrees with the oracle.
"""

import argparse
import csv
import os
import random
import re
import shlex
import subprocess
import sys
import time
from collections import Counter

popcount = int.bit_count if hasattr(int, "bit_count") else (lambda x: bin(x).count("1"))

# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------

def count_naive(nL, nU, edges):
    """Enumerate every 6-cycle once by DFS. edges: list of (l,u,s)."""
    sign = {(l, u): s for l, u, s in edges}
    adjL = [[] for _ in range(nL)]
    adjU = [[] for _ in range(nU)]
    for l, u, _ in edges:
        adjL[l].append(u)
        adjU[u].append(l)
    bal = unb = 0
    for l1 in range(nL):                       # l1 = smallest L vertex of the cycle
        for u1 in adjL[l1]:
            for l2 in adjU[u1]:
                if l2 <= l1:
                    continue
                for u2 in adjL[l2]:
                    if u2 == u1:
                        continue
                    for l3 in adjU[u2]:
                        if l3 <= l2:           # l1 < l2 < l3 fixes the direction
                            continue
                        for u3 in adjL[l3]:
                            if u3 == u1 or u3 == u2 or (l1, u3) not in sign:
                                continue
                            par = (sign[(l1, u1)] ^ sign[(l2, u1)] ^ sign[(l2, u2)] ^
                                   sign[(l3, u2)] ^ sign[(l3, u3)] ^ sign[(l1, u3)])
                            if par == 0:
                                bal += 1
                            else:
                                unb += 1
    return bal, unb


def count_bitmask(nL, nU, edges):
    """Fast exact oracle: L-triples (smaller side) x bitmask common neighbours."""
    if nL > nU:                                 # triples over the smaller side
        nL, nU = nU, nL
        edges = [(u, l, s) for l, u, s in edges]
    m0 = [0] * nL                               # positive-edge neighbour masks
    m1 = [0] * nL                               # negative-edge neighbour masks
    for a, b, s in edges:
        if s == 0:
            m0[a] |= 1 << b
        else:
            m1[a] |= 1 << b

    def common(a, b):
        even = (m0[a] & m0[b]) | (m1[a] & m1[b])   # wedge parity 0
        odd = (m0[a] & m1[b]) | (m1[a] & m0[b])    # wedge parity 1
        return even, odd

    def bits(mask):
        while mask:
            low = mask & -mask
            yield low.bit_length() - 1
            mask ^= low

    bal = unb = 0
    for a in range(nL):
        for b in range(a + 1, nL):
            xe, xo = common(a, b)
            if not (xe | xo):
                continue
            for c in range(b + 1, nL):
                ye, yo = common(b, c)
                if not (ye | yo):
                    continue
                ze, zo = common(c, a)
                if not (ze | zo):
                    continue
                z = (ze, zo)
                for px, xm in ((0, xe), (1, xo)):
                    for x in bits(xm):
                        for py, ym in ((0, ye), (1, yo)):
                            for y in bits(ym):
                                if y == x:
                                    continue
                                excl = ~((1 << x) | (1 << y))
                                want = px ^ py              # z parity that closes an EVEN cycle
                                bal += popcount(z[want] & excl)
                                unb += popcount(z[1 - want] & excl)
    return bal, unb


def selftest():
    def K(a, b, sgn=lambda l, u: 0):
        return [(l, u, sgn(l, u)) for l in range(a) for u in range(b)]

    known = [
        ("K33 all +", 3, 3, K(3, 3), (6, 0)),
        ("K33 all -", 3, 3, K(3, 3, lambda l, u: 1), (6, 0)),
        ("K33 one - edge", 3, 3, K(3, 3, lambda l, u: 1 if (l, u) == (0, 0) else 0), (2, 4)),
        ("K44 all +", 4, 4, K(4, 4), (96, 0)),
        ("K26 (no bi-triangle)", 2, 6, K(2, 6), (0, 0)),
        ("K55 all +", 5, 5, K(5, 5), (6 * 10 * 10, 0)),
    ]
    for name, nL, nU, ed, exp in known:
        for fn in (count_naive, count_bitmask):
            got = fn(nL, nU, ed)
            if got != exp:
                sys.exit("SELFTEST FAILED (%s): %s gave %s, expected %s" % (name, fn.__name__, got, exp))
    rng = random.Random(2024)
    for t in range(300):
        nL, nU = rng.randint(3, 8), rng.randint(3, 8)
        p, q = rng.uniform(0.2, 0.95), rng.random()
        ed = [(l, u, 1 if rng.random() < q else 0)
              for l in range(nL) for u in range(nU) if rng.random() < p]
        a, b = count_naive(nL, nU, ed), count_bitmask(nL, nU, ed)
        if a != b:
            sys.exit("SELFTEST FAILED: naive %s != bitmask %s on random graph #%d" % (a, b, t))
    print("[selftest] oracle OK (6 known-answer graphs, 300 random naive-vs-bitmask comparisons)")

# --------------------------------------------------------------------------
# Structure generators: return a set of (l,u) pairs
# --------------------------------------------------------------------------

def gen_er_p(rng, nL, nU, p):
    return {(l, u) for l in range(nL) for u in range(nU) if rng.random() < p}


def gen_er_m(rng, nL, nU, m):
    pairs = [(l, u) for l in range(nL) for u in range(nU)]
    return set(rng.sample(pairs, min(m, len(pairs))))


def gen_chung_lu(rng, nL, nU, m, aL, aU):
    """Expected-degree model with power-law weights (alpha = 0 -> uniform side)."""
    wl = [(i + 1) ** -aL for i in range(nL)]
    wu = [(i + 1) ** -aU for i in range(nU)]
    pl, pu = list(range(nL)), list(range(nU))   # shuffle so hubs are not always low ids
    rng.shuffle(pl); rng.shuffle(pu)
    c = m / (sum(wl) * sum(wu))
    return {(pl[i], pu[j]) for i in range(nL) for j in range(nU)
            if rng.random() < min(1.0, c * wl[i] * wu[j])}


def gen_hub_star(rng, nL, nU):
    hl = rng.sample(range(nL), min(nL, rng.randint(1, 3)))
    hu = rng.sample(range(nU), min(nU, rng.randint(1, 3)))
    bg = rng.uniform(0.03, 0.12)
    ed = set()
    for l in range(nL):
        for u in range(nU):
            if l in hl or u in hu:
                if rng.random() < 0.9:
                    ed.add((l, u))
            elif rng.random() < bg:
                ed.add((l, u))
    return ed


def gen_planted_blocks(rng, nL, nU):
    sl, su = rng.randint(1, nL - 1), rng.randint(1, nU - 1)
    pin, pout = rng.uniform(0.6, 0.9), rng.uniform(0.02, 0.1)
    ed = set()
    for l in range(nL):
        for u in range(nU):
            same = (l < sl) == (u < su)
            if rng.random() < (pin if same else pout):
                ed.add((l, u))
    return ed


def gen_near_regular(rng, nL, nU):
    d = rng.randint(2, min(6, nU))
    return {(l, u) for l in range(nL) for u in rng.sample(range(nU), d)}

# --------------------------------------------------------------------------
# Sign models: return list of (l,u,s)
# --------------------------------------------------------------------------

def assign_signs(rng, nL, nU, edges, model, param):
    edges = sorted(edges)
    if model == "iid":
        return [(l, u, 1 if rng.random() < param else 0) for l, u in edges]
    if model in ("switching", "near_balanced"):
        aL = [rng.randint(0, 1) for _ in range(nL)]
        aU = [rng.randint(0, 1) for _ in range(nU)]
        out = []
        for l, u in edges:
            s = aL[l] ^ aU[u]
            if model == "near_balanced" and rng.random() < param:
                s ^= 1
            out.append((l, u, s))
        return out
    if model == "hub_neg":                       # negative iff an endpoint is a hub
        dL, dU = Counter(), Counter()
        for l, u in edges:
            dL[l] += 1; dU[u] += 1
        k = max(1, (nL + nU) // 10)
        hubL = {v for v, _ in dL.most_common(k)}
        hubU = {v for v, _ in dU.most_common(k)}
        return [(l, u, 1 if (l in hubL or u in hubU) else 0) for l, u in edges]
    if model == "block":                         # sign biased by which half the L endpoint is in
        return [(l, u, (1 if rng.random() < 0.85 else 0) if l < nL // 2
                 else (1 if rng.random() < 0.15 else 0)) for l, u in edges]
    raise ValueError(model)


def sign_label(model, param):
    return "%s_%.2f" % (model, param) if param is not None else model

# --------------------------------------------------------------------------
# Plan: fixed graphs + deterministic coverage of family x sign x size
# --------------------------------------------------------------------------

FAMILIES = ["er_p", "er_m", "powerlaw_both", "powerlaw_L", "powerlaw_U", "hub_star",
            "lopsided", "planted_blocks", "sparse_treelike", "near_regular", "complete"]
SIGN_MODELS = [("iid", 0.0), ("iid", 0.1), ("iid", 0.25), ("iid", 0.5), ("iid", 0.75),
               ("iid", 0.9), ("iid", 1.0), ("switching", None), ("near_balanced", 0.08),
               ("hub_neg", None), ("block", None)]
SIZE_CLASSES = ["tiny", "small", "medium", "large"]


def draw_dims(rng, size_class, maxv):
    if size_class == "tiny":
        return rng.randint(3, 6), rng.randint(3, 6)
    if size_class == "small":
        return rng.randint(6, 12), rng.randint(6, 12)
    if size_class == "medium":
        return rng.randint(12, 24), rng.randint(12, 24)
    while True:                                  # large: total 38..maxv
        a, b = rng.randint(8, maxv - 8), rng.randint(8, maxv - 8)
        if 38 <= a + b <= maxv:
            return a, b


def draw_structure(rng, family, size_class, maxv):
    nL, nU = draw_dims(rng, size_class, maxv)
    if family == "lopsided":
        small = rng.randint(3, 8)
        big = rng.randint(30, maxv - small)
        nL, nU = (small, big) if rng.random() < 0.5 else (big, small)
        return nL, nU, gen_er_p(rng, nL, nU, rng.uniform(0.2, 0.7))
    if family == "complete":
        nL, nU = rng.randint(3, 16), rng.randint(3, 16)
        nL, nU = min(nL, maxv - nU), nU
        return nL, nU, {(l, u) for l in range(nL) for u in range(nU)}
    if family == "er_p":
        return nL, nU, gen_er_p(rng, nL, nU, rng.uniform(0.08, 0.95))
    if family == "er_m":
        m = rng.randint(nL + nU, max(nL + nU + 1, int(0.8 * nL * nU)))
        return nL, nU, gen_er_m(rng, nL, nU, m)
    if family in ("powerlaw_both", "powerlaw_L", "powerlaw_U"):
        aL = rng.uniform(0.6, 1.5) if family != "powerlaw_U" else 0.0
        aU = rng.uniform(0.6, 1.5) if family != "powerlaw_L" else 0.0
        m = rng.randint(2 * max(nL, nU), max(2 * max(nL, nU) + 1, int(0.6 * nL * nU)))
        return nL, nU, gen_chung_lu(rng, nL, nU, m, aL, aU)
    if family == "hub_star":
        return nL, nU, gen_hub_star(rng, nL, nU)
    if family == "planted_blocks":
        return nL, nU, gen_planted_blocks(rng, nL, nU)
    if family == "sparse_treelike":
        return nL, nU, gen_er_m(rng, nL, nU, int((nL + nU) * rng.uniform(0.8, 1.6)))
    if family == "near_regular":
        return nL, nU, gen_near_regular(rng, nL, nU)
    raise ValueError(family)


def fixed_graphs(rng, maxv):
    G = []

    def add(name, model, nL, nU, edges, expect=None):
        G.append(dict(family="fixed_" + name, sign_model=model, size_class="fixed",
                      nL=nL, nU=nU, edges=edges, expect=expect, seed=""))

    K = lambda a, b, f=lambda l, u: 0: [(l, u, f(l, u)) for l in range(a) for u in range(b)]
    add("empty", "none", 5, 5, [], (0, 0))
    add("single_edge", "pos", 4, 4, [(1, 2, 0)], (0, 0))
    add("star", "pos", 2, 10, [(0, u, 0) for u in range(10)], (0, 0))
    add("K33", "all_pos", 3, 3, K(3, 3), (6, 0))
    add("K33", "all_neg", 3, 3, K(3, 3, lambda l, u: 1), (6, 0))
    add("K33", "one_neg_edge", 3, 3, K(3, 3, lambda l, u: 1 if (l, u) == (0, 0) else 0), (2, 4))
    add("K44", "all_pos", 4, 4, K(4, 4), (96, 0))
    aL, aU = [rng.randint(0, 1) for _ in range(4)], [rng.randint(0, 1) for _ in range(4)]
    add("K44", "switching", 4, 4, K(4, 4, lambda l, u: aL[l] ^ aU[u]), (96, 0))
    add("K26", "mixed", 2, 6, K(2, 6, lambda l, u: rng.randint(0, 1)), (0, 0))
    c6 = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (0, 2)]
    add("C6", "one_neg", 3, 3, [(l, u, 1 if i == 0 else 0) for i, (l, u) in enumerate(c6)], (0, 1))
    add("C6", "two_neg", 3, 3, [(l, u, 1 if i in (0, 3) else 0) for i, (l, u) in enumerate(c6)], (1, 0))
    nL, nU = 24, min(25, maxv - 24)
    add("max_dense", "iid_0.50", nL, nU,
        [(l, u, rng.randint(0, 1)) for l, u in sorted(gen_er_p(rng, nL, nU, 0.9))])
    return G

# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def stats_of(nL, nU, edges):
    dL, dU = Counter(), Counter()
    neg = 0
    for l, u, s in edges:
        dL[l] += 1; dU[u] += 1; neg += s
    nE = len(edges)
    return dict(nE=nE,
                density=round(nE / (nL * nU), 4) if nL * nU else 0.0,
                neg_frac=round(neg / nE, 4) if nE else 0.0,
                max_deg_L=max(dL.values(), default=0),
                max_deg_U=max(dU.values(), default=0))


def write_graph(path, nL, nU, edges, rng):
    edges = list(edges)
    rng.shuffle(edges)
    with open(path, "w") as f:
        f.write("%d %d %d\n" % (nL, nU, len(edges)))
        for l, u, s in edges:
            f.write("%d %d %d\n" % (l, u, s))


def generate(args):
    os.makedirs(args.out, exist_ok=True)
    master = random.Random(args.seed)
    fixed = fixed_graphs(master, args.max_vertices)
    if args.count < len(fixed):
        sys.exit("--count must be at least %d (the fixed graphs)" % len(fixed))
    specs = list(fixed)
    n_random = args.count - len(fixed)

    t_oracle = 0.0
    rows = []
    for idx in range(args.count):
        if idx < len(fixed):
            g = fixed[idx]
            nL, nU, edges = g["nL"], g["nU"], g["edges"]
            family, model_name, size_class, seed = g["family"], g["sign_model"], "fixed", ""
        else:
            i = idx - len(fixed)
            family = FAMILIES[i % len(FAMILIES)]
            model, param = SIGN_MODELS[(i * 5 + i // len(FAMILIES)) % len(SIGN_MODELS)]
            size_class = SIZE_CLASSES[(i + i // len(FAMILIES)) % len(SIZE_CLASSES)]
            seed = args.seed * 100003 + i
            rng = random.Random(seed)
            model_name = sign_label(model, param)
            for attempt in range(max(1, args.retries)):
                nL, nU, struct = draw_structure(rng, family, size_class, args.max_vertices)
                if nL + nU > args.max_vertices:
                    continue
                edges = assign_signs(rng, nL, nU, struct, model, param)
                if args.no_oracle:
                    break
                bal, unb = count_bitmask(nL, nU, edges)
                if bal + unb > 0:
                    break
            if nL + nU > args.max_vertices:
                sys.exit("internal error: graph exceeds --max-vertices")

        assert nL + nU <= args.max_vertices
        assert all(0 <= l < nL and 0 <= u < nU and s in (0, 1) for l, u, s in edges)
        assert len({(l, u) for l, u, _ in edges}) == len(edges), "duplicate edge"

        fname = "g%02d_%s__%s.txt" % (idx + 1, family, model_name)
        write_graph(os.path.join(args.out, fname), nL, nU, edges, random.Random(args.seed + idx))

        row = dict(file=fname, family=family, sign_model=model_name, size_class=size_class,
                   seed=seed, nL=nL, nU=nU, **stats_of(nL, nU, edges))
        if not args.no_oracle:
            t0 = time.time()
            bal, unb = count_bitmask(nL, nU, edges)
            t_oracle += time.time() - t0
            if idx < len(fixed) and fixed[idx]["expect"] is not None and (bal, unb) != fixed[idx]["expect"]:
                sys.exit("oracle disagrees with the hand-derived answer for %s: got %s, expected %s"
                         % (fname, (bal, unb), fixed[idx]["expect"]))
            if model_name in ("switching",) and unb != 0:
                sys.exit("switching-balanced graph %s has unbalanced=%d (oracle bug?)" % (fname, unb))
            if row["neg_frac"] in (0.0, 1.0) and unb != 0:       # all + or all -: every cycle is balanced
                sys.exit("all-same-sign graph %s has unbalanced=%d (oracle bug?)" % (fname, unb))
            row.update(balanced=bal, unbalanced=unb, total=bal + unb)
        else:
            row.update(balanced="", unbalanced="", total="")
        rows.append(row)

    cols = ["file", "family", "sign_model", "size_class", "seed", "nL", "nU", "nE", "density",
            "neg_frac", "max_deg_L", "max_deg_U", "balanced", "unbalanced", "total"]
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print("[generate] wrote %d graphs + manifest.csv to %s/  (oracle time %.1fs)"
          % (len(rows), args.out, t_oracle))
    summarize(rows)


def summarize(rows):
    def rng_of(key, cast):
        vals = [cast(r[key]) for r in rows]
        return "%s..%s" % (min(vals), max(vals))

    print("  vertices (nL+nU) : %s" % rng_of_expr(rows))
    print("  edges            : %s" % rng_of("nE", int))
    print("  density          : %s" % rng_of("density", float))
    print("  neg fraction     : %s" % rng_of("neg_frac", float))
    if rows[0]["total"] != "":
        nz = sum(1 for r in rows if int(r["total"]) > 0)
        print("  bi-triangles     : %s .. %s total  (%d/%d graphs have >= 1)"
              % (min(int(r["total"]) for r in rows), max(int(r["total"]) for r in rows), nz, len(rows)))
        print("  graphs with unbalanced = 0 : %d, with balanced = 0 : %d"
              % (sum(1 for r in rows if int(r["unbalanced"]) == 0),
                 sum(1 for r in rows if int(r["balanced"]) == 0)))
    print("  families         : %s" % dict(Counter(r["family"] for r in rows)))
    print("  size classes     : %s" % dict(Counter(r["size_class"] for r in rows)))
    print("  sign models      : %s" % dict(Counter(r["sign_model"] for r in rows)))


def rng_of_expr(rows):
    v = [int(r["nL"]) + int(r["nU"]) for r in rows]
    return "%d..%d" % (min(v), max(v))

# --------------------------------------------------------------------------
# Checker
# --------------------------------------------------------------------------

RE_BAL = re.compile(r"^\s*Balanced[^:\n]*:\s*(\d+)", re.M)
RE_UNB = re.compile(r"^\s*Unbalanced[^:\n]*:\s*(\d+)", re.M)


def run_checks(args):
    path = os.path.join(args.out, "manifest.csv")
    if not os.path.exists(path):
        sys.exit("no manifest.csv in %s -- generate first" % args.out)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and rows[0]["balanced"] == "":
        sys.exit("manifest has no oracle counts (generated with --no-oracle)")

    any_fail = False
    for cmd in args.check:
        fails, total_t = [], 0.0
        for r in rows:
            gpath = os.path.join(args.out, r["file"])
            line = cmd.replace("{graph}", shlex.quote(gpath)) if "{graph}" in cmd \
                else cmd + " " + shlex.quote(gpath)
            t0 = time.time()
            try:
                p = subprocess.run(shlex.split(line), capture_output=True, text=True,
                                   timeout=args.timeout)
                out = p.stdout
                rc = p.returncode
            except subprocess.TimeoutExpired:
                out, rc = "", "timeout"
            total_t += time.time() - t0
            mb, mu = RE_BAL.search(out), RE_UNB.search(out)
            got = (int(mb.group(1)), int(mu.group(1))) if mb and mu else None
            exp = (int(r["balanced"]), int(r["unbalanced"]))
            ok = got == exp
            if args.verbose or not ok:
                print("  %-4s %-46s expected %s  got %s%s"
                      % ("ok" if ok else "FAIL", r["file"], exp, got,
                         "" if rc == 0 else "  (exit %s)" % rc))
            if not ok:
                fails.append(r["file"])
        status = "PASS" if not fails else "FAIL"
        print("[check] %s  %d/%d graphs exact  (%.1fs)  <- %s"
              % (status, len(rows) - len(fails), len(rows), total_t, cmd))
        any_fail |= bool(fails)
    return 1 if any_fail else 0

# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Small signed bipartite test graphs with exact ground truth.")
    ap.add_argument("--out", default="tests", help="output directory (default: tests)")
    ap.add_argument("--count", type=int, default=40, help="total graphs incl. 12 fixed ones (default 40)")
    ap.add_argument("--seed", type=int, default=12345, help="master seed (default 12345)")
    ap.add_argument("--max-vertices", type=int, default=49, help="max nL+nU per graph (default 49)")
    ap.add_argument("--retries", type=int, default=60, help="redraws to get >=1 bi-triangle (default 60)")
    ap.add_argument("--no-oracle", action="store_true", help="do not compute ground truth")
    ap.add_argument("--skip-selftest", action="store_true", help="skip oracle self-test")
    ap.add_argument("--check", action="append", default=[],
                    help='command to test, e.g. "./prog {graph}" (repeatable)')
    ap.add_argument("--check-only", action="store_true", help="do not regenerate, only run --check")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-run timeout in seconds")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every graph in --check")
    args = ap.parse_args()

    if args.max_vertices > 49 and not args.check_only:
        print("note: --max-vertices > 49 requested; the default suite stays below 50.", file=sys.stderr)

    if not args.check_only:
        if not args.skip_selftest and not args.no_oracle:
            selftest()
        generate(args)
    rc = 0
    if args.check:
        rc = run_checks(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
