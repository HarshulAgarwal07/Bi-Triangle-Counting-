#!/usr/bin/env python3
import os
import random
import math

# Set to True if your C++ parser expects "|L| |U| |E|" on the first line
WRITE_HEADER = True
OUTPUT_DIR = "Test_Graphs"
SEED = 42
NEG_PROB = 0.20  # 20% negative (parity = 1), 80% positive (parity = 0)

def generate_powerlaw_bipartite(filename, nL, nU, nE, neg_prob=0.20, seed=42):
    random.seed(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Zipf-like Preferential Attachment weights so the graph has realistic hubs
    weights_L = [1.0 / ((i + 10) ** 0.75) for i in range(nL)]
    weights_U = [1.0 / ((j + 10) ** 0.85) for j in range(nU)]

    # Build cumulative distribution for O(log V) sampling
    import bisect
    cum_L, total_L = [], 0.0
    for w in weights_L:
        total_L += w
        cum_L.append(total_L)

    cum_U, total_U = [], 0.0
    for w in weights_U:
        total_U += w
        cum_U.append(total_U)

    edges = set()
    print(f"Generating {filename}: |L|={nL}, |U|={nU}, target |E|={nE}...")

    while len(edges) < nE:
        rL = random.random() * total_L
        rU = random.random() * total_U
        u = bisect.bisect_left(cum_L, rL)
        v = bisect.bisect_left(cum_U, rU)
        edges.add((u, v))

    # Sort edges for clean file output
    sorted_edges = sorted(edges)

    with open(filepath, "w") as f:
        if WRITE_HEADER:
            f.write(f"{nL} {nU} {nE}\n")
        for u, v in sorted_edges:
            # 0 = positive edge (+1), 1 = negative edge (-1)
            parity = 1 if random.random() < neg_prob else 0
            f.write(f"{u} {v} {parity}\n")

    print(f" -> Saved to {filepath} ({len(sorted_edges)} edges, weights in {{0, 1}})")

if __name__ == "__main__":
    # 1. Syn-Small: Fast enough for naive Signed-SWJ to finish (~10-60s)
    #    while M-BBTC finishes in < 0.02s (proves 100x+ speedup & exact match)
    generate_powerlaw_bipartite(
        filename="signed_syn_small.txt",
        nL=2000,
        nU=1000,
        nE=25000,
        neg_prob=NEG_PROB,
        seed=SEED
    )

    # 2. Syn-Scale: Medium-large benchmark (~1.5M edges) bridging Twitter (834K)
    #    and MovieLens (10M) for smooth multi-core scaling curves
    generate_powerlaw_bipartite(
        filename="signed_syn_scale.txt",
        nL=50000,
        nU=20000,
        nE=1500000,
        neg_prob=NEG_PROB,
        seed=SEED
    )