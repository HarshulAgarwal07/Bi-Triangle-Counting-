"""
sign_via_two_faction.py

Takes a bipartite graph already in your "nL nU nE" + "l u sign" format
(e.g. the output of KONECT_Converter.py, where every sign is currently 0)
and writes a NEW file with signs assigned via a planted two-faction model:

    - every L-vertex and every U-vertex gets a random +1/-1 "faction" label
    - an edge (l, u) is positive (0) if l and u agree, negative (1) if they don't

IMPORTANT: with flip_prob=0 (the default) this produces a graph that is
PERFECTLY structurally balanced -- every cycle, including every bi-triangle,
has an even number of negative edges by construction (the labels telescope
to zero around any cycle). That means unbalanced=0 is *guaranteed*, not a
sign of a bug -- but it also means your odd-parity code paths never get
exercised. Set --flip-prob to a small positive value to independently flip
some edges' signs after the fact; this breaks the telescoping and produces
a controllable fraction of unbalanced cycles (roughly
  P(a given 6-cycle is unbalanced) = 1/2 * (1 - (1-2*flip_prob)**6)
so flip_prob=0.01 -> ~6% unbalanced, flip_prob=0.05 -> ~23% unbalanced).

KONECT_Converter.py is left untouched -- this reads its output and produces
a second file.
"""

import argparse
import random


def build_faction_labels(n, seed, p_majority):
    """+1/-1 label per vertex. p_majority = P(label == +1); 0.5 = balanced
    factions (~50% negative edges in expectation); closer to 1.0 or 0.0
    skews most vertices into one faction (fewer negative edges, more
    realistic of e.g. Senate/House-style ~45% or Bonanza-style ~2%)."""
    rng = random.Random(seed)
    return [1 if rng.random() < p_majority else -1 for _ in range(n)]


def sign_via_two_faction(
    input_path,
    output_path=None,
    seed=42,
    p_majority=0.85,
    flip_prob=0.0,
    progress_every=5_000_000,
):
    if not output_path:
        suffix = "_signed_2faction" if flip_prob == 0.0 else f"_signed_2faction_flip{flip_prob}"
        output_path = input_path.replace(".txt", "") + suffix + ".txt"

    print(f"🎲 Reading unsigned graph: {input_path}...")

    with open(input_path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        nL, nU, nE = int(header[0]), int(header[1]), int(header[2])

        print(f"   nL={nL:,} nU={nU:,} nE={nE:,}")
        print(f"   assigning factions (seed={seed}, p_majority={p_majority}, flip_prob={flip_prob})...")

        label_L = build_faction_labels(nL, seed, p_majority)
        label_U = build_faction_labels(nU, seed + 1, p_majority)  # independent stream
        rng_flip = random.Random(seed + 2)  # separate stream, only consumed if flip_prob > 0

        pos_count = 0
        neg_count = 0
        flipped_count = 0

        with open(output_path, "w", encoding="utf-8") as out:
            out.write(f"{nL} {nU} {nE}\n")

            for i, line in enumerate(f):
                parts = line.split()
                if len(parts) < 2:
                    continue
                l, u = int(parts[0]), int(parts[1])

                sign = 0 if label_L[l] == label_U[u] else 1

                if flip_prob > 0.0 and rng_flip.random() < flip_prob:
                    sign = 1 - sign
                    flipped_count += 1

                if sign == 0:
                    pos_count += 1
                else:
                    neg_count += 1

                out.write(f"{l} {u} {sign}\n")

                if progress_every and (i + 1) % progress_every == 0:
                    print(f"   ...{i + 1:,} edges signed")

    total = pos_count + neg_count
    pct_neg = 100 * neg_count / total if total else 0.0
    print(f"✨ Done. {pos_count:,} positive, {neg_count:,} negative "
          f"({pct_neg:.1f}% negative), {flipped_count:,} edges flipped from base signing")
    print(f"   Saved to: {output_path}")

    return pos_count, neg_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="unsigned graph file (nL nU nE header + l u sign lines)")
    parser.add_argument("-o", "--output", dest="output_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--p-majority",
        type=float,
        default=0.85,
        help="P(vertex label == +1); 0.5 = ~50%% negative edges, closer to 0/1 = fewer negative edges",
    )
    parser.add_argument(
        "--flip-prob",
        type=float,
        default=0.0,
        help="probability of flipping an edge's sign after two-faction assignment; "
             "0.0 (default) gives a perfectly balanced graph (unbalanced=0 guaranteed); "
             "set >0 to get real unbalanced bi-triangles (e.g. 0.01-0.05)",
    )
    args = parser.parse_args()

    sign_via_two_faction(
        input_path=args.input_path,
        output_path=args.output_path,
        seed=args.seed,
        p_majority=args.p_majority,
        flip_prob=args.flip_prob,
    )
