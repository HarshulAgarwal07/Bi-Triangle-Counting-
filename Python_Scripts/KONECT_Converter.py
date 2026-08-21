import bz2
import os

def convert_konect_to_signed_bipartite(
    input_path="out.digg-votes",
    output_path="digg_konect_bipartite.txt",
    default_sign=0,
    inject_neg_prob=0.0  # Set > 0 (e.g., 0.25) if you want to generate negative edges for signed stress tests
):
    """
    Converts KONECT bipartite datasets (e.g., out.digg-votes) into:
    
    Header: nL nU nE
    Edges:  l u sign (strictly 0-based: l in [0, nL-1], u in [0, nU-1], sign: 0=pos, 1=neg)
    
    Handles plain text, .bz2, and standard KONECT comment headers (%).
    """
    import random

    is_bz2 = input_path.endswith(".bz2")
    open_fn = bz2.open if is_bz2 else open
    mode = "rt" if is_bz2 else "r"

    left_to_id = {}    # Left partition (Users)
    right_to_id = {}   # Right partition (Stories / Items)
    edges = {}         # (l, u) -> sign

    print(f"Parsing KONECT file: {input_path}...")
    with open_fn(input_path, mode, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            # Skip KONECT comments and metadata headers
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            src = parts[0]
            tgt = parts[1]

            # Re-map vertex identifiers to strict contiguous 0-based indices
            if src not in left_to_id:
                left_to_id[src] = len(left_to_id)
            if tgt not in right_to_id:
                right_to_id[tgt] = len(right_to_id)

            l = left_to_id[src]
            u = right_to_id[tgt]

            # If the dataset has a signed rating column (column 3):
            if len(parts) >= 3 and parts[2] not in ("1", "0") and not parts[2].isdigit():
                try:
                    weight = float(parts[2])
                    sign = 0 if weight > 0 else 1
                except ValueError:
                    sign = default_sign
            else:
                if inject_neg_prob > 0.0:
                    sign = 1 if random.random() < inject_neg_prob else 0
                else:
                    sign = default_sign

            edges[(l, u)] = sign

    nL = len(left_to_id)
    nU = len(right_to_id)
    nE = len(edges)

    print(f"Writing {nE:,} edges to {output_path} (nL={nL:,}, nU={nU:,})...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{nL} {nU} {nE}\n")
        for (l, u), s in edges.items():
            f.write(f"{l} {u} {s}\n")

    print(f"Finished. Graph saved: {nL} left nodes, {nU} right nodes, {nE} edges.")


if __name__ == "__main__":
    # Point this to your uncompressed 'out.digg-votes' or compressed 'digg-votes.tar.bz2' / 'out.digg-votes.bz2'
    convert_konect_to_signed_bipartite(
        input_path="out.epinions",
        output_path="epinions_converted.txt"
    )