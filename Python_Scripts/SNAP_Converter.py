import gzip
import os

def convert_snap_to_signed_bipartite(
    input_path, 
    output_path, 
    src_col=0, 
    tgt_col=1, 
    sign_col=2, 
    delimiter=None, 
    pos_val=1, 
    neg_val=-1
):
    """
    Converts a local SNAP/network dataset file into a 0-indexed signed bipartite graph.
    
    Header: nL nU nE
    Edges:  l u sign (l in [0, nL-1], u in [0, nU-1], sign: 0=pos, 1=neg)
    
    Supports: .txt, .csv, .tsv, .gz files with standard edge-list/column formats.
    """
    open_fn = gzip.open if input_path.endswith(".gz") else open
    mode = "rt" if input_path.endswith(".gz") else "r"

    voter_to_id = {}      # Left partition (L)
    candidate_to_id = {}  # Right partition (U)
    edges = {}            # (l_idx, u_idx) -> sign (0 or 1)

    print(f"Reading {input_path}...")
    with open_fn(input_path, mode, encoding="utf-8", errors="ignore") as f:
        # Check if file is record-based (like raw Wiki-RfA) or tabular edge-list
        first_line = f.readline()
        f.seek(0)

        if first_line.startswith("SRC:") or "SRC:" in first_line:
            # Mode A: SNAP Record-based format (e.g., wiki-RfA.txt)
            current = {}
            for line in f:
                line = line.strip()
                if not line:
                    if "SRC" in current and "TGT" in current and "VOT" in current:
                        src, tgt = current["SRC"], current["TGT"]
                        try:
                            vot = int(current["VOT"])
                            if vot in (pos_val, neg_val):
                                if src not in voter_to_id:
                                    voter_to_id[src] = len(voter_to_id)
                                if tgt not in candidate_to_id:
                                    candidate_to_id[tgt] = len(candidate_to_id)
                                
                                l = voter_to_id[src]
                                u = candidate_to_id[tgt]
                                edges[(l, u)] = 0 if vot == pos_val else 1
                        except ValueError:
                            pass
                    current = {}
                elif ":" in line:
                    k, v = line.split(":", 1)
                    current[k.strip()] = v.strip()
        else:
            # Mode B: Tabular / Edge-list format (TSV, CSV, space-separated)
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                
                parts = line.split(delimiter) if delimiter else line.split()
                if len(parts) <= max(src_col, tgt_col, sign_col):
                    continue

                src = parts[src_col]
                tgt = parts[tgt_col]
                raw_sign = parts[sign_col]

                try:
                    val = float(raw_sign)
                    # Sign mapping: > 0 is positive (0), < 0 is negative (1)
                    if val > 0:
                        sign = 0
                    elif val < 0:
                        sign = 1
                    else:
                        continue  # Skip neutral edges (weight 0)
                except ValueError:
                    if raw_sign in ("+", "pos", "1", str(pos_val)):
                        sign = 0
                    elif raw_sign in ("-", "neg", "-1", str(neg_val)):
                        sign = 1
                    else:
                        continue

                if src not in voter_to_id:
                    voter_to_id[src] = len(voter_to_id)
                if tgt not in candidate_to_id:
                    candidate_to_id[tgt] = len(candidate_to_id)

                l = voter_to_id[src]
                u = candidate_to_id[tgt]
                edges[(l, u)] = sign

    nL = len(voter_to_id)
    nU = len(candidate_to_id)
    nE = len(edges)

    print(f"Writing {output_path} (nL={nL}, nU={nU}, nE={nE})...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{nL} {nU} {nE}\n")
        for (l, u), sign in edges.items():
            f.write(f"{l} {u} {sign}\n")

    print(f"Successfully converted: {nL} left nodes, {nU} right nodes, {nE} edges.")


# ==========================================
# Usage Examples
# ==========================================
if __name__ == "__main__":
    # Example 1: Standard edge-list (space or comma separated: src, tgt, weight/sign)
    # convert_snap_to_signed_bipartite("soc-sign-bitcoinotc.csv", "bitcoin_bipartite.txt", src_col=0, tgt_col=1, sign_col=2, delimiter=",")

    # Example 2: Wiki-RfA record file (.txt or .txt.gz)
    convert_snap_to_signed_bipartite("wiki-RfA.txt.gz", "wiki_rfa_converted.txt")