import os
import bz2
import pandas as pd

def streaming_convert_konect(
    input_path,
    output_path=None,
    default_sign=0,
    chunk_size=5_000_000  # Processes 5 million rows at a time to stay under ~500MB RAM
):
    if not output_path:
        base_name = os.path.basename(input_path).replace(".bz2", "")
        output_path = f"{base_name}_bipartite.txt"

    print(f"🚀 Streaming-parsing KONECT file: {input_path}...")
    
    compression_type = 'bz2' if input_path.endswith(".bz2") else None

    # Global tracking sets for mapping IDs and unique edges without full Pandas dataframes
    left_mapping = {}
    right_mapping = {}
    unique_edges = set()  # Tracks (l, u) to quickly throw away duplicate rows
    
    # Temporary file to store parsed data without holding it in RAM
    temp_output = output_path + ".tmp"
    
    # 1. First Pass: Read in chunks to map IDs and filter duplicate entries
    chunks = pd.read_csv(
        input_path, 
        sep=r'\s+', 
        comment='%', 
        header=None, 
        compression=compression_type,
        on_bad_lines='skip',
        chunksize=chunk_size,
        dtype=str # Keeps IDs uniform as strings to avoid mixed-type bugs
    )

    total_edges = 0
    
    with open(temp_output, "w", encoding="utf-8") as tmp_f:
        for i, chunk in enumerate(chunks):
            print(f"   Processing chunk {i + 1}...")
            
            # Map columns safely
            src_col = chunk.iloc[:, 0].values
            tgt_col = chunk.iloc[:, 1].values
            
            # Extract weights if available
            has_weights = chunk.shape[1] >= 3
            weights = chunk.iloc[:, 2].values if has_weights else None

            for idx in range(len(chunk)):
                src = src_col[idx]
                tgt = tgt_col[idx]
                
                # Assign 0-based contiguous IDs using fast dictionary lookups
                if src not in left_mapping:
                    left_mapping[src] = len(left_mapping)
                if tgt not in right_mapping:
                    right_mapping[tgt] = len(right_mapping)
                    
                l = left_mapping[src]
                u = right_mapping[tgt]
                
                # Instantly drop duplicates using Python's highly optimized hash sets
                if (l, u) in unique_edges:
                    continue
                unique_edges.add((l, u))
                
                # Parse sign
                sign = default_sign
                if has_weights and weights[idx]:
                    try:
                        sign = 0 if float(weights[idx]) > 0 else 1
                    except ValueError:
                        pass
                
                # Stream directly to the temporary file
                tmp_f.write(f"{l} {u} {sign}\n")
                total_edges += 1

    nL = len(left_mapping)
    nU = len(right_mapping)
    
    print(f"💾 Finalising file generation. Writing {total_edges:,} edges...")
    
    # 2. Final Pass: Prepend the required header row and append the data stream
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(f"{nL} {nU} {total_edges}\n")
        
        # Stream from temp file to final destination to avoid memory usage
        with open(temp_output, "r", encoding="utf-8") as tmp_f:
            for line in tmp_f:
                out_f.write(line)
                
    # Clean up the temporary scratch file
    if os.path.exists(temp_output):
        os.remove(temp_output)
        
    print(f"✨ Finished successfully! Saved to: {output_path}\n")


if __name__ == "__main__":
    # Supply any target file name or path dynamically
    streaming_convert_konect(
        input_path="out.twitter", 
        output_path="twitter.txt"
    )
