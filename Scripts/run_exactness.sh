#!/usr/bin/env bash
# run_exactness.sh
# Runs BBTC (--threads=1 --verify) and M-BBTC (--threads=8) on g01_*.txt ... g40_*.txt.
# Produces a long-format CSV: one row per (graph, run).
# Columns:
#   graph, run, threads, balanced, unbalanced, total, time_sec,
#   stage1_sec ... stage5_sec, bf_bal, bf_unb, bf_total, verify,
#   core_L, core_U, core_E, fanout, exit_code

set -u
shopt -s nullglob

BIN="./signed_rswj_count_omp"
TEST_DIR="Python_Scripts/tests"
OUT_CSV="results/exactness_and_mbbtc.csv"
TIMEOUT=1800   # 30 min per run; raise if needed

[ -x "$BIN" ]      || { echo "ERROR: $BIN not executable" >&2; exit 1; }
[ -d "$TEST_DIR" ] || { echo "ERROR: $TEST_DIR not found" >&2; exit 1; }
mkdir -p "$(dirname "$OUT_CSV")"

tmp1=$(mktemp); tmp2=$(mktemp)
trap 'rm -f "$tmp1" "$tmp2"' EXIT

echo "graph,run,threads,balanced,unbalanced,total,time_sec,stage1_sec,stage2_sec,stage3_sec,stage4_sec,stage5_sec,bf_bal,bf_unb,bf_total,verify,core_L,core_U,core_E,fanout,exit_code" > "$OUT_CSV"

# Parser: reads one run's output, writes one CSV row.
# Arguments: <graph_name> <run_label> <exit_code> <output_file>
append_row() {
    local graph="$1" run="$2" rc="$3" f="$4"
    awk -v graph="$graph" -v run="$run" -v rc="$rc" '
    BEGIN {
        sec="none"
        bal="NA"; unb="NA"; tot="NA"; th="NA"; tm="NA"
        s1="NA"; s2="NA"; s3="NA"; s4="NA"; s5="NA"
        bf_bal="NA"; bf_unb="NA"; bf_tot="NA"; verify="NA"
        cL="NA"; cU="NA"; cE="NA"; fan="NA"
    }
    /^Alive L-vertices/ { cL = $(NF-2) }
    /^Alive U-vertices/ { cU = $(NF-2) }
    /^Core edges/       { cE = $(NF-2) }
    /^Total fanout/     { fan = $NF }
    /^===== Signed RSWJ-Count/ { sec="main"; next }
    /^===== Brute-force/       { sec="bf";   next }
    /^===== Comparison/        { sec="cmp";  next }
    sec=="main" {
        if      (/^Balanced bi-triangles/)   bal = $NF
        else if (/^Unbalanced bi-triangles/) unb = $NF
        else if (/^Total bi-triangles/)      tot = $NF
        else if (/^Threads/)                 th  = $NF
        else if (/^Time/)                    tm  = $(NF-1)
        else if (/^Stage 1 time/)            s1  = $(NF-1)
        else if (/^Stage 2 time/)            s2  = $(NF-1)
        else if (/^Stage 3 time/)            s3  = $(NF-1)
        else if (/^Stage 4 time/)            s4  = $(NF-1)
        else if (/^Stage 5 time/)            s5  = $(NF-1)
    }
    sec=="bf" {
        if      (/^Balanced bi-triangles/)   bf_bal = $NF
        else if (/^Unbalanced bi-triangles/) bf_unb = $NF
        else if (/^Total bi-triangles/)      bf_tot = $NF
    }
    sec=="cmp" {
        if      (/^PASS/) verify="PASS"
        else if (/^FAIL/) verify="FAIL"
    }
    END {
        printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
               graph, run, th, bal, unb, tot, tm,
               s1, s2, s3, s4, s5,
               bf_bal, bf_unb, bf_tot, verify,
               cL, cU, cE, fan, rc)
    }
    ' "$f" >> "$OUT_CSV"
}

total=0
for i in $(seq -w 1 40); do
    matches=( "$TEST_DIR"/g"${i}"_*.txt )
    if [ ${#matches[@]} -eq 0 ]; then
        echo "WARN: no graph matching g${i}_*.txt" >&2
        continue
    fi
    graph="${matches[0]}"
    gname="$(basename "$graph")"
    total=$((total+1))
    echo "[$total/40] $gname" >&2

    # Run 1: BBTC + brute-force verification
    gtimeout "$TIMEOUT" "$BIN" "$graph" --threads=1 --verify > "$tmp1" 2>&1
    append_row "$gname" "BBTC_verify" "$?" "$tmp1"

    # Run 2: M-BBTC, no verification
    gtimeout "$TIMEOUT" "$BIN" "$graph" --threads=8 > "$tmp2" 2>&1
    append_row "$gname" "M-BBTC" "$?" "$tmp2"
done

echo "Done. $total graphs, $(($total*2)) rows -> $OUT_CSV" >&2