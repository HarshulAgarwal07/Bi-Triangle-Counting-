#!/usr/bin/env bash
# run_swj_baseline.sh
# Runs signed_swj_baseline with a 6-hour timeout on a fixed set of graphs.
# Each graph gets its own log; a summary CSV records status and wall time.
#
# Usage:
#   ./run_swj_baseline.sh                    # default graphs below
#   ./run_swj_baseline.sh signed_wiki        # override with one graph
#   ./run_swj_baseline.sh a b c              # override with many
#
# Adjust THREADS to change the thread count.

set -u

BIN="./signed_swj_baseline"
GRAPHS_DIR="Test_Graphs"
OUT_DIR="results/swj_baseline"
TIMEOUT="6h"
THREADS=1

# Default set; positional args override.
if [ "$#" -gt 0 ]; then
    GRAPHS=("$@")
else
    GRAPHS=("signed_wiki" "signed_youtube" "signed_twitter")
fi

# --- sanity checks ---
[ -x "$BIN" ] || { echo "ERROR: $BIN not found or not executable" >&2; exit 1; }
[ -d "$GRAPHS_DIR" ] || { echo "ERROR: $GRAPHS_DIR not found" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# --- timeout command (gtimeout on macOS, timeout on Linux) ---
if command -v gtimeout >/dev/null 2>&1; then
    TO_CMD="gtimeout"
elif command -v timeout >/dev/null 2>&1; then
    TO_CMD="timeout"
else
    echo "ERROR: neither gtimeout nor timeout found. Install coreutils." >&2
    exit 1
fi

echo "Binary    : $BIN"
echo "Graphs dir: $GRAPHS_DIR"
echo "Timeout   : $TIMEOUT per graph"
echo "Threads   : $THREADS"
echo "Output    : $OUT_DIR"
echo "Graphs    : ${GRAPHS[*]}"
echo

# --- summary CSV ---
summary="$OUT_DIR/summary.csv"
printf 'graph,status,exit_code,wall_sec,log_file\n' > "$summary"

overall=0
for g in "${GRAPHS[@]}"; do
    graph="$GRAPHS_DIR/${g}.txt"
    if [ ! -f "$graph" ]; then
        echo "WARN: $graph not found; skipping" >&2
        printf '"%s","MISSING",NA,NA,""\n' "$g" >> "$summary"
        overall=1
        continue
    fi

    log="$OUT_DIR/${g}.log"
    echo "=== [$(date +%H:%M:%S)] running $g ==="
    t0=$(date +%s)

    "$TO_CMD" "$TIMEOUT" "$BIN" "$graph" --threads="$THREADS" --core22 \
        > "$log" 2>&1
    rc=$?

    t1=$(date +%s)
    dt=$((t1 - t0))

    case "$rc" in
        0)   status="OK" ;;
        124) status="TIMEOUT" ;;
        *)   status="ERROR($rc)" ;;
    esac

    printf '"%s","%s",%d,%d,"%s"\n' "$g" "$status" "$rc" "$dt" "$log" \
        >> "$summary"

    echo "  $g: $status in ${dt}s (exit=$rc)"
    echo "  log: $log"

    if [ "$rc" -ne 0 ]; then
        overall=1
    fi
done

echo
echo "Done. Summary: $summary"
if [ "$overall" -ne 0 ]; then
    echo "One or more graphs did not finish cleanly (see summary)." >&2
fi
exit "$overall"