#!/usr/bin/env bash

# Prevent Mac from sleeping until this script terminates (watches script PID $$)
caffeinate -w $$ &

BIN="./signed_rswj_count_omp"
GRAPH_DIR="Test_Graphs"
LOG_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).log"

# Check binary
if [[ ! -x "$BIN" ]]; then
    if [[ -f "$BIN" ]]; then
        chmod +x "$BIN"
    else
        echo "Error: Binary '$BIN' not found in $(pwd)"
        exit 1
    fi
fi

# Check directory
if [[ ! -d "$GRAPH_DIR" ]]; then
    echo "Error: Directory '$GRAPH_DIR' not found in $(pwd)"
    exit 1
fi

GRAPHS=(
    "signed_wiki"
    "signed_youtube"
    "signed_twitter"
    "signed_movielens"
    "signed_epinions"
)

THREADS=(1 2 4 8)

echo "==========================================================" | tee -a "$LOG_FILE"
echo "Starting OpenMP Scaling Benchmark on macOS" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Log:  $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

for graph in "${GRAPHS[@]}"; do
    GRAPH_PATH="${GRAPH_DIR}/${graph}.txt"

    if [[ ! -f "$GRAPH_PATH" ]]; then
        echo "" | tee -a "$LOG_FILE"
        echo "[WARNING] File not found: '$GRAPH_PATH'. Skipping..." | tee -a "$LOG_FILE"
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "==========================================================" | tee -a "$LOG_FILE"
    echo ">>> Running Dataset: $graph ($GRAPH_PATH)" | tee -a "$LOG_FILE"
    echo "==========================================================" | tee -a "$LOG_FILE"

    for t in "${THREADS[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "[$(date +%T)] Starting --threads=$t..." | tee -a "$LOG_FILE"
        
        export OMP_NUM_THREADS=$t
        
        # High-resolution start time using Python
        START_TIME=$(python3 -c 'import time; print(time.time())')
        
        # Execute binary
        $BIN "$GRAPH_PATH" --threads=$t 2>&1 | tee -a "$LOG_FILE"
        STATUS=${PIPESTATUS[0]}
        
        END_TIME=$(python3 -c 'import time; print(time.time())')
        ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.3f}')")

        if [[ $STATUS -eq 0 ]]; then
            echo "[$(date +%T)] Completed: $graph (Threads: $t) in ${ELAPSED}s" | tee -a "$LOG_FILE"
        else
            echo "[$(date +%T)] Failed: $graph (Threads: $t, Exit Code: $STATUS) after ${ELAPSED}s" | tee -a "$LOG_FILE"
        fi
        echo "----------------------------------------------------------" | tee -a "$LOG_FILE"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "Benchmark completed at $(date)." | tee -a "$LOG_FILE"