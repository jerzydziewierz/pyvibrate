#!/bin/bash
#
# Batch implement multiple Falstad circuits in parallel
#
# Usage:
#   ./implement_batch.sh                    # Run all compatible circuits (sequential)
#   ./implement_batch.sh -j 3               # Run 3 in parallel
#   ./implement_batch.sh -j 3 circuit1.txt circuit2.txt  # Specific circuits, 3 parallel
#   ./implement_batch.sh --list             # List compatible circuits
#   ./implement_batch.sh --dry-run          # Show what would run
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPLEMENT_SCRIPT="$SCRIPT_DIR/implement_circuit.sh"

# Default parallelism
PARALLEL=1
DRY_RUN=false
LIST_ONLY=false
CIRCUITS=()

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    echo "Usage: $0 [OPTIONS] [circuit1.txt circuit2.txt ...]"
    echo ""
    echo "Options:"
    echo "  -j N, --parallel N   Run N circuits in parallel (default: 1)"
    echo "  --list               List all compatible circuits and exit"
    echo "  --dry-run            Show what would be run without executing"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --list                          # See available circuits"
    echo "  $0 bandpass.txt filt-hipass.txt    # Implement specific circuits"
    echo "  $0 -j 3                            # Run all compatible, 3 at a time"
    echo "  $0 -j 2 --dry-run                  # Preview parallel execution"
    exit 0
}

# Get list of compatible but not-yet-implemented circuits
get_compatible_circuits() {
    python3 "$SCRIPT_DIR/analyze_compatibility.py" --output csv 2>/dev/null | \
        tail -n +2 | \
        awk -F',' '$3=="True" && $4=="False" {print $1}'
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
        *)
            CIRCUITS+=("$1")
            shift
            ;;
    esac
done

# List mode
if $LIST_ONLY; then
    echo -e "${CYAN}Compatible circuits not yet implemented:${NC}"
    echo ""
    get_compatible_circuits | while read circuit; do
        echo "  $circuit"
    done
    echo ""
    echo -e "${YELLOW}Total: $(get_compatible_circuits | wc -l) circuits${NC}"
    exit 0
fi

# If no circuits specified, use all compatible ones
if [ ${#CIRCUITS[@]} -eq 0 ]; then
    mapfile -t CIRCUITS < <(get_compatible_circuits)
fi

if [ ${#CIRCUITS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No circuits to implement (all compatible circuits already done?)${NC}"
    exit 0
fi

echo -e "${CYAN}=== Batch Implementation ===${NC}"
echo "Circuits to implement: ${#CIRCUITS[@]}"
echo "Parallelism: $PARALLEL"
echo ""

# Dry run mode
if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN - would execute:${NC}"
    for circuit in "${CIRCUITS[@]}"; do
        echo "  $IMPLEMENT_SCRIPT $circuit"
    done
    exit 0
fi

# Create a status tracking directory
STATUS_DIR="$SCRIPT_DIR/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STATUS_DIR"

echo -e "${YELLOW}Status directory: $STATUS_DIR${NC}"
echo ""

# Function to run a single circuit and track status
run_circuit() {
    local circuit="$1"
    local status_file="$STATUS_DIR/${circuit%.txt}.status"

    echo "RUNNING" > "$status_file"

    if "$IMPLEMENT_SCRIPT" "$circuit"; then
        echo "SUCCESS" > "$status_file"
    else
        echo "FAILED" > "$status_file"
    fi
}

export -f run_circuit
export IMPLEMENT_SCRIPT
export STATUS_DIR

# Run with GNU parallel if available and parallelism > 1, otherwise use xargs or sequential
if [ "$PARALLEL" -gt 1 ]; then
    if command -v parallel &> /dev/null; then
        echo -e "${GREEN}Using GNU parallel with $PARALLEL jobs${NC}"
        printf '%s\n' "${CIRCUITS[@]}" | parallel -j "$PARALLEL" run_circuit {}
    else
        echo -e "${YELLOW}GNU parallel not found, using background jobs${NC}"

        # Simple parallel execution with job control
        running=0
        for circuit in "${CIRCUITS[@]}"; do
            if [ $running -ge $PARALLEL ]; then
                wait -n  # Wait for any job to finish
                ((running--))
            fi

            echo -e "${CYAN}Starting: $circuit${NC}"
            run_circuit "$circuit" &
            ((running++))
        done

        # Wait for remaining jobs
        wait
    fi
else
    # Sequential execution
    for circuit in "${CIRCUITS[@]}"; do
        echo -e "${CYAN}=== Processing: $circuit ===${NC}"
        run_circuit "$circuit"
        echo ""
    done
fi

# Summary
echo ""
echo -e "${CYAN}=== Summary ===${NC}"

SUCCESS=0
FAILED=0

for status_file in "$STATUS_DIR"/*.status; do
    [ -f "$status_file" ] || continue
    circuit=$(basename "$status_file" .status)
    status=$(cat "$status_file")

    if [ "$status" = "SUCCESS" ]; then
        echo -e "  ${GREEN}[SUCCESS]${NC} $circuit"
        ((SUCCESS++))
    else
        echo -e "  ${RED}[FAILED]${NC} $circuit"
        ((FAILED++))
    fi
done

echo ""
echo -e "Total: $((SUCCESS + FAILED)) | ${GREEN}Success: $SUCCESS${NC} | ${RED}Failed: $FAILED${NC}"
echo "Logs in: $SCRIPT_DIR/logs/"
