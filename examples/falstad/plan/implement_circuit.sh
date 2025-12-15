#!/bin/bash
#
# Implement a Falstad circuit using Claude Code (interactive mode)
#
# Usage:
#   ./implement_circuit.sh [--haiku|--sonnet|--opus] <falstad_file.txt>
#   ./implement_circuit.sh bandpass.txt                  # uses default (sonnet)
#   ./implement_circuit.sh --haiku filt-hipass.txt       # uses haiku (faster, cheaper)
#   ./implement_circuit.sh --opus allpass1.txt           # uses opus (most capable)
#
# Runs interactively - you'll be prompted to approve file writes.
# Run one circuit at a time (sequential).
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEMPLATE="$SCRIPT_DIR/claude_prompt_implement_circuit.md"
# Falstad circuits are at ../from-source/circuit-simulator/src/circuits/ relative to project root
FALSTAD_DIR="$PROJECT_DIR/../from-source/circuit-simulator/src/circuits"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default model
MODEL="sonnet"

usage() {
    echo "Usage: $0 [--haiku|--sonnet|--opus] <falstad_file.txt>"
    echo ""
    echo "Options:"
    echo "  --haiku   Use Haiku model (faster, cheaper)"
    echo "  --sonnet  Use Sonnet model (default, balanced)"
    echo "  --opus    Use Opus model (most capable)"
    echo ""
    echo "Examples:"
    echo "  $0 bandpass.txt"
    echo "  $0 --haiku filt-hipass.txt"
    echo "  $0 --opus allpass1.txt"
    echo ""
    echo "Available compatible circuits:"
    echo "  Run: python $SCRIPT_DIR/analyze_compatibility.py --output table | grep Compatible"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --haiku)
            MODEL="haiku"
            shift
            ;;
        --sonnet)
            MODEL="sonnet"
            shift
            ;;
        --opus)
            MODEL="opus"
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
            FALSTAD_FILE="$1"
            shift
            ;;
    esac
done

# Check we have a circuit file
if [ -z "$FALSTAD_FILE" ]; then
    echo -e "${RED}Error: No circuit file specified${NC}"
    usage
fi

# Validate file exists
if [ ! -f "$FALSTAD_DIR/$FALSTAD_FILE" ]; then
    echo -e "${RED}Error: Falstad file not found: $FALSTAD_DIR/$FALSTAD_FILE${NC}"
    exit 1
fi

# Validate template exists
if [ ! -f "$TEMPLATE" ]; then
    echo -e "${RED}Error: Template not found: $TEMPLATE${NC}"
    exit 1
fi

# Create unique temp file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RANDOM_SUFFIX=$(head /dev/urandom | tr -dc 'a-z0-9' | head -c 6)
BASENAME="${FALSTAD_FILE%.txt}"
TEMP_PROMPT="/tmp/claude_implement_${BASENAME}_${TIMESTAMP}_${RANDOM_SUFFIX}.md"

# Show the Falstad file contents for reference
echo -e "${YELLOW}=== Falstad circuit contents ===${NC}"
cat "$FALSTAD_DIR/$FALSTAD_FILE"
echo ""

# Create prompt by building it in parts (avoids shell escaping issues)
echo -e "${YELLOW}Creating prompt for: $FALSTAD_FILE${NC}"

# Use a temp file for the circuit contents to avoid escaping issues
TEMP_CONTENTS="/tmp/falstad_contents_${RANDOM_SUFFIX}.txt"
cat "$FALSTAD_DIR/$FALSTAD_FILE" > "$TEMP_CONTENTS"

# Build the prompt: replace {{FALSTAD_FILE}} with sed, then use awk to insert file contents
sed "s/{{FALSTAD_FILE}}/$FALSTAD_FILE/g" "$TEMPLATE" | \
awk -v contentsfile="$TEMP_CONTENTS" '
/\{\{FALSTAD_CONTENTS\}\}/ {
    while ((getline line < contentsfile) > 0) print line
    next
}
{ print }
' > "$TEMP_PROMPT"

rm -f "$TEMP_CONTENTS"

echo -e "${GREEN}Prompt created: $TEMP_PROMPT${NC}"
echo ""

# Create log file for this session
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${BASENAME}_${TIMESTAMP}.log"

echo -e "${YELLOW}=== Starting Claude Code ===${NC}"
echo -e "Model: ${CYAN}$MODEL${NC}"
echo "Log file: $LOG_FILE"
echo ""

# Run Claude Code with the prompt (interactive mode for user approval of writes)
cd "$PROJECT_DIR"

# The prompt file will be passed as initial input
claude --print "$(cat "$TEMP_PROMPT")" 2>&1 | tee "$LOG_FILE"

script -q -c "$TEMP_RUNNER" "$LOG_FILE"

EXIT_CODE=$?
rm -f "$TEMP_RUNNER"

# Cleanup temp file
rm -f "$TEMP_PROMPT"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Implementation complete ===${NC}"
    echo "Log saved: $LOG_FILE"
else
    echo ""
    echo -e "${RED}=== Claude Code exited with code $EXIT_CODE ===${NC}"
    echo "Check log: $LOG_FILE"
fi

exit $EXIT_CODE
