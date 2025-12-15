#!/bin/bash
#
# Implement a Falstad circuit using Claude Code
#
# Usage:
#   ./implement_circuit.sh <falstad_file.txt>
#   ./implement_circuit.sh bandpass.txt
#   ./implement_circuit.sh filt-hipass.txt
#
# For parallel execution:
#   ./implement_circuit.sh bandpass.txt &
#   ./implement_circuit.sh filt-hipass.txt &
#   ./implement_circuit.sh allpass1.txt &
#   wait
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
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <falstad_file.txt>"
    echo ""
    echo "Examples:"
    echo "  $0 bandpass.txt"
    echo "  $0 filt-hipass.txt"
    echo ""
    echo "Available compatible circuits:"
    echo "  Run: python $SCRIPT_DIR/analyze_compatibility.py --output table | grep Compatible"
    exit 1
}

# Check arguments
if [ $# -lt 1 ]; then
    usage
fi

FALSTAD_FILE="$1"

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

# Perform substitution
echo -e "${YELLOW}Creating prompt for: $FALSTAD_FILE${NC}"
sed "s/{{FALSTAD_FILE}}/$FALSTAD_FILE/g" "$TEMPLATE" > "$TEMP_PROMPT"

echo -e "${GREEN}Prompt created: $TEMP_PROMPT${NC}"
echo ""

# Show the Falstad file contents for reference
echo -e "${YELLOW}=== Falstad circuit contents ===${NC}"
head -20 "$FALSTAD_DIR/$FALSTAD_FILE"
echo "..."
echo ""

# Create log file for this session
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${BASENAME}_${TIMESTAMP}.log"

echo -e "${YELLOW}=== Starting Claude Code ===${NC}"
echo "Log file: $LOG_FILE"
echo ""

# Run Claude Code with the prompt
# Using --print to show conversation, redirecting to log as well
cd "$PROJECT_DIR"

# The prompt file will be passed as initial input
claude --print "$(cat "$TEMP_PROMPT")" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

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
