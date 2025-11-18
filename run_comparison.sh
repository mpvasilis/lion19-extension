#!/bin/bash

################################################################################
# Script: run_comparison.sh
# Description: Easy-to-use wrapper for running Passive+Active vs Active-Only comparison
# Usage: ./run_comparison.sh [algorithm] [benchmarks...]
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ALGORITHM="mquacq2"
BENCHMARKS=""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run comparison experiments between Passive+Active and Active-Only approaches.

OPTIONS:
    -a, --algorithm ALGO    Algorithm to use (mquacq2 or growacq). Default: mquacq2
    -b, --benchmarks LIST   Space-separated list of benchmarks to run. Default: all
    -h, --help             Show this help message

AVAILABLE BENCHMARKS:
    sudoku, sudoku_gt, jsudoku, latin_square, 
    graph_coloring_register, examtt_v1, examtt_v2

EXAMPLES:
    # Run all benchmarks with MQuAcq-2
    $0

    # Run all benchmarks with GrowAcq
    $0 --algorithm growacq

    # Run specific benchmarks with MQuAcq-2
    $0 --benchmarks sudoku jsudoku latin_square

    # Run specific benchmarks with GrowAcq
    $0 --algorithm growacq --benchmarks sudoku sudoku_gt

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        -b|--benchmarks)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                BENCHMARKS="$BENCHMARKS $1"
                shift
            done
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate algorithm
if [[ "$ALGORITHM" != "mquacq2" ]] && [[ "$ALGORITHM" != "growacq" ]]; then
    print_error "Invalid algorithm: $ALGORITHM"
    print_info "Valid algorithms: mquacq2, growacq"
    exit 1
fi

# Print header
echo ""
echo "================================================================================"
echo "  Passive+Active vs Active-Only Comparison"
echo "================================================================================"
echo ""
print_info "Algorithm: ${ALGORITHM}"
if [[ -n "$BENCHMARKS" ]]; then
    print_info "Benchmarks: ${BENCHMARKS}"
else
    print_info "Benchmarks: ALL"
fi
echo ""

# Check if Phase 1 outputs exist
print_info "Checking Phase 1 prerequisites..."

PHASE1_DIR="phase1_output"
if [[ ! -d "$PHASE1_DIR" ]]; then
    print_error "Phase 1 output directory not found: $PHASE1_DIR"
    print_info "Please run Phase 1 first:"
    echo "    python run_phase1_experiments.py"
    exit 1
fi

# Check for Phase 1 pickle files
REQUIRED_PICKLES=(
    "sudoku_phase1.pkl"
    "sudoku_gt_phase1.pkl"
    "jsudoku_phase1.pkl"
    "latin_square_phase1.pkl"
    "graph_coloring_register_phase1.pkl"
    "examtt_v1_phase1.pkl"
    "examtt_v2_phase1.pkl"
)

MISSING_PICKLES=()
for pickle in "${REQUIRED_PICKLES[@]}"; do
    if [[ ! -f "$PHASE1_DIR/$pickle" ]]; then
        MISSING_PICKLES+=("$pickle")
    fi
done

if [[ ${#MISSING_PICKLES[@]} -gt 0 ]]; then
    print_warning "Some Phase 1 pickles are missing:"
    for pickle in "${MISSING_PICKLES[@]}"; do
        echo "    - $PHASE1_DIR/$pickle"
    done
    echo ""
    print_warning "Comparison will skip benchmarks with missing Phase 1 data."
    read -p "Do you want to continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Aborted. Run Phase 1 first:"
        echo "    python run_phase1_experiments.py"
        exit 1
    fi
fi

print_success "Phase 1 prerequisites check complete"
echo ""

# Create comparison_results directory if it doesn't exist
mkdir -p comparison_results

# Build the command
CMD="python run_comparison_passive_active_vs_active_only.py --algorithm $ALGORITHM"
if [[ -n "$BENCHMARKS" ]]; then
    CMD="$CMD --benchmarks $BENCHMARKS"
fi

# Print the command
print_info "Running command:"
echo "    $CMD"
echo ""

# Run the comparison
print_info "Starting comparison experiments..."
echo "================================================================================"
echo ""

if $CMD; then
    echo ""
    echo "================================================================================"
    print_success "Comparison completed successfully!"
    echo ""
    print_info "Results saved to: comparison_results/"
    echo ""
    
    # Find the most recent results file
    LATEST_RESULT=$(ls -t comparison_results/passive_active_vs_active_only_${ALGORITHM}_*.json 2>/dev/null | head -1)
    if [[ -n "$LATEST_RESULT" ]]; then
        print_info "Latest results file: $LATEST_RESULT"
        
        # Try to extract summary statistics if jq is available
        if command -v jq &> /dev/null; then
            echo ""
            print_info "Quick Summary (from results file):"
            echo "    Algorithm: $(jq -r '.algorithm' "$LATEST_RESULT")"
            echo "    Timestamp: $(jq -r '.timestamp' "$LATEST_RESULT")"
            echo "    Benchmarks: $(jq -r '.benchmarks' "$LATEST_RESULT")"
        fi
    fi
    
    echo ""
    echo "================================================================================"
    exit 0
else
    echo ""
    echo "================================================================================"
    print_error "Comparison failed!"
    echo ""
    print_info "Check the output above for error details."
    echo "================================================================================"
    exit 1
fi

