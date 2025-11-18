#!/bin/bash

################################################################################
# Script: run_solution_variance.sh
# Description: Run passive+active comparison with different numbers of solutions
# Usage: ./run_solution_variance.sh [algorithm] [benchmarks...]
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
BASE_DIR="solution_variance_output22"
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

Run passive+active experiments with different numbers of solutions from Phase 1.
Analyzes how the quantity of training examples affects active learning performance.

OPTIONS:
    -a, --algorithm ALGO     Algorithm to use (mquacq2 or growacq). Default: mquacq2
    -b, --benchmarks LIST    Space-separated list of benchmarks to run. Default: all
    -d, --dir DIRECTORY      Base directory with solution variance pickles. 
                            Default: solution_variance_output22
    -h, --help              Show this help message

AVAILABLE BENCHMARKS (with solution variants):
    examtt_v1 (2, 5, 10, 50 solutions)
    examtt_v2 (2, 5, 10, 50 solutions)
    graph_coloring_register (2, 5, 10, 50 solutions)
    jsudoku (2, 20, 200, 500 solutions)
    nurse (2, 5, 10, 50 solutions)

EXAMPLES:
    # Run all benchmarks with MQuAcq-2
    $0

    # Run all benchmarks with GrowAcq
    $0 --algorithm growacq

    # Run specific benchmarks
    $0 --benchmarks examtt_v1 jsudoku nurse

    # Use alternative directory
    $0 --dir solution_variance_output --benchmarks examtt_v1

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        -d|--dir)
            BASE_DIR="$2"
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
echo "  Solution Variance Comparison: Passive+Active with Different Solution Counts"
echo "================================================================================"
echo ""
print_info "Algorithm: ${ALGORITHM}"
print_info "Base directory: ${BASE_DIR}"
if [[ -n "$BENCHMARKS" ]]; then
    print_info "Benchmarks: ${BENCHMARKS}"
else
    print_info "Benchmarks: ALL AVAILABLE"
fi
echo ""

# Check if base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    print_error "Base directory not found: $BASE_DIR"
    echo ""
    print_info "Available directories:"
    ls -d solution_variance_output* 2>/dev/null || echo "  None found"
    exit 1
fi

print_success "Base directory found: $BASE_DIR"
echo ""

# Count available solution variants
print_info "Scanning for solution variance experiments..."
VARIANT_COUNT=$(find "$BASE_DIR" -name "*_phase1.pkl" | wc -l | tr -d ' ')
print_success "Found $VARIANT_COUNT solution variant pickle files"
echo ""

# Show what will be analyzed
print_info "Example variants found:"
find "$BASE_DIR" -type d -name "*_sol*" | head -5 | while read dir; do
    basename "$dir"
done | sed 's/^/  /'
echo ""

# Create output directory
mkdir -p solution_variance_comparison_results

# Build the command
CMD="python3 run_solution_variance_comparison.py --algorithm $ALGORITHM --base_dir $BASE_DIR"
if [[ -n "$BENCHMARKS" ]]; then
    CMD="$CMD --benchmarks $BENCHMARKS"
fi

# Print the command
print_info "Running command:"
echo "    $CMD"
echo ""

# Run the comparison
print_info "Starting solution variance experiments..."
echo "================================================================================"
echo ""

if $CMD; then
    echo ""
    echo "================================================================================"
    print_success "Solution variance comparison completed successfully!"
    echo ""
    print_info "Results saved to: solution_variance_comparison_results/"
    echo ""
    
    # Find the most recent results file
    LATEST_RESULT=$(ls -t solution_variance_comparison_results/solution_variance_${ALGORITHM}_*.json 2>/dev/null | head -1)
    if [[ -n "$LATEST_RESULT" ]]; then
        print_info "Latest results file: $LATEST_RESULT"
        
        # Try to extract summary if jq is available
        if command -v jq &> /dev/null; then
            echo ""
            print_info "Quick Summary:"
            echo "    Algorithm: $(jq -r '.algorithm' "$LATEST_RESULT")"
            echo "    Timestamp: $(jq -r '.timestamp' "$LATEST_RESULT")"
            echo "    Benchmarks: $(jq -r '.benchmarks' "$LATEST_RESULT")"
            echo "    Total experiments: $(jq -r '.total_experiments' "$LATEST_RESULT")"
        fi
    fi
    
    echo ""
    echo "================================================================================"
    exit 0
else
    echo ""
    echo "================================================================================"
    print_error "Solution variance comparison failed!"
    echo ""
    print_info "Check the output above for error details."
    echo "================================================================================"
    exit 1
fi

