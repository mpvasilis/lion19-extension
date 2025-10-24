"""
Generate Results Table from Complete Pipeline Results

This script reads Phase 1, Phase 2, and Phase 3 outputs from the HCAR pipeline
and generates a comprehensive table with all metrics as specified in the paper format.

Table Columns:
- Prob.: Problem name
- Sols: Number of given solutions (positive examples)
- StartC: Number of candidate constraints from passive learning (Phase 1)
- InvC: Number of constraints invalidated by refinement (Phase 2)
- CT: Number of AllDifferent constraints in target model
- Bias: Size of generated bias (Phase 1)
- ViolQ: Violation queries (Phase 2 refinement)
- MQuQ: Active learning queries (Phase 3 MQuAcq-2)
- TQ: Total queries (ViolQ + MQuQ)
- ALQ: Total queries for purely Active Learning baseline (MQuAcq-2)
- PAQ: Total queries for Passive+Active baseline (no refinement)
- VT(s): Duration of violation phase (Phase 2)
- MQuT(s): Duration of active learning phase (Phase 3)
- TT(s): Overall runtime
- ALT(s): Total runtime for Active Learning baseline
- PAT(s): Total runtime for Passive+Active baseline
"""

import os
import sys
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


# Benchmark display names mapping
BENCHMARK_DISPLAY_NAMES = {
    'sudoku': 'Sudoku',
    'sudoku_gt': 'Sudoku-GT',
    'jsudoku': 'JSudoku',
    'latin_square': 'Latin Square',
    'graph_coloring_register': 'Graph Coloring',
    'examtt_v1': 'ExamTT-V1',
    'examtt_v2': 'ExamTT-V2'
}

# Target constraint counts (ground truth for each benchmark)
TARGET_CONSTRAINTS = {
    'sudoku': 27,  # 9 rows + 9 cols + 9 blocks
    'sudoku_gt': 37,  # 27 AllDifferent + 10 greater-than
    'jsudoku': 31,  # JSudoku variant
    'latin_square': 18,  # 9 rows + 9 cols
    'graph_coloring_register': 5,  # Register allocation
    'examtt_v1': 7,  # Small exam timetabling
    'examtt_v2': 9   # Large exam timetabling
}


def load_phase1_data(benchmark_name: str) -> Dict[str, Any]:
    """Load Phase 1 pickle data."""
    pickle_path = f"phase1_output/{benchmark_name}_phase1.pkl"
    
    if not os.path.exists(pickle_path):
        print(f"[WARNING] Phase 1 pickle not found: {pickle_path}")
        return None
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load Phase 1 data for {benchmark_name}: {e}")
        return None


def load_phase3_results(benchmark_name: str) -> Dict[str, Any]:
    """Load Phase 3 JSON results."""
    json_path = f"phase3_output/{benchmark_name}_phase3_results.json"
    
    if not os.path.exists(json_path):
        print(f"[WARNING] Phase 3 results not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load Phase 3 results for {benchmark_name}: {e}")
        return None


def extract_metrics(benchmark_name: str) -> Dict[str, Any]:
    """Extract all metrics for a single benchmark."""
    
    # Load data
    phase1_data = load_phase1_data(benchmark_name)
    phase3_results = load_phase3_results(benchmark_name)
    
    if phase1_data is None or phase3_results is None:
        return None
    
    # Extract metrics
    metrics = {}
    
    # Problem name
    metrics['Prob.'] = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name)
    
    # Sols: Number of positive examples (always 5 in our experiments)
    metrics['Sols'] = phase3_results.get('phase1', {}).get('E_plus_size', 5)
    
    # StartC: Number of candidate constraints from Phase 1
    # This is the size of B_globals (CG)
    StartC = len(phase1_data.get('CG', []))
    metrics['StartC'] = StartC
    
    # CT: Number of AllDifferent constraints in target model
    metrics['CT'] = TARGET_CONSTRAINTS.get(benchmark_name, 'N/A')
    
    # Bias: Size of generated bias (B_fixed from Phase 1)
    metrics['Bias'] = phase3_results.get('phase1', {}).get('B_fixed_size', 0)
    
    # ViolQ: Violation queries in Phase 2
    metrics['ViolQ'] = phase3_results.get('phase2', {}).get('queries', 0)
    
    # InvC: Invalidated constraints (StartC - validated)
    validated_count = phase3_results.get('phase2', {}).get('validated_globals', 0)
    metrics['InvC'] = StartC - validated_count
    
    # MQuQ: Active learning queries in Phase 3
    metrics['MQuQ'] = phase3_results.get('phase3', {}).get('queries', 0)
    
    # TQ: Total queries (ViolQ + MQuQ)
    metrics['TQ'] = metrics['ViolQ'] + metrics['MQuQ']
    
    # VT(s): Duration of violation phase (Phase 2)
    metrics['VT(s)'] = round(phase3_results.get('phase2', {}).get('time', 0), 2)
    
    # MQuT(s): Duration of active learning phase (Phase 3)
    metrics['MQuT(s)'] = round(phase3_results.get('phase3', {}).get('time', 0), 2)
    
    # TT(s): Overall runtime (Phase 2 + Phase 3)
    metrics['TT(s)'] = round(phase3_results.get('total', {}).get('time', 0), 2)
    
    # Baseline metrics (if available)
    # Note: These need to be run separately as baseline experiments
    # For now, mark as N/A
    metrics['ALQ'] = 'N/A'  # MQuAcq-2 baseline queries
    metrics['PAQ'] = 'N/A'  # Passive+Active baseline queries
    metrics['ALT(s)'] = 'N/A'  # MQuAcq-2 baseline time
    metrics['PAT(s)'] = 'N/A'  # Passive+Active baseline time
    
    # Additional evaluation metrics (optional, for reference)
    eval_data = phase3_results.get('evaluation', {})
    constraint_level = eval_data.get('constraint_level', {})
    solution_level = eval_data.get('solution_level', {})
    
    metrics['_precision'] = round(constraint_level.get('precision', 0) * 100, 2)
    metrics['_recall'] = round(constraint_level.get('recall', 0) * 100, 2)
    metrics['_s_precision'] = round(solution_level.get('s_precision', 0) * 100, 2)
    metrics['_s_recall'] = round(solution_level.get('s_recall', 0) * 100, 2)
    
    return metrics


def generate_table(benchmarks: List[str]) -> pd.DataFrame:
    """Generate results table for all benchmarks."""
    
    all_metrics = []
    
    for benchmark in benchmarks:
        print(f"Processing {benchmark}...")
        metrics = extract_metrics(benchmark)
        
        if metrics is not None:
            all_metrics.append(metrics)
        else:
            print(f"  [SKIPPED] Incomplete data for {benchmark}")
    
    if not all_metrics:
        print("\n[ERROR] No valid data found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns to match paper format
    main_columns = [
        'Prob.', 'Sols', 'StartC', 'InvC', 'CT', 'Bias',
        'ViolQ', 'MQuQ', 'TQ', 'ALQ', 'PAQ',
        'VT(s)', 'MQuT(s)', 'TT(s)', 'ALT(s)', 'PAT(s)'
    ]
    
    # Add evaluation columns (optional)
    eval_columns = ['_precision', '_recall', '_s_precision', '_s_recall']
    
    # Select available columns
    available_main = [col for col in main_columns if col in df.columns]
    available_eval = [col for col in eval_columns if col in df.columns]
    
    df = df[available_main + available_eval]
    
    return df


def print_latex_table(df: pd.DataFrame):
    """Print table in LaTeX format."""
    print("\n" + "="*80)
    print("LaTeX Format")
    print("="*80)
    
    latex = df.to_latex(
        index=False,
        float_format="%.2f",
        na_rep='--',
        caption='HCAR Pipeline Results',
        label='tab:hcar_results',
        escape=False
    )
    print(latex)


def print_markdown_table(df: pd.DataFrame):
    """Print table in Markdown format."""
    print("\n" + "="*80)
    print("Markdown Format")
    print("="*80)
    
    markdown = df.to_markdown(index=False)
    print(markdown)


def save_results(df: pd.DataFrame, output_dir: str = "results"):
    """Save results to CSV, JSON, and text files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV
    csv_path = os.path.join(output_dir, "hcar_pipeline_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] CSV: {csv_path}")
    
    # JSON
    json_path = os.path.join(output_dir, "hcar_pipeline_results.json")
    df.to_json(json_path, orient='records', indent=2)
    print(f"[SAVED] JSON: {json_path}")
    
    # Text (formatted table)
    txt_path = os.path.join(output_dir, "hcar_pipeline_results.txt")
    with open(txt_path, 'w') as f:
        f.write("HCAR Pipeline Results\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("Legend:\n")
        f.write("  Sols: Number of given solutions (positive examples)\n")
        f.write("  StartC: Number of candidate constraints from passive learning\n")
        f.write("  InvC: Number of constraints invalidated by refinement\n")
        f.write("  CT: Number of AllDifferent constraints in target model\n")
        f.write("  Bias: Size of generated bias\n")
        f.write("  ViolQ: Violation queries (Phase 2)\n")
        f.write("  MQuQ: Active learning queries (Phase 3)\n")
        f.write("  TQ: Total queries (ViolQ + MQuQ)\n")
        f.write("  ALQ: Queries for purely Active Learning baseline\n")
        f.write("  PAQ: Queries for Passive+Active baseline\n")
        f.write("  VT(s): Duration of violation phase\n")
        f.write("  MQuT(s): Duration of active learning phase\n")
        f.write("  TT(s): Overall runtime\n")
        f.write("  ALT(s): Runtime for Active Learning baseline\n")
        f.write("  PAT(s): Runtime for Passive+Active baseline\n")
    print(f"[SAVED] Text: {txt_path}")
    
    # LaTeX
    latex_path = os.path.join(output_dir, "hcar_pipeline_results.tex")
    with open(latex_path, 'w') as f:
        latex = df.to_latex(
            index=False,
            float_format="%.2f",
            na_rep='--',
            caption='HCAR Pipeline Results',
            label='tab:hcar_results',
            escape=False
        )
        f.write(latex)
    print(f"[SAVED] LaTeX: {latex_path}")


def main():
    """Main function."""
    print("="*80)
    print("HCAR Pipeline Results Table Generator")
    print("="*80)
    
    # List of benchmarks to process
    benchmarks = [
        'sudoku',
        'sudoku_gt',
        'jsudoku',
        'latin_square',
        'graph_coloring_register',
        'examtt_v1',
        'examtt_v2'
    ]
    
    # Generate table
    print("\nGenerating results table...\n")
    df = generate_table(benchmarks)
    
    if df is None:
        print("\n[ERROR] Failed to generate table!")
        sys.exit(1)
    
    # Display table
    print("\n" + "="*80)
    print("HCAR Pipeline Results")
    print("="*80)
    print(df.to_string(index=False))
    
    # Print in different formats
    print_markdown_table(df)
    print_latex_table(df)
    
    # Save results
    save_results(df)
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    # Only compute stats for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 0:
        print(f"Number of benchmarks: {len(df)}")
        print(f"Average queries (TQ): {df['TQ'].mean():.1f}")
        print(f"Average time (TT): {df['TT(s)'].mean():.1f}s")
        print(f"Average invalidated constraints: {df['InvC'].mean():.1f}")
        print(f"Total queries across all benchmarks: {df['TQ'].sum()}")
        print(f"Total time across all benchmarks: {df['TT(s)'].sum():.1f}s")
    
    print("\n[SUCCESS] Results table generated successfully!")
    print("\nNote: Baseline metrics (ALQ, PAQ, ALT, PAT) are marked as 'N/A'.")
    print("      To populate these, run separate baseline experiments with:")
    print("      - Pure MQuAcq-2 (no passive learning, no refinement)")
    print("      - Passive+Active (passive learning + MQuAcq-2, no refinement)")


if __name__ == "__main__":
    main()

