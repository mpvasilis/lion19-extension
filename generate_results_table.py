

import os
import sys
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

BENCHMARK_DISPLAY_NAMES = {
    'sudoku': 'Sudoku',
    'sudoku_gt': 'Sudoku-GT',
    'jsudoku': 'JSudoku',
    'latin_square': 'Latin Square',
    'graph_coloring_register': 'Graph Coloring',
    'examtt_v1': 'ExamTT-V1',
    'examtt_v2': 'ExamTT-V2'
}

TARGET_CONSTRAINTS = {
    'sudoku': 27,  
    'sudoku_gt': 37,  
    'jsudoku': 31,  
    'latin_square': 18,  
    'graph_coloring_register': 5,  
    'examtt_v1': 7,  
    'examtt_v2': 9   
}


def load_phase1_data(benchmark_name: str) -> Dict[str, Any]:
    
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
    

    phase1_data = load_phase1_data(benchmark_name)
    phase3_results = load_phase3_results(benchmark_name)
    
    if phase1_data is None or phase3_results is None:
        return None

    metrics = {}

    metrics['Prob.'] = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name)

    metrics['Sols'] = phase3_results.get('phase1', {}).get('E_plus_size', 5)


    StartC = len(phase1_data.get('CG', []))
    metrics['StartC'] = StartC

    metrics['CT'] = TARGET_CONSTRAINTS.get(benchmark_name, 'N/A')

    metrics['Bias'] = phase3_results.get('phase1', {}).get('B_fixed_size', 0)

    metrics['ViolQ'] = phase3_results.get('phase2', {}).get('queries', 0)

    validated_count = phase3_results.get('phase2', {}).get('validated_globals', 0)
    metrics['InvC'] = StartC - validated_count

    metrics['MQuQ'] = phase3_results.get('phase3', {}).get('queries', 0)

    metrics['TQ'] = metrics['ViolQ'] + metrics['MQuQ']

    metrics['VT(s)'] = round(phase3_results.get('phase2', {}).get('time', 0), 2)

    metrics['MQuT(s)'] = round(phase3_results.get('phase3', {}).get('time', 0), 2)

    metrics['TT(s)'] = round(phase3_results.get('total', {}).get('time', 0), 2)



    metrics['ALQ'] = 'N/A'  
    metrics['PAQ'] = 'N/A'  
    metrics['ALT(s)'] = 'N/A'  
    metrics['PAT(s)'] = 'N/A'  

    eval_data = phase3_results.get('evaluation', {})
    constraint_level = eval_data.get('constraint_level', {})
    solution_level = eval_data.get('solution_level', {})
    
    metrics['_precision'] = round(constraint_level.get('precision', 0) * 100, 2)
    metrics['_recall'] = round(constraint_level.get('recall', 0) * 100, 2)
    metrics['_s_precision'] = round(solution_level.get('s_precision', 0) * 100, 2)
    metrics['_s_recall'] = round(solution_level.get('s_recall', 0) * 100, 2)
    
    return metrics


def generate_table(benchmarks: List[str]) -> pd.DataFrame:
    
    
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

    df = pd.DataFrame(all_metrics)

    main_columns = [
        'Prob.', 'Sols', 'StartC', 'InvC', 'CT', 'Bias',
        'ViolQ', 'MQuQ', 'TQ', 'ALQ', 'PAQ',
        'VT(s)', 'MQuT(s)', 'TT(s)', 'ALT(s)', 'PAT(s)'
    ]

    eval_columns = ['_precision', '_recall', '_s_precision', '_s_recall']

    available_main = [col for col in main_columns if col in df.columns]
    available_eval = [col for col in eval_columns if col in df.columns]
    
    df = df[available_main + available_eval]
    
    return df


def print_latex_table(df: pd.DataFrame):
    
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
    
    print("\n" + "="*80)
    print("Markdown Format")
    print("="*80)
    
    markdown = df.to_markdown(index=False)
    print(markdown)


def save_results(df: pd.DataFrame, output_dir: str = "results"):
    
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "hcar_pipeline_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] CSV: {csv_path}")

    json_path = os.path.join(output_dir, "hcar_pipeline_results.json")
    df.to_json(json_path, orient='records', indent=2)
    print(f"[SAVED] JSON: {json_path}")

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
    
    print("="*80)
    print("HCAR Pipeline Results Table Generator")
    print("="*80)

    benchmarks = [
        'sudoku',
        'sudoku_gt',
        'jsudoku',
        'latin_square',
        'graph_coloring_register',
        'examtt_v1',
        'examtt_v2',
        'nurse'
    ]

    print("\nGenerating results table...\n")
    df = generate_table(benchmarks)
    
    if df is None:
        print("\n[ERROR] Failed to generate table!")
        sys.exit(1)

    print("\n" + "="*80)
    print("HCAR Pipeline Results")
    print("="*80)
    print(df.to_string(index=False))

    print_markdown_table(df)
    print_latex_table(df)

    save_results(df)

    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 0:
        print(f"Number of benchmarks: {len(df)}")
        print(f"Average queries (TQ): {df['TQ'].mean():.1f}")
        print(f"Average time (TT): {df['TT(s)'].mean():.1f}s")
        print(f"Average invalidated constraints: {df['InvC'].mean():.1f}")
        print(f"Total queries across all benchmarks: {df['TQ'].sum()}")
        print(f"Total time across all benchmarks: {df['TT(s)'].sum():.1f}s")
    

if __name__ == "__main__":
    main()

