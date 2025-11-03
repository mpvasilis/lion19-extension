"""
Comparison script for COP vs LION19 Phase 2 approaches.

This script loads results from both COP and LION19 experiments and generates
side-by-side comparisons in various formats (text, JSON, LaTeX tables).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def load_json_results(json_path):
    """Load results from a JSON file."""
    if not os.path.exists(json_path):
        print(f"Warning: File not found: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_comparison_table(cop_results, lion19_results, output_dir):
    """
    Create comparison tables for COP vs LION19 approaches.
    
    Args:
        cop_results: Results from COP approach
        lion19_results: Results from LION19 approach
        output_dir: Directory to save output files
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize results by benchmark
    cop_by_benchmark = {}
    if cop_results:
        for result in cop_results:
            cop_by_benchmark[result['benchmark']] = result
    
    lion19_by_benchmark = {}
    if lion19_results:
        for result in lion19_results:
            lion19_by_benchmark[result['benchmark']] = result
    
    # Get all benchmarks
    all_benchmarks = sorted(set(list(cop_by_benchmark.keys()) + list(lion19_by_benchmark.keys())))
    
    if not all_benchmarks:
        print("No benchmarks found in either result set!")
        return
    
    # Generate text report
    txt_file = os.path.join(output_dir, 'cop_vs_lion19_comparison.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("COP vs LION19 Phase 2 Comparison\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total benchmarks: {len(all_benchmarks)}\n")
        f.write("="*100 + "\n\n")
        
        # Summary statistics
        f.write("OVERALL SUMMARY\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Metric':<30} {'COP':<20} {'LION19':<20} {'Winner':<20}\n")
        f.write("-"*100 + "\n")
        
        # Calculate averages
        cop_avg_queries = sum(r.get('queries', 0) for r in cop_by_benchmark.values() if r.get('queries')) / len(cop_by_benchmark) if cop_by_benchmark else 0
        lion19_avg_queries = sum(r.get('queries', 0) for r in lion19_by_benchmark.values() if r.get('queries')) / len(lion19_by_benchmark) if lion19_by_benchmark else 0
        
        cop_avg_time = sum(r.get('time', 0) for r in cop_by_benchmark.values() if r.get('time')) / len(cop_by_benchmark) if cop_by_benchmark else 0
        lion19_avg_time = sum(r.get('time', 0) for r in lion19_by_benchmark.values() if r.get('time')) / len(lion19_by_benchmark) if lion19_by_benchmark else 0
        
        cop_avg_precision = sum(r.get('precision', 0) for r in cop_by_benchmark.values() if r.get('precision') is not None) / len([r for r in cop_by_benchmark.values() if r.get('precision') is not None]) if cop_by_benchmark else 0
        lion19_avg_precision = sum(r.get('precision', 0) for r in lion19_by_benchmark.values() if r.get('precision') is not None) / len([r for r in lion19_by_benchmark.values() if r.get('precision') is not None]) if lion19_by_benchmark else 0
        
        cop_avg_recall = sum(r.get('recall', 0) for r in cop_by_benchmark.values() if r.get('recall') is not None) / len([r for r in cop_by_benchmark.values() if r.get('recall') is not None]) if cop_by_benchmark else 0
        lion19_avg_recall = sum(r.get('recall', 0) for r in lion19_by_benchmark.values() if r.get('recall') is not None) / len([r for r in lion19_by_benchmark.values() if r.get('recall') is not None]) if lion19_by_benchmark else 0
        
        cop_success = sum(1 for r in cop_by_benchmark.values() if r.get('status') == 'SUCCESS')
        lion19_success = sum(1 for r in lion19_by_benchmark.values() if r.get('status') == 'SUCCESS')
        
        f.write(f"{'Average Queries':<30} {cop_avg_queries:>18.1f}  {lion19_avg_queries:>18.1f}  {'COP' if cop_avg_queries < lion19_avg_queries else 'LION19' if lion19_avg_queries < cop_avg_queries else 'Tie':<20}\n")
        f.write(f"{'Average Time (s)':<30} {cop_avg_time:>18.2f}  {lion19_avg_time:>18.2f}  {'COP' if cop_avg_time < lion19_avg_time else 'LION19' if lion19_avg_time < cop_avg_time else 'Tie':<20}\n")
        f.write(f"{'Average Precision':<30} {cop_avg_precision:>17.2%}  {lion19_avg_precision:>17.2%}  {'COP' if cop_avg_precision > lion19_avg_precision else 'LION19' if lion19_avg_precision > cop_avg_precision else 'Tie':<20}\n")
        f.write(f"{'Average Recall':<30} {cop_avg_recall:>17.2%}  {lion19_avg_recall:>17.2%}  {'COP' if cop_avg_recall > lion19_avg_recall else 'LION19' if lion19_avg_recall > cop_avg_recall else 'Tie':<20}\n")
        f.write(f"{'Perfect Learning Count':<30} {cop_success:>18}  {lion19_success:>18}  {'COP' if cop_success > lion19_success else 'LION19' if lion19_success > cop_success else 'Tie':<20}\n")
        f.write("\n")
        
        # Per-benchmark comparison
        f.write("\nPER-BENCHMARK COMPARISON\n")
        f.write("-"*100 + "\n")
        
        for benchmark in all_benchmarks:
            f.write(f"\n{'='*100}\n")
            f.write(f"Benchmark: {benchmark}\n")
            f.write(f"{'='*100}\n")
            
            cop_result = cop_by_benchmark.get(benchmark, {})
            lion19_result = lion19_by_benchmark.get(benchmark, {})
            
            f.write(f"\n{'Metric':<25} {'COP':<20} {'LION19':<20} {'Better':<15}\n")
            f.write("-"*100 + "\n")
            
            # Status
            cop_status = cop_result.get('status', 'N/A')
            lion19_status = lion19_result.get('status', 'N/A')
            f.write(f"{'Status':<25} {cop_status:<20} {lion19_status:<20}\n")
            
            # Queries
            cop_queries = cop_result.get('queries', None)
            lion19_queries = lion19_result.get('queries', None)
            cop_q_str = str(cop_queries) if cop_queries is not None else 'N/A'
            lion19_q_str = str(lion19_queries) if lion19_queries is not None else 'N/A'
            better = ''
            if cop_queries is not None and lion19_queries is not None:
                if cop_queries < lion19_queries:
                    better = 'COP ✓'
                elif lion19_queries < cop_queries:
                    better = 'LION19 ✓'
                else:
                    better = 'Tie'
            f.write(f"{'Queries':<25} {cop_q_str:<20} {lion19_q_str:<20} {better:<15}\n")
            
            # Time
            cop_time = cop_result.get('time', None)
            lion19_time = lion19_result.get('time', None)
            cop_t_str = f"{cop_time:.2f}s" if cop_time is not None else 'N/A'
            lion19_t_str = f"{lion19_time:.2f}s" if lion19_time is not None else 'N/A'
            better = ''
            if cop_time is not None and lion19_time is not None:
                if cop_time < lion19_time:
                    better = 'COP ✓'
                elif lion19_time < cop_time:
                    better = 'LION19 ✓'
                else:
                    better = 'Tie'
            f.write(f"{'Time':<25} {cop_t_str:<20} {lion19_t_str:<20} {better:<15}\n")
            
            # Precision
            cop_prec = cop_result.get('precision', None)
            lion19_prec = lion19_result.get('precision', None)
            cop_p_str = f"{cop_prec:.2%}" if cop_prec is not None else 'N/A'
            lion19_p_str = f"{lion19_prec:.2%}" if lion19_prec is not None else 'N/A'
            better = ''
            if cop_prec is not None and lion19_prec is not None:
                if cop_prec > lion19_prec:
                    better = 'COP ✓'
                elif lion19_prec > cop_prec:
                    better = 'LION19 ✓'
                else:
                    better = 'Tie'
            f.write(f"{'Precision':<25} {cop_p_str:<20} {lion19_p_str:<20} {better:<15}\n")
            
            # Recall
            cop_rec = cop_result.get('recall', None)
            lion19_rec = lion19_result.get('recall', None)
            cop_r_str = f"{cop_rec:.2%}" if cop_rec is not None else 'N/A'
            lion19_r_str = f"{lion19_rec:.2%}" if lion19_rec is not None else 'N/A'
            better = ''
            if cop_rec is not None and lion19_rec is not None:
                if cop_rec > lion19_rec:
                    better = 'COP ✓'
                elif lion19_rec > cop_rec:
                    better = 'LION19 ✓'
                else:
                    better = 'Tie'
            f.write(f"{'Recall':<25} {cop_r_str:<20} {lion19_r_str:<20} {better:<15}\n")
            
            # Correct/Missing/Spurious
            f.write(f"\n{'Constraint Accuracy':<25}\n")
            f.write(f"  Correct: {cop_result.get('correct', 'N/A'):<17} {lion19_result.get('correct', 'N/A'):<20}\n")
            f.write(f"  Missing: {cop_result.get('missing', 'N/A'):<17} {lion19_result.get('missing', 'N/A'):<20}\n")
            f.write(f"  Spurious: {cop_result.get('spurious', 'N/A'):<16} {lion19_result.get('spurious', 'N/A'):<20}\n")
    
    print(f"\nText comparison saved to: {txt_file}")
    
    # Generate LaTeX table
    latex_file = os.path.join(output_dir, 'cop_vs_lion19_comparison.tex')
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("% COP vs LION19 Phase 2 Comparison Table\n")
        f.write("% Generated: " + datetime.now().isoformat() + "\n\n")
        
        f.write("\\begin{table}[ht!]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of COP and LION19 Approaches for Phase 2 Query-Driven Refinement}\n")
        f.write("\\label{tab:cop_vs_lion19}\n")
        f.write("\\begin{tabular}{l|rr|rr|rr}\n")
        f.write("\\hline\n")
        f.write("& \\multicolumn{2}{c|}{\\textbf{Queries}} & \\multicolumn{2}{c|}{\\textbf{Time (s)}} & \\multicolumn{2}{c}{\\textbf{Recall}} \\\\\n")
        f.write("\\textbf{Benchmark} & COP & LION19 & COP & LION19 & COP & LION19 \\\\\n")
        f.write("\\hline\n")
        
        for benchmark in all_benchmarks:
            cop_result = cop_by_benchmark.get(benchmark, {})
            lion19_result = lion19_by_benchmark.get(benchmark, {})
            
            cop_q = cop_result.get('queries', '')
            lion19_q = lion19_result.get('queries', '')
            cop_t = f"{cop_result.get('time', 0):.1f}" if cop_result.get('time') else ''
            lion19_t = f"{lion19_result.get('time', 0):.1f}" if lion19_result.get('time') else ''
            cop_r = f"{cop_result.get('recall', 0):.2f}" if cop_result.get('recall') is not None else ''
            lion19_r = f"{lion19_result.get('recall', 0):.2f}" if lion19_result.get('recall') is not None else ''
            
            # Format benchmark name for LaTeX
            benchmark_latex = benchmark.replace('_', '\\_')
            
            f.write(f"{benchmark_latex} & {cop_q} & {lion19_q} & {cop_t} & {lion19_t} & {cop_r} & {lion19_r} \\\\\n")
        
        f.write("\\hline\n")
        
        # Add average row
        f.write(f"\\textbf{{Average}} & {cop_avg_queries:.1f} & {lion19_avg_queries:.1f} & {cop_avg_time:.1f} & {lion19_avg_time:.1f} & {cop_avg_recall:.2f} & {lion19_avg_recall:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX comparison saved to: {latex_file}")
    
    # Generate JSON comparison
    comparison_data = {
        'generated': datetime.now().isoformat(),
        'summary': {
            'cop': {
                'avg_queries': cop_avg_queries,
                'avg_time': cop_avg_time,
                'avg_precision': cop_avg_precision,
                'avg_recall': cop_avg_recall,
                'success_count': cop_success,
                'total_benchmarks': len(cop_by_benchmark)
            },
            'lion19': {
                'avg_queries': lion19_avg_queries,
                'avg_time': lion19_avg_time,
                'avg_precision': lion19_avg_precision,
                'avg_recall': lion19_avg_recall,
                'success_count': lion19_success,
                'total_benchmarks': len(lion19_by_benchmark)
            }
        },
        'benchmarks': {}
    }
    
    for benchmark in all_benchmarks:
        comparison_data['benchmarks'][benchmark] = {
            'cop': cop_by_benchmark.get(benchmark, {}),
            'lion19': lion19_by_benchmark.get(benchmark, {})
        }
    
    json_file = os.path.join(output_dir, 'cop_vs_lion19_comparison.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"JSON comparison saved to: {json_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<30} {'COP':<20} {'LION19':<20}")
    print("-"*80)
    print(f"{'Average Queries':<30} {cop_avg_queries:>18.1f}  {lion19_avg_queries:>18.1f}")
    print(f"{'Average Time (s)':<30} {cop_avg_time:>18.2f}  {lion19_avg_time:>18.2f}")
    print(f"{'Average Precision':<30} {cop_avg_precision:>17.2%}  {lion19_avg_precision:>17.2%}")
    print(f"{'Average Recall':<30} {cop_avg_recall:>17.2%}  {lion19_avg_recall:>17.2%}")
    print(f"{'Perfect Learning Count':<30} {cop_success:>18}  {lion19_success:>18}")
    print("="*80)


def main():
    """Main function to compare COP and LION19 results."""
    
    # Define paths to result files
    cop_json = 'phase2_output/phase2_results.json'
    lion19_json = 'phase2_lion19_output/phase2_lion19_results.json'
    
    print("="*80)
    print("COP vs LION19 Comparison Tool")
    print("="*80)
    print(f"\nLoading results...")
    print(f"  COP results: {cop_json}")
    print(f"  LION19 results: {lion19_json}")
    
    # Load results
    cop_results = load_json_results(cop_json)
    lion19_results = load_json_results(lion19_json)
    
    if cop_results is None and lion19_results is None:
        print("\nError: No results found for either approach!")
        print("Please run the experiments first:")
        print("  1. python run_phase2_experiments.py  (for COP)")
        print("  2. python run_phase2_lion19_experiments.py  (for LION19)")
        return 1
    
    if cop_results is None:
        print("\nWarning: COP results not found. Showing LION19 results only.")
        cop_results = []
    
    if lion19_results is None:
        print("\nWarning: LION19 results not found. Showing COP results only.")
        lion19_results = []
    
    # Create comparison
    output_dir = 'phase2_comparison'
    print(f"\nGenerating comparison reports in: {output_dir}")
    
    create_comparison_table(cop_results, lion19_results, output_dir)
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

