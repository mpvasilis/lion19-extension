

import os
import sys
import json
import time
import subprocess
import statistics
from datetime import datetime
from pathlib import Path


def run_phase2_for_benchmark(benchmark_name, phase1_pickle_path, output_dir, log_file):
    
    print(f"\n{'='*80}")
    print(f"Running Phase 2 for: {benchmark_name}")
    print(f"{'='*80}")
    print(f"Phase 1 pickle: {phase1_pickle_path}")
    print(f"Log file: {log_file}")
    
    start_time = time.time()

    cmd = [
        'python',
        'main_alldiff_cop.py',
        '--experiment', benchmark_name,
        '--phase1_pickle', phase1_pickle_path,
    ]
    
    print(f"Command: {' '.join(cmd)}\n")

    try:
        with open(log_file, 'w', encoding='utf-8') as f:

            f.write("="*80 + "\n")
            f.write(f"PHASE 2 EXPERIMENT LOG\n")
            f.write(f"Benchmark: {benchmark_name}\n")
            f.write(f"Phase 1 pickle: {phase1_pickle_path}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
            f.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                    output_lines.append(line.strip())
            
            process.wait()
            return_code = process.returncode
        
        end_time = time.time()
        duration = end_time - start_time

        results = parse_phase2_output(output_lines, benchmark_name)
        results['return_code'] = return_code
        results['duration'] = duration
        results['log_file'] = log_file

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Status: {'SUCCESS' if return_code == 0 else 'FAILED'}\n")
            f.write("="*80 + "\n")
        
        if return_code == 0:
            print(f"\n[SUCCESS] {benchmark_name} completed in {duration:.2f}s")
            return results
        else:
            print(f"\n[FAILED] {benchmark_name} returned error code {return_code}")
            results['status'] = 'FAILED'
            return results
    
    except Exception as e:
        print(f"\n[ERROR] Exception running {benchmark_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'benchmark': benchmark_name,
            'status': 'ERROR',
            'error': str(e),
            'duration': duration,
            'log_file': log_file
        }


def parse_phase2_output(output_lines, benchmark_name):
    
    results = {
        'benchmark': benchmark_name,
        'status': 'UNKNOWN',
        'queries': None,
        'time': None,
        'validated': None,
        'rejected': None,
        'target_count': None,
        'learned_count': None,
        'correct': None,
        'missing': None,
        'spurious': None,
        'precision': None,
        'recall': None
    }

    for line in output_lines:
        if 'Total queries:' in line:
            try:
                results['queries'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif 'Total time:' in line:
            try:
                time_str = line.split(':')[1].strip().replace('s', '')
                results['time'] = float(time_str)
            except:
                pass
        
        elif 'Validated constraints:' in line:
            try:
                results['validated'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif 'Rejected constraints:' in line:
            try:
                results['rejected'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif 'Target AllDifferent constraints:' in line:
            try:
                results['target_count'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif 'Learned AllDifferent constraints:' in line:
            try:
                results['learned_count'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif line.startswith('Correct:'):
            try:
                results['correct'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif line.startswith('Missing:'):
            try:
                results['missing'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif line.startswith('Spurious:'):
            try:
                results['spurious'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif '[SUCCESS] Perfect learning!' in line:
            results['status'] = 'SUCCESS'

    if results['learned_count'] is not None and results['learned_count'] > 0:
        if results['correct'] is not None:
            results['precision'] = results['correct'] / results['learned_count']
    
    if results['target_count'] is not None and results['target_count'] > 0:
        if results['correct'] is not None:
            results['recall'] = results['correct'] / results['target_count']

    if results['status'] == 'UNKNOWN':
        if results['correct'] == results['target_count'] and results['spurious'] == 0:
            results['status'] = 'SUCCESS'
        elif results['queries'] is not None:
            results['status'] = 'COMPLETED'
        else:
            results['status'] = 'FAILED'
    
    return results


def aggregate_benchmark_results(benchmark_runs, benchmark_name, num_runs):
    """Aggregate results across multiple runs for a benchmark."""

    if not benchmark_runs or all(r.get('status') in ['ERROR', 'FAILED'] for r in benchmark_runs):
        # Return the first error result if all runs failed
        agg_result = benchmark_runs[0].copy()
        agg_result['runs_completed'] = 0
        agg_result['runs_successful'] = 0
        return agg_result

    # Filter out successful runs for aggregation
    successful_runs = [r for r in benchmark_runs if r.get('status') in ['SUCCESS', 'COMPLETED']]

    if not successful_runs:
        # No successful runs
        agg_result = benchmark_runs[0].copy()
        agg_result['runs_completed'] = 0
        agg_result['runs_successful'] = 0
        return agg_result

    # Aggregate numeric metrics
    agg_result = {
        'benchmark': benchmark_name,
        'runs_completed': len(successful_runs),
        'runs_total': num_runs,
        'runs_successful': len([r for r in successful_runs if r.get('status') == 'SUCCESS']),
        'status': 'AGGREGATED'
    }

    # Numeric fields to aggregate
    numeric_fields = ['queries', 'time', 'duration', 'validated', 'rejected',
                     'target_count', 'learned_count', 'correct', 'missing', 'spurious',
                     'precision', 'recall']

    for field in numeric_fields:
        values = [r[field] for r in successful_runs if r.get(field) is not None]
        if values:
            mean_val = statistics.mean(values)
            if len(values) > 1:
                std_val = statistics.stdev(values)
            else:
                std_val = 0.0

            agg_result[f'{field}_mean'] = mean_val
            agg_result[f'{field}_std'] = std_val
            agg_result[field] = f"{mean_val:.2f}±{std_val:.2f}"
        else:
            agg_result[field] = 'N/A'

    # Most common status (prioritize SUCCESS over COMPLETED)
    statuses = [r.get('status', 'UNKNOWN') for r in successful_runs]
    if 'SUCCESS' in statuses:
        agg_result['most_common_status'] = 'SUCCESS'
    elif 'COMPLETED' in statuses:
        agg_result['most_common_status'] = 'COMPLETED'
    else:
        agg_result['most_common_status'] = statuses[0] if statuses else 'UNKNOWN'

    return agg_result


def create_summary_report(all_results, aggregated_results, output_dir, num_runs):

    # Create aggregated results
    benchmark_aggregates = {}
    for bench_name, bench_runs in aggregated_results.items():
        benchmark_aggregates[bench_name] = aggregate_benchmark_results(bench_runs, bench_name, num_runs)

    # Aggregated summary file
    summary_file = os.path.join(output_dir, 'phase2_summary_aggregated.txt')
    detailed_file = os.path.join(output_dir, 'phase2_summary_detailed.txt')
    json_file = os.path.join(output_dir, 'phase2_results_all_runs.json')
    agg_json_file = os.path.join(output_dir, 'phase2_results_aggregated.json')

    # Create aggregated summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"PHASE 2 EXPERIMENTS - AGGREGATED SUMMARY (over {num_runs} runs)\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total benchmarks: {len(benchmark_aggregates)}\n")
        f.write("="*100 + "\n\n")

        success_count = sum(1 for r in benchmark_aggregates.values() if r['runs_successful'] > 0)
        partial_success_count = sum(1 for r in benchmark_aggregates.values() if r['runs_completed'] > 0 and r['runs_successful'] == 0)
        failed_count = sum(1 for r in benchmark_aggregates.values() if r['runs_completed'] == 0)

        f.write("OVERALL STATISTICS\n")
        f.write("-"*100 + "\n")
        f.write(f"Benchmarks with successful runs: {success_count}\n")
        f.write(f"Benchmarks with partial success: {partial_success_count}\n")
        f.write(f"Benchmarks with all failed runs: {failed_count}\n")
        f.write("\n")

        f.write("AGGREGATED RESULTS PER BENCHMARK\n")
        f.write("-"*100 + "\n\n")

        for bench_name, agg_result in benchmark_aggregates.items():
            f.write(f"Benchmark: {bench_name}\n")
            f.write(f"  Runs completed: {agg_result['runs_completed']}/{agg_result['runs_total']}\n")
            f.write(f"  Successful runs: {agg_result['runs_successful']}\n")
            f.write(f"  Most common status: {agg_result.get('most_common_status', 'N/A')}\n")

            if agg_result.get('queries') != 'N/A':
                f.write(f"  Queries: {agg_result['queries']}\n")
            if agg_result.get('time') != 'N/A':
                f.write(f"  Time: {agg_result['time']}s\n")
            if agg_result.get('duration') != 'N/A':
                f.write(f"  Duration: {agg_result['duration']}s\n")

            f.write(f"\n  Constraint Learning:\n")
            if agg_result.get('target_count') != 'N/A':
                f.write(f"    Target constraints: {agg_result['target_count']}\n")
            if agg_result.get('learned_count') != 'N/A':
                f.write(f"    Learned constraints: {agg_result['learned_count']}\n")
            if agg_result.get('validated') != 'N/A':
                f.write(f"    Validated: {agg_result['validated']}\n")

            f.write(f"\n  Accuracy:\n")
            if agg_result.get('precision') != 'N/A':
                f.write(f"    Precision: {agg_result['precision']}\n")
            if agg_result.get('recall') != 'N/A':
                f.write(f"    Recall: {agg_result['recall']}\n")

            f.write("\n" + "-"*100 + "\n\n")

        f.write("AGGREGATED SUMMARY TABLE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Benchmark':<20} {'Runs':<8} {'Success':<10} {'Queries':<12} {'Time(s)':<12} {'Precision':<12} {'Recall':<10}\n")
        f.write("-"*100 + "\n")

        for bench_name, agg_result in benchmark_aggregates.items():
            runs_str = f"{agg_result['runs_completed']}/{agg_result['runs_total']}"
            success_str = f"{agg_result['runs_successful']}"
            queries_str = agg_result.get('queries', 'N/A')
            time_str = agg_result.get('time', 'N/A')
            prec_str = agg_result.get('precision', 'N/A')
            rec_str = agg_result.get('recall', 'N/A')

            f.write(f"{bench_name:<20} {runs_str:<8} {success_str:<10} {queries_str:<12} {time_str:<12} {prec_str:<12} {rec_str:<10}\n")

        f.write("="*100 + "\n")

    # Create detailed summary with all individual runs
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2 EXPERIMENTS - DETAILED RESULTS (ALL INDIVIDUAL RUNS)\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total individual runs: {len(all_results)}\n")
        f.write("="*80 + "\n\n")

        for bench_name, bench_runs in aggregated_results.items():
            f.write(f"BENCHMARK: {bench_name}\n")
            f.write("-"*80 + "\n")

            for run_result in bench_runs:
                run_num = run_result.get('run_number', 'N/A')
                f.write(f"  Run {run_num}: {run_result['status']}\n")

                if run_result.get('queries') is not None:
                    f.write(f"    Queries: {run_result['queries']}\n")
                if run_result.get('time') is not None:
                    f.write(f"    Time: {run_result['time']:.2f}s\n")
                if run_result.get('precision') is not None and run_result.get('recall') is not None:
                    f.write(f"    Precision: {run_result['precision']:.2%}, Recall: {run_result['recall']:.2%}\n")

                if run_result.get('error'):
                    f.write(f"    Error: {run_result['error']}\n")

            f.write("\n")

        f.write("DETAILED SUMMARY TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Benchmark':<15} {'Run':<5} {'Status':<12} {'Queries':<10} {'Time(s)':<10} {'Precision':<12} {'Recall':<10}\n")
        f.write("-"*80 + "\n")

        for result in all_results:
            run_str = str(result.get('run_number', 'N/A'))
            queries_str = str(result.get('queries', 'N/A'))
            time_str = f"{result.get('time', 0):.1f}" if result.get('time') else 'N/A'
            prec_str = f"{result.get('precision', 0):.2%}" if result.get('precision') is not None else 'N/A'
            rec_str = f"{result.get('recall', 0):.2%}" if result.get('recall') is not None else 'N/A'

            f.write(f"{result['benchmark']:<15} {run_str:<5} {result['status']:<12} {queries_str:<10} {time_str:<10} {prec_str:<12} {rec_str:<10}\n")

        f.write("="*80 + "\n")

    # Save JSON files
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    with open(agg_json_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_aggregates, f, indent=2)

    print(f"\nSummary reports saved to:")
    print(f"  - Aggregated: {summary_file}")
    print(f"  - Detailed: {detailed_file}")
    print(f"  - All runs JSON: {json_file}")
    print(f"  - Aggregated JSON: {agg_json_file}")

    return summary_file, detailed_file, json_file, agg_json_file


def main(num_runs=10):

    benchmarks = [
        {
            'name': 'sudoku',
            'pickle': 'phase1_output/sudoku_phase1.pkl',
            'description': 'Regular 9x9 Sudoku'
        },
        {
            'name': 'sudoku_gt',
            'pickle': 'phase1_output/sudoku_gt_phase1.pkl',
            'description': 'Sudoku with Greater-Than Constraints'
        },
        {
            'name': 'sudoku_4x4_gt',
            'pickle': 'phase1_output/sudoku_4x4_gt_phase1.pkl',
            'description': '4x4 Sudoku with Greater-Than Constraints'
        },
        {
            'name': 'examtt_v1',
            'pickle': 'phase1_output/examtt_v1_phase1.pkl',
            'description': 'Exam Timetabling Variant 1 (Small)'
        },
        {
            'name': 'examtt_v2',
            'pickle': 'phase1_output/examtt_v2_phase1.pkl',
            'description': 'Exam Timetabling Variant 2 (Large)'
        }
    ]
    
    output_dir = 'phase2_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("PHASE 2 BATCH RUNNER")
    print("="*80)
    print(f"\nRunning Phase 2 for {len(benchmarks)} benchmarks:")
    for i, bench in enumerate(benchmarks, 1):
        print(f"  {i}. {bench['name']}: {bench['description']}")
    print(f"\nRunning each benchmark {num_runs} times")
    print(f"Output directory: {output_dir}")
    print("="*80)

    all_results = []
    aggregated_results = {}

    total_tasks = len(benchmarks) * num_runs
    task_count = 0

    for i, bench in enumerate(benchmarks, 1):
        print(f"\n\n{'='*80}")
        print(f"BENCHMARK {i}/{len(benchmarks)}: {bench['name']}")
        print(f"Description: {bench['description']}")
        print(f"{'='*80}\n")

        if not os.path.exists(bench['pickle']):
            print(f"[ERROR] Phase 1 pickle not found: {bench['pickle']}")
            error_result = {
                'benchmark': bench['name'],
                'status': 'ERROR',
                'error': f"Phase 1 pickle not found: {bench['pickle']}"
            }
            all_results.append(error_result)
            aggregated_results[bench['name']] = [error_result]
            continue

        # Run this benchmark multiple times
        benchmark_runs = []

        for run_num in range(1, num_runs + 1):
            task_count += 1
            print(f"\n{'-'*60}")
            print(f"[RUN {run_num}/{num_runs}] Benchmark {i}/{len(benchmarks)}: {bench['name']} (Task {task_count}/{total_tasks})")
            print(f"{'-'*60}")

            log_file = os.path.join(output_dir, f"{bench['name']}_phase2_run{run_num}.log")

            result = run_phase2_for_benchmark(
                benchmark_name=bench['name'],
                phase1_pickle_path=bench['pickle'],
                output_dir=output_dir,
                log_file=log_file
            )

            # Add run number to result
            result['run_number'] = run_num
            benchmark_runs.append(result)
            all_results.append(result)

        # Store runs for this benchmark for aggregation
        aggregated_results[bench['name']] = benchmark_runs

    print("\n\n" + "="*80)
    print("CREATING SUMMARY REPORTS")
    print("="*80)

    summary_file, detailed_file, json_file, agg_json_file = create_summary_report(all_results, aggregated_results, output_dir, num_runs)

    print("\n" + "="*80)
    print("PHASE 2 BATCH RUNNER - FINAL SUMMARY")
    print("="*80)

    # Create aggregated results for summary
    benchmark_aggregates = {}
    for bench_name, bench_runs in aggregated_results.items():
        benchmark_aggregates[bench_name] = aggregate_benchmark_results(bench_runs, bench_name, num_runs)

    success_count = sum(1 for r in benchmark_aggregates.values() if r['runs_successful'] > 0)
    partial_success_count = sum(1 for r in benchmark_aggregates.values() if r['runs_completed'] > 0 and r['runs_successful'] == 0)
    failed_count = sum(1 for r in benchmark_aggregates.values() if r['runs_completed'] == 0)

    print(f"\nTotal benchmarks: {len(benchmark_aggregates)}")
    print(f"Each benchmark run {num_runs} times")
    print(f"Total individual runs: {len(all_results)}")
    print(f"  [+] Benchmarks with successful runs: {success_count}")
    print(f"  [~] Benchmarks with partial success: {partial_success_count}")
    print(f"  [-] Benchmarks with all failed runs: {failed_count}")

    print("\nAggregated results per benchmark:")
    for bench_name, agg_result in benchmark_aggregates.items():
        success_rate = agg_result['runs_successful'] / num_runs if num_runs > 0 else 0
        status_symbol = "[+]" if agg_result['runs_successful'] > 0 else "[-]" if agg_result['runs_completed'] == 0 else "[~]"
        print(f"  {status_symbol} {bench_name}: {agg_result['runs_successful']}/{num_runs} successful runs")
        if agg_result.get('queries') != 'N/A':
            print(f"      Queries: {agg_result['queries']}")
        if agg_result.get('precision') != 'N/A' and agg_result.get('recall') != 'N/A':
            print(f"      Precision: {agg_result['precision']}, Recall: {agg_result['recall']}")

    print(f"\nOutput files:")
    print(f"  - Aggregated summary: {summary_file}")
    print(f"  - Detailed results: {detailed_file}")
    print(f"  - All runs JSON: {json_file}")
    print(f"  - Aggregated JSON: {agg_json_file}")
    print("="*80)

    # Check if any benchmarks had at least one successful run
    all_failed = all(r['runs_completed'] == 0 for r in benchmark_aggregates.values())
    if all_failed:
        print("\nALL BENCHMARKS FAILED!")
        return 1
    else:
        print(f"\nEXPERIMENTS COMPLETED!")
        return 0


if __name__ == "__main__":
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sys.exit(main(num_runs))

