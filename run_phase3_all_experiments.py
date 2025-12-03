#!/usr/bin/env python3
"""
Run Phase 3 (Active Learning) for all experiments with Phase 2 outputs.
Logs results to individual files and creates a summary.
"""

import os
import sys
import glob
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
def setup_logging(output_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(output_dir, f"phase3_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('phase3_runner')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def find_phase2_pickles(phase2_dir="phase2_output"):
    """Find all Phase 2 pickle files."""
    pattern = os.path.join(phase2_dir, "*_phase2.pkl")
    pickles = glob.glob(pattern)
    return sorted(pickles)


def extract_experiment_name(pickle_path):
    """Extract experiment name from pickle file path."""
    basename = os.path.basename(pickle_path)
    # Remove _phase2.pkl suffix
    return basename.replace("_phase2.pkl", "")


def run_single_experiment(experiment_name, pickle_path, output_dir, algorithm='growacq', 
                          max_queries=1000, timeout=600, logger=None):
    """
    Run Phase 3 for a single experiment.
    
    Returns:
        dict: Results dictionary or None if failed
    """
    from run_phase3 import run_phase3
    
    log_file = os.path.join(output_dir, f"{experiment_name}_phase3.log")
    
    # Redirect stdout/stderr to log file for this experiment
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        with open(log_file, 'w') as f:
            # Create a tee-like object that writes to both file and console
            class TeeWriter:
                def __init__(self, file, stream):
                    self.file = file
                    self.stream = stream
                
                def write(self, data):
                    self.file.write(data)
                    self.stream.write(data)
                    self.file.flush()
                
                def flush(self):
                    self.file.flush()
                    self.stream.flush()
            
            sys.stdout = TeeWriter(f, original_stdout)
            sys.stderr = TeeWriter(f, original_stderr)
            
            start_time = time.time()
            
            results = run_phase3(
                experiment_name=experiment_name,
                phase2_pickle_path=pickle_path,
                max_queries=max_queries,
                timeout=timeout,
                algorithm=algorithm
            )
            
            elapsed_time = time.time() - start_time
            results['wall_clock_time'] = elapsed_time
            
            return results
            
    except Exception as e:
        if logger:
            logger.error(f"Experiment {experiment_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'experiment': experiment_name,
            'status': 'failed',
            'error': str(e)
        }
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Phase 3 (Active Learning) for all experiments'
    )
    parser.add_argument(
        '--phase2_dir', type=str, default='phase2_output',
        help='Directory containing Phase 2 pickle files'
    )
    parser.add_argument(
        '--output_dir', type=str, default='phase3_output',
        help='Directory to store Phase 3 outputs'
    )
    parser.add_argument(
        '--algorithm', type=str, default='growacq',
        choices=['mquacq2', 'growacq'],
        help='Active learning algorithm to use'
    )
    parser.add_argument(
        '--max_queries', type=int, default=1000,
        help='Maximum queries per experiment'
    )
    parser.add_argument(
        '--timeout', type=int, default=600,
        help='Timeout per experiment in seconds'
    )
    parser.add_argument(
        '--experiments', type=str, nargs='*', default=None,
        help='Specific experiments to run (default: all)'
    )
    parser.add_argument(
        '--skip', type=str, nargs='*', default=None,
        help='Experiments to skip'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='List experiments without running them'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger, main_log_file = setup_logging(args.output_dir)
    
    logger.info("=" * 80)
    logger.info("Phase 3 Runner - All Experiments")
    logger.info("=" * 80)
    logger.info(f"Phase 2 directory: {args.phase2_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Max queries: {args.max_queries}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Main log file: {main_log_file}")
    logger.info("=" * 80)
    
    # Find all Phase 2 pickles
    all_pickles = find_phase2_pickles(args.phase2_dir)
    
    if not all_pickles:
        logger.error(f"No Phase 2 pickle files found in {args.phase2_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(all_pickles)} Phase 2 pickle files")
    
    # Filter experiments if specified
    experiments_to_run = []
    for pkl in all_pickles:
        exp_name = extract_experiment_name(pkl)
        
        # Skip if in skip list
        if args.skip and exp_name in args.skip:
            logger.info(f"  Skipping: {exp_name} (in skip list)")
            continue
        
        # Include only specified experiments if provided
        if args.experiments and exp_name not in args.experiments:
            continue
        
        experiments_to_run.append((exp_name, pkl))
    
    logger.info(f"\nExperiments to run: {len(experiments_to_run)}")
    for i, (exp_name, pkl) in enumerate(experiments_to_run, 1):
        logger.info(f"  {i}. {exp_name}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Exiting without running experiments")
        return
    
    # Run experiments
    results_summary = []
    total_start_time = time.time()
    
    for i, (exp_name, pkl_path) in enumerate(experiments_to_run, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{len(experiments_to_run)}] Running: {exp_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Phase 2 pickle: {pkl_path}")
        
        exp_start_time = time.time()
        
        try:
            results = run_single_experiment(
                experiment_name=exp_name,
                pickle_path=pkl_path,
                output_dir=args.output_dir,
                algorithm=args.algorithm,
                max_queries=args.max_queries,
                timeout=args.timeout,
                logger=logger
            )
            
            exp_elapsed = time.time() - exp_start_time
            
            if results and 'status' not in results:
                # Successful run
                summary_entry = {
                    'experiment': exp_name,
                    'status': 'success',
                    'wall_clock_time': exp_elapsed,
                    'phase3_queries': results.get('phase3', {}).get('queries', 0),
                    'phase3_time': results.get('phase3', {}).get('time', 0),
                    'total_queries': results.get('total', {}).get('queries', 0),
                    'total_time': results.get('total', {}).get('time', 0),
                    'precision': results.get('evaluation', {}).get('constraint_level', {}).get('precision', 0),
                    'recall': results.get('evaluation', {}).get('constraint_level', {}).get('recall', 0),
                    'f1': results.get('evaluation', {}).get('constraint_level', {}).get('f1', 0),
                    's_precision': results.get('evaluation', {}).get('solution_level', {}).get('s_precision', 0),
                    's_recall': results.get('evaluation', {}).get('solution_level', {}).get('s_recall', 0),
                    's_f1': results.get('evaluation', {}).get('solution_level', {}).get('s_f1', 0)
                }
                logger.info(f"[SUCCESS] {exp_name} completed in {exp_elapsed:.2f}s")
                logger.info(f"  Phase 3 queries: {summary_entry['phase3_queries']}")
                logger.info(f"  F1 Score: {summary_entry['f1']:.2%}")
            else:
                # Failed run
                summary_entry = {
                    'experiment': exp_name,
                    'status': 'failed',
                    'wall_clock_time': exp_elapsed,
                    'error': results.get('error', 'Unknown error') if results else 'No results returned'
                }
                logger.error(f"[FAILED] {exp_name}: {summary_entry.get('error', 'Unknown error')}")
            
            results_summary.append(summary_entry)
            
        except Exception as e:
            exp_elapsed = time.time() - exp_start_time
            logger.error(f"[ERROR] {exp_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'experiment': exp_name,
                'status': 'crashed',
                'wall_clock_time': exp_elapsed,
                'error': str(e)
            })
    
    total_elapsed = time.time() - total_start_time
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    
    successful = [r for r in results_summary if r['status'] == 'success']
    failed = [r for r in results_summary if r['status'] != 'success']
    
    logger.info(f"Total experiments: {len(results_summary)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total wall-clock time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    
    if successful:
        logger.info(f"\nSuccessful Experiments:")
        logger.info(f"{'Experiment':<40} {'Queries':>10} {'Time':>10} {'F1':>10} {'S-F1':>10}")
        logger.info("-" * 80)
        for r in successful:
            logger.info(f"{r['experiment']:<40} {r['phase3_queries']:>10} {r['phase3_time']:>10.2f}s {r['f1']:>10.2%} {r['s_f1']:>10.2%}")
    
    if failed:
        logger.info(f"\nFailed Experiments:")
        for r in failed:
            logger.info(f"  - {r['experiment']}: {r.get('error', 'Unknown error')}")
    
    # Save summary to JSON
    summary_file = os.path.join(args.output_dir, f"phase3_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'run_config': {
                'phase2_dir': args.phase2_dir,
                'output_dir': args.output_dir,
                'algorithm': args.algorithm,
                'max_queries': args.max_queries,
                'timeout': args.timeout
            },
            'total_experiments': len(results_summary),
            'successful': len(successful),
            'failed': len(failed),
            'total_wall_clock_time': total_elapsed,
            'results': results_summary
        }, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info(f"Main log saved to: {main_log_file}")
    
    # Also create a simple CSV for easy analysis
    csv_file = os.path.join(args.output_dir, f"phase3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, 'w') as f:
        f.write("experiment,status,phase3_queries,phase3_time,total_queries,total_time,precision,recall,f1,s_precision,s_recall,s_f1,wall_clock_time\n")
        for r in results_summary:
            if r['status'] == 'success':
                f.write(f"{r['experiment']},{r['status']},{r['phase3_queries']},{r['phase3_time']:.2f},"
                        f"{r['total_queries']},{r['total_time']:.2f},{r['precision']:.4f},{r['recall']:.4f},"
                        f"{r['f1']:.4f},{r['s_precision']:.4f},{r['s_recall']:.4f},{r['s_f1']:.4f},{r['wall_clock_time']:.2f}\n")
            else:
                f.write(f"{r['experiment']},{r['status']},,,,,,,,,,{r['wall_clock_time']:.2f}\n")
    
    logger.info(f"CSV results saved to: {csv_file}")
    
    return len(failed)  # Return number of failures as exit code


if __name__ == "__main__":
    sys.exit(main())

