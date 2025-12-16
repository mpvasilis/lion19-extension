"""
Check and Fix All Phase 2 Pickle Files
======================================
This script scans all output directories for Phase 2 pickle files,
checks each one for missing binary constraints from the oracle,
and optionally fixes them by adding missing constraints to B_fixed.

Usage:
    # Check only (no modifications):
    python3 check_and_fix_all_pickles.py --check-only
    
    # Check and fix all:
    python3 check_and_fix_all_pickles.py --fix
    
    # Check specific directory:
    python3 check_and_fix_all_pickles.py --directory solution_variance_output --check-only
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from cpmpy import cpm_array
from pycona import ConstraintOracle

# Import benchmark constructors
from benchmarks_global import construct_sudoku, construct_jsudoku, construct_latin_square
from benchmarks_global import construct_graph_coloring_register, construct_graph_coloring_scheduling
from benchmarks_global import construct_sudoku_greater_than
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering as nr_global


def construct_instance(experiment_name):
    """Construct instance and oracle for a given benchmark."""
    
    if 'sudoku_gt' in experiment_name.lower() or 'sudoku_greater' in experiment_name.lower():
        result = construct_sudoku_greater_than(3, 3, 9)
    elif 'sudoku' in experiment_name.lower():
        result = construct_sudoku(3, 3, 9)
    elif 'jsudoku' in experiment_name.lower():
        result = construct_jsudoku(grid_size=9)
    elif 'latin' in experiment_name.lower():
        result = construct_latin_square(n=9)
    elif 'graph_coloring_register' in experiment_name.lower() or 'register' in experiment_name.lower():
        result = construct_graph_coloring_register()
    elif 'examtt_v1' in experiment_name.lower():
        result = construct_examtt_variant1(nsemesters=6, courses_per_semester=5,
                                           slots_per_day=6, days_for_exams=10)
    elif 'examtt_v2' in experiment_name.lower():
        result = construct_examtt_variant2(nsemesters=8, courses_per_semester=8,
                                           slots_per_day=10, days_for_exams=10)
    elif 'examtt' in experiment_name.lower():
        result = ces_global(nsemesters=9, courses_per_semester=6,
                           slots_per_day=9, days_for_exams=14)
    elif 'nurse' in experiment_name.lower():
        result = nr_global()
    else:
        return None, None
    
    if len(result) == 3:
        instance, oracle, _ = result
    else:
        instance, oracle = result
    
    return instance, oracle


def decompose_global_constraints(global_constraints):
    """Decompose global constraints to binary."""
    binary_constraints = []
    
    for c in global_constraints:
        if hasattr(c, 'name') and c.name == "alldifferent":
            c_str = str(c)
            if '//' in c_str or '/' in c_str or '*' in c_str or '+' in c_str or '%' in c_str:
                continue
            
            decomposed = c.decompose()
            if decomposed and len(decomposed) > 0:
                binary_constraints.extend(decomposed[0])
        else:
            binary_constraints.append(c)
    
    # Remove duplicates
    unique_constraints = list({str(c): c for c in binary_constraints}.values())
    return unique_constraints


def extract_experiment_name(pickle_path):
    """Extract experiment name from pickle file path or name."""
    path_str = str(pickle_path)
    
    # Try to extract from directory name or file name
    if 'sudoku_gt' in path_str or 'sudoku_greater' in path_str:
        return 'sudoku_gt'
    elif 'jsudoku' in path_str:
        return 'jsudoku'
    elif 'sudoku' in path_str:
        return 'sudoku'
    elif 'latin' in path_str:
        return 'latin_square'
    elif 'graph_coloring_register' in path_str or 'register' in path_str:
        return 'graph_coloring_register'
    elif 'examtt_v1' in path_str:
        return 'examtt_v1'
    elif 'examtt_v2' in path_str:
        return 'examtt_v2'
    elif 'examtt' in path_str:
        return 'examtt'
    elif 'nurse' in path_str:
        return 'nurse'
    
    return None


def check_pickle_file(pickle_path, experiment_name=None, fix=False):
    """
    Check a single Phase 2 pickle file for missing constraints.
    
    Args:
        pickle_path: Path to the Phase 2 pickle file
        experiment_name: Name of the experiment (auto-detected if None)
        fix: If True, add missing constraints and save
    
    Returns:
        dict with check results
    """
    
    # Auto-detect experiment name if not provided
    if experiment_name is None:
        experiment_name = extract_experiment_name(pickle_path)
        if experiment_name is None:
            return {
                'status': 'error',
                'message': 'Could not determine experiment name',
                'pickle_path': str(pickle_path)
            }
    
    try:
        # Load Phase 2 data
        with open(pickle_path, 'rb') as f:
            phase2_data = pickle.load(f)
        
        C_validated = phase2_data.get('C_validated', [])
        phase1_data = phase2_data.get('phase1_data', None)
        
        if phase1_data is None:
            return {
                'status': 'error',
                'message': 'No Phase 1 data in pickle',
                'pickle_path': str(pickle_path)
            }
        
        B_fixed = phase1_data.get('B_fixed', [])
        
        # Construct oracle
        instance_global, oracle_global = construct_instance(experiment_name)
        if instance_global is None or oracle_global is None:
            return {
                'status': 'error',
                'message': f'Unknown experiment: {experiment_name}',
                'pickle_path': str(pickle_path)
            }
        
        oracle_global.variables_list = cpm_array(instance_global.X)
        
        # Create decomposed oracle
        decomposed_constraints = []
        non_global_constraints = []
        
        for c in oracle_global.constraints:
            if hasattr(c, 'name') and c.name == 'alldifferent':
                decomposed = c.decompose()
                if decomposed and len(decomposed) > 0:
                    decomposed_constraints.extend(decomposed[0])
            else:
                non_global_constraints.append(c)
        
        unique_binary = list({str(c): c for c in decomposed_constraints}.values())
        all_binary_constraints = unique_binary + non_global_constraints
        
        # Decompose C_validated
        CL_init = decompose_global_constraints(C_validated)
        
        # Find missing constraints
        CL_init_strs = set(str(c) for c in CL_init)
        B_fixed_strs = set(str(c) for c in B_fixed)
        
        missing_constraints = []
        for c in all_binary_constraints:
            c_str = str(c)
            if c_str not in CL_init_strs and c_str not in B_fixed_strs:
                missing_constraints.append(c)
        
        result = {
            'status': 'ok',
            'pickle_path': str(pickle_path),
            'experiment': experiment_name,
            'C_validated_size': len(C_validated),
            'B_fixed_size': len(B_fixed),
            'CL_init_size': len(CL_init),
            'oracle_size': len(all_binary_constraints),
            'missing_count': len(missing_constraints),
            'missing_constraints': [str(c) for c in missing_constraints],
        }
        
        # Fix if requested
        if fix and len(missing_constraints) > 0:
            print(f"  [FIX] Adding {len(missing_constraints)} missing constraints to B_fixed...")
            B_fixed.extend(missing_constraints)
            
            # Save updated pickle
            with open(pickle_path, 'wb') as f:
                pickle.dump(phase2_data, f)
            
            result['fixed'] = True
            result['B_fixed_size_after'] = len(B_fixed)
        else:
            result['fixed'] = False
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'pickle_path': str(pickle_path),
            'exception_type': type(e).__name__
        }


def find_phase2_pickles(directory):
    """Find all Phase 2 pickle files in a directory recursively."""
    pickles = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Look for Phase 2 pickle files
            if file.endswith('_phase2.pkl') or 'phase2' in file.lower():
                full_path = os.path.join(root, file)
                pickles.append(full_path)
    
    return pickles


def main():
    parser = argparse.ArgumentParser(
        description='Check and fix all Phase 2 pickle files for missing binary constraints'
    )
    parser.add_argument('--directory', type=str, nargs='+',
                       default=['solution_variance_output', 'solution_variance_output_phase2',
                               'lion19_postval_comparison_output', 'phase2_output'],
                       help='Directories to scan for pickle files (default: all output directories)')
    parser.add_argument('--fix', action='store_true',
                       help='Fix pickle files by adding missing constraints')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check, do not modify any files (default if --fix not specified)')
    parser.add_argument('--output-report', type=str, default='pickle_check_report.txt',
                       help='Output file for detailed report')
    
    args = parser.parse_args()
    
    # Default is check-only unless --fix is specified
    fix_mode = args.fix and not args.check_only
    
    print(f"\n{'='*80}")
    print(f"Checking Phase 2 Pickle Files for Missing Constraints")
    print(f"{'='*80}")
    print(f"Mode: {'FIX (will modify files)' if fix_mode else 'CHECK ONLY (read-only)'}")
    print(f"Directories: {args.directory}")
    print(f"{'='*80}\n")
    
    # Find all pickle files
    all_pickles = []
    for directory in args.directory:
        if os.path.exists(directory):
            print(f"Scanning {directory}...")
            pickles = find_phase2_pickles(directory)
            all_pickles.extend(pickles)
            print(f"  Found {len(pickles)} Phase 2 pickle files")
        else:
            print(f"  [SKIP] Directory not found: {directory}")
    
    print(f"\nTotal Phase 2 pickle files found: {len(all_pickles)}\n")
    
    if len(all_pickles) == 0:
        print("[WARNING] No Phase 2 pickle files found!")
        sys.exit(0)
    
    # Check each pickle
    results = []
    files_with_missing = []
    files_ok = []
    files_error = []
    
    for i, pickle_path in enumerate(all_pickles, 1):
        rel_path = os.path.relpath(pickle_path)
        print(f"\n[{i}/{len(all_pickles)}] Checking: {rel_path}")
        
        result = check_pickle_file(pickle_path, experiment_name=None, fix=fix_mode)
        results.append(result)
        
        if result['status'] == 'error':
            files_error.append(rel_path)
            print(f"  [ERROR] {result.get('message', 'Unknown error')}")
        elif result['missing_count'] > 0:
            files_with_missing.append(rel_path)
            print(f"  [MISSING] {result['missing_count']} constraints not in CL_init or B_fixed")
            print(f"    Experiment: {result['experiment']}")
            print(f"    C_validated: {result['C_validated_size']}, B_fixed: {result['B_fixed_size']}, Oracle: {result['oracle_size']}")
            
            # Show missing constraints
            if result['missing_count'] <= 10:
                for c in result['missing_constraints']:
                    print(f"      - {c}")
            else:
                for c in result['missing_constraints'][:5]:
                    print(f"      - {c}")
                print(f"      ... and {result['missing_count'] - 5} more")
            
            if result.get('fixed', False):
                print(f"  [FIXED] Added to B_fixed. New size: {result['B_fixed_size_after']}")
        else:
            files_ok.append(rel_path)
            print(f"  [OK] All {result['oracle_size']} oracle constraints covered")
    
    # Generate summary report
    print(f"\n\n{'='*80}")
    print(f"SUMMARY REPORT")
    print(f"{'='*80}")
    print(f"Total files checked: {len(all_pickles)}")
    print(f"  - OK (no missing constraints): {len(files_ok)}")
    print(f"  - Has missing constraints: {len(files_with_missing)}")
    print(f"  - Errors: {len(files_error)}")
    
    if fix_mode:
        fixed_count = sum(1 for r in results if r.get('fixed', False))
        print(f"  - Fixed: {fixed_count}")
    
    # Detailed results by experiment
    print(f"\n{'='*80}")
    print(f"Results by Experiment")
    print(f"{'='*80}")
    
    by_experiment = {}
    for result in results:
        if result['status'] == 'ok':
            exp = result['experiment']
            if exp not in by_experiment:
                by_experiment[exp] = []
            by_experiment[exp].append(result)
    
    for exp in sorted(by_experiment.keys()):
        exp_results = by_experiment[exp]
        total = len(exp_results)
        with_missing = sum(1 for r in exp_results if r['missing_count'] > 0)
        
        print(f"\n{exp}:")
        print(f"  Files: {total}")
        print(f"  With missing constraints: {with_missing}")
        
        if with_missing > 0:
            # Show summary of missing counts
            missing_counts = [r['missing_count'] for r in exp_results if r['missing_count'] > 0]
            avg_missing = sum(missing_counts) / len(missing_counts)
            max_missing = max(missing_counts)
            min_missing = min(missing_counts)
            print(f"    Missing range: {min_missing}-{max_missing} (avg: {avg_missing:.1f})")
            
            # Show which specific constraints are commonly missing
            all_missing = {}
            for r in exp_results:
                if r['missing_count'] > 0:
                    for c in r['missing_constraints']:
                        all_missing[c] = all_missing.get(c, 0) + 1
            
            if all_missing:
                sorted_missing = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)
                print(f"    Most common missing constraints:")
                for c, count in sorted_missing[:5]:
                    print(f"      - {c} (in {count}/{with_missing} files)")
    
    # Save detailed report
    print(f"\n{'='*80}")
    print(f"Saving detailed report to: {args.output_report}")
    print(f"{'='*80}")
    
    with open(args.output_report, 'w') as f:
        f.write(f"Phase 2 Pickle Files - Missing Constraints Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {'FIX' if fix_mode else 'CHECK ONLY'}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"SUMMARY\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Total files checked: {len(all_pickles)}\n")
        f.write(f"  OK: {len(files_ok)}\n")
        f.write(f"  With missing constraints: {len(files_with_missing)}\n")
        f.write(f"  Errors: {len(files_error)}\n")
        if fix_mode:
            f.write(f"  Fixed: {sum(1 for r in results if r.get('fixed', False))}\n")
        f.write(f"\n")
        
        # Details
        f.write(f"\nDETAILED RESULTS\n")
        f.write(f"{'='*80}\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"[{i}] {result['pickle_path']}\n")
            
            if result['status'] == 'error':
                f.write(f"  Status: ERROR\n")
                f.write(f"  Message: {result.get('message', 'Unknown')}\n")
            else:
                f.write(f"  Experiment: {result['experiment']}\n")
                f.write(f"  C_validated: {result['C_validated_size']}\n")
                f.write(f"  B_fixed: {result['B_fixed_size']}\n")
                f.write(f"  CL_init: {result['CL_init_size']}\n")
                f.write(f"  Oracle total: {result['oracle_size']}\n")
                f.write(f"  Missing: {result['missing_count']}\n")
                
                if result['missing_count'] > 0:
                    f.write(f"  Missing constraints:\n")
                    for c in result['missing_constraints']:
                        f.write(f"    - {c}\n")
                    
                    if result.get('fixed', False):
                        f.write(f"  FIXED: B_fixed updated to {result['B_fixed_size_after']}\n")
            
            f.write(f"\n")
    
    print(f"Report saved successfully!\n")
    
    # Final summary
    if len(files_with_missing) > 0:
        print(f"\n{'='*80}")
        print(f"FILES WITH MISSING CONSTRAINTS ({len(files_with_missing)}):")
        print(f"{'='*80}")
        for f in files_with_missing[:20]:
            print(f"  - {f}")
        if len(files_with_missing) > 20:
            print(f"  ... and {len(files_with_missing) - 20} more")
    
    if len(files_error) > 0:
        print(f"\n{'='*80}")
        print(f"FILES WITH ERRORS ({len(files_error)}):")
        print(f"{'='*80}")
        for f in files_error:
            print(f"  - {f}")
    
    print(f"\n")
    
    if fix_mode and len(files_with_missing) > 0:
        print(f"[SUCCESS] Fixed {sum(1 for r in results if r.get('fixed', False))} pickle files")
    elif len(files_with_missing) > 0:
        print(f"[INFO] Found {len(files_with_missing)} files with missing constraints")
        print(f"[INFO] Run with --fix to update pickle files")
    else:
        print(f"[SUCCESS] All pickle files have complete constraint coverage!")


if __name__ == "__main__":
    main()

