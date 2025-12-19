"""
Check and Fix All Phase 2 Pickle Files
======================================
This script scans all output directories for Phase 2 pickle files,
checks each one for:
  1. Missing binary constraints from the oracle (not in CL_init or B_fixed)
  2. Spurious constraints in CL_init (constraints not in the target oracle)
  3. Completeness: B_fixed + CL_init should equal the target oracle model

If any constraint is missing, it adds them to B_fixed.

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
    Check a single Phase 2 pickle file for:
      1. Missing constraints from the oracle (not in CL_init or B_fixed)
      2. Spurious constraints in CL_init (not in the target oracle)
      3. Completeness: B_fixed + CL_init should equal the target oracle
    
    Args:
        pickle_path: Path to the Phase 2 pickle file
        experiment_name: Name of the experiment (auto-detected if None)
        fix: If True, add missing constraints to B_fixed and save
    
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
        
        # Create decomposed oracle (target model)
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
        oracle_strs = set(str(c) for c in all_binary_constraints)
        
        # Decompose C_validated to get CL_init
        CL_init = decompose_global_constraints(C_validated)
        
        # Get string representations
        CL_init_strs = set(str(c) for c in CL_init)
        B_fixed_strs = set(str(c) for c in B_fixed)
        combined_strs = CL_init_strs | B_fixed_strs  # Union of CL_init and B_fixed
        
        # CHECK 1: Find missing constraints (in oracle but not in CL_init ∪ B_fixed)
        missing_constraints = []
        for c in all_binary_constraints:
            c_str = str(c)
            if c_str not in combined_strs:
                missing_constraints.append(c)
        
        # CHECK 2: Find spurious constraints in CL_init (not in oracle)
        spurious_constraints = []
        for c in CL_init:
            c_str = str(c)
            if c_str not in oracle_strs:
                spurious_constraints.append(c)
        
        # CHECK 3: Completeness check - B_fixed + CL_init should equal oracle
        # This is essentially: missing_count == 0 AND spurious_count == 0
        is_complete = (len(missing_constraints) == 0)
        has_only_oracle_constraints = (len(spurious_constraints) == 0)
        
        result = {
            'status': 'ok',
            'pickle_path': str(pickle_path),
            'experiment': experiment_name,
            'C_validated_size': len(C_validated),
            'B_fixed_size': len(B_fixed),
            'CL_init_size': len(CL_init),
            'oracle_size': len(all_binary_constraints),
            'combined_size': len(combined_strs),
            # Missing constraints (need to be added to B_fixed)
            'missing_count': len(missing_constraints),
            'missing_constraints': [str(c) for c in missing_constraints],
            # Spurious constraints (in CL_init but not in oracle)
            'spurious_count': len(spurious_constraints),
            'spurious_constraints': [str(c) for c in spurious_constraints],
            # Completeness status
            'is_complete': is_complete,
            'has_only_oracle_constraints': has_only_oracle_constraints,
            'model_match': is_complete and has_only_oracle_constraints,
        }
        
        # Fix if requested
        needs_save = False
        
        # Fix 1: Add missing constraints to B_fixed
        if fix and len(missing_constraints) > 0:
            print(f"  [FIX] Adding {len(missing_constraints)} missing constraints to B_fixed...")
            B_fixed.extend(missing_constraints)
            result['fixed_missing'] = True
            result['B_fixed_size_after'] = len(B_fixed)
            needs_save = True
        else:
            result['fixed_missing'] = False
        
        # Fix 2: Remove spurious constraints from C_validated
        if fix and len(spurious_constraints) > 0:
            print(f"  [FIX] Removing {len(spurious_constraints)} spurious constraints from C_validated...")
            spurious_strs = set(str(c) for c in spurious_constraints)
            # Filter out spurious constraints from C_validated
            C_validated_cleaned = [c for c in C_validated if str(c) not in spurious_strs]
            # Also check decomposed form - some constraints might be global that decompose to spurious
            # We need to be careful here: remove any constraint whose decomposition contains spurious
            final_cleaned = []
            for c in C_validated_cleaned:
                if hasattr(c, 'name') and c.name == "alldifferent":
                    # Check if any decomposed constraint is spurious
                    decomposed = c.decompose()
                    if decomposed and len(decomposed) > 0:
                        decomposed_strs = set(str(dc) for dc in decomposed[0])
                        # Keep if all decomposed constraints are in oracle
                        if decomposed_strs.issubset(oracle_strs):
                            final_cleaned.append(c)
                        else:
                            print(f"    Removing global constraint with spurious decomposition: {c}")
                    else:
                        final_cleaned.append(c)
                else:
                    # Non-global constraint - check directly
                    if str(c) in oracle_strs:
                        final_cleaned.append(c)
            
            phase2_data['C_validated'] = final_cleaned
            result['fixed_spurious'] = True
            result['C_validated_size_after'] = len(final_cleaned)
            needs_save = True
        else:
            result['fixed_spurious'] = False
        
        result['fixed'] = result['fixed_missing'] or result['fixed_spurious']
        
        # Save updated pickle if any fixes were made
        if needs_save:
            with open(pickle_path, 'wb') as f:
                pickle.dump(phase2_data, f)
        
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
    files_with_spurious = []
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
        else:
            has_issues = False
            
            # Check for missing constraints
            if result['missing_count'] > 0:
                files_with_missing.append(rel_path)
                has_issues = True
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
                
                if result.get('fixed_missing', False):
                    print(f"  [FIXED] Added to B_fixed. New size: {result['B_fixed_size_after']}")
            
            # Check for spurious constraints (in CL_init but not in oracle)
            if result['spurious_count'] > 0:
                files_with_spurious.append(rel_path)
                has_issues = True
                print(f"  [SPURIOUS] {result['spurious_count']} constraints in CL_init NOT in oracle!")
                
                # Show spurious constraints
                if result['spurious_count'] <= 10:
                    for c in result['spurious_constraints']:
                        print(f"      - {c}")
                else:
                    for c in result['spurious_constraints'][:5]:
                        print(f"      - {c}")
                    print(f"      ... and {result['spurious_count'] - 5} more")
                
                if result.get('fixed_spurious', False):
                    print(f"  [FIXED] Removed spurious from C_validated. New size: {result['C_validated_size_after']}")
            
            # Model match status
            if not has_issues:
                files_ok.append(rel_path)
                print(f"  [OK] B_fixed + CL_init = Oracle ({result['oracle_size']} constraints)")
            else:
                # Show completeness summary
                print(f"  [MODEL CHECK] B_fixed({result['B_fixed_size']}) + CL_init({result['CL_init_size']}) = {result['combined_size']} unique | Oracle = {result['oracle_size']}")
    
    # Generate summary report
    print(f"\n\n{'='*80}")
    print(f"SUMMARY REPORT")
    print(f"{'='*80}")
    print(f"Total files checked: {len(all_pickles)}")
    print(f"  - OK (B_fixed + CL_init = Oracle): {len(files_ok)}")
    print(f"  - Has missing constraints: {len(files_with_missing)}")
    print(f"  - Has spurious constraints (not in oracle): {len(files_with_spurious)}")
    print(f"  - Errors: {len(files_error)}")
    
    if fix_mode:
        fixed_missing = sum(1 for r in results if r.get('fixed_missing', False))
        fixed_spurious = sum(1 for r in results if r.get('fixed_spurious', False))
        total_fixed = sum(1 for r in results if r.get('fixed', False))
        print(f"  - Fixed total: {total_fixed}")
        print(f"    - Missing constraints added to B_fixed: {fixed_missing}")
        print(f"    - Spurious constraints removed from C_validated: {fixed_spurious}")
    
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
        with_spurious = sum(1 for r in exp_results if r['spurious_count'] > 0)
        model_matches = sum(1 for r in exp_results if r.get('model_match', False))
        
        print(f"\n{exp}:")
        print(f"  Files: {total}")
        print(f"  Model matches (B_fixed + CL_init = Oracle): {model_matches}/{total}")
        print(f"  With missing constraints: {with_missing}")
        print(f"  With spurious constraints: {with_spurious}")
        
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
        
        if with_spurious > 0:
            # Show summary of spurious counts
            spurious_counts = [r['spurious_count'] for r in exp_results if r['spurious_count'] > 0]
            avg_spurious = sum(spurious_counts) / len(spurious_counts)
            max_spurious = max(spurious_counts)
            min_spurious = min(spurious_counts)
            print(f"    Spurious range: {min_spurious}-{max_spurious} (avg: {avg_spurious:.1f})")
            
            # Show which specific constraints are commonly spurious
            all_spurious = {}
            for r in exp_results:
                if r['spurious_count'] > 0:
                    for c in r['spurious_constraints']:
                        all_spurious[c] = all_spurious.get(c, 0) + 1
            
            if all_spurious:
                sorted_spurious = sorted(all_spurious.items(), key=lambda x: x[1], reverse=True)
                print(f"    Most common spurious constraints (NOT in oracle):")
                for c, count in sorted_spurious[:5]:
                    print(f"      - {c} (in {count}/{with_spurious} files)")
    
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
        f.write(f"  OK (B_fixed + CL_init = Oracle): {len(files_ok)}\n")
        f.write(f"  With missing constraints: {len(files_with_missing)}\n")
        f.write(f"  With spurious constraints: {len(files_with_spurious)}\n")
        f.write(f"  Errors: {len(files_error)}\n")
        if fix_mode:
            fixed_missing = sum(1 for r in results if r.get('fixed_missing', False))
            fixed_spurious = sum(1 for r in results if r.get('fixed_spurious', False))
            total_fixed = sum(1 for r in results if r.get('fixed', False))
            f.write(f"  Fixed total: {total_fixed}\n")
            f.write(f"    - Missing constraints added to B_fixed: {fixed_missing}\n")
            f.write(f"    - Spurious constraints removed from C_validated: {fixed_spurious}\n")
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
                f.write(f"  CL_init (decomposed): {result['CL_init_size']}\n")
                f.write(f"  Combined (B_fixed ∪ CL_init): {result['combined_size']}\n")
                f.write(f"  Oracle (target model): {result['oracle_size']}\n")
                f.write(f"  Model match: {result.get('model_match', False)}\n")
                f.write(f"  Missing: {result['missing_count']}\n")
                f.write(f"  Spurious: {result['spurious_count']}\n")
                
                if result['missing_count'] > 0:
                    f.write(f"  Missing constraints (to add to B_fixed):\n")
                    for c in result['missing_constraints']:
                        f.write(f"    - {c}\n")
                    
                    if result.get('fixed_missing', False):
                        f.write(f"  FIXED: B_fixed updated to {result['B_fixed_size_after']}\n")
                
                if result['spurious_count'] > 0:
                    f.write(f"  Spurious constraints (in CL_init but NOT in oracle):\n")
                    for c in result['spurious_constraints']:
                        f.write(f"    - {c}\n")
                    
                    if result.get('fixed_spurious', False):
                        f.write(f"  FIXED: Spurious removed, C_validated updated to {result['C_validated_size_after']}\n")
            
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
    
    if len(files_with_spurious) > 0:
        print(f"\n{'='*80}")
        print(f"FILES WITH SPURIOUS CONSTRAINTS ({len(files_with_spurious)}):")
        print(f"{'='*80}")
        for f in files_with_spurious[:20]:
            print(f"  - {f}")
        if len(files_with_spurious) > 20:
            print(f"  ... and {len(files_with_spurious) - 20} more")
    
    if len(files_error) > 0:
        print(f"\n{'='*80}")
        print(f"FILES WITH ERRORS ({len(files_error)}):")
        print(f"{'='*80}")
        for f in files_error:
            print(f"  - {f}")
    
    print(f"\n")
    
    if fix_mode:
        fixed_missing_count = sum(1 for r in results if r.get('fixed_missing', False))
        fixed_spurious_count = sum(1 for r in results if r.get('fixed_spurious', False))
        total_fixed = sum(1 for r in results if r.get('fixed', False))
        
        if total_fixed > 0:
            print(f"[SUCCESS] Fixed {total_fixed} pickle files:")
            if fixed_missing_count > 0:
                print(f"  - {fixed_missing_count} files: added missing constraints to B_fixed")
            if fixed_spurious_count > 0:
                print(f"  - {fixed_spurious_count} files: removed spurious constraints from C_validated")
    
    if len(files_with_missing) > 0 and not fix_mode:
        print(f"[INFO] Found {len(files_with_missing)} files with missing constraints")
        print(f"[INFO] Run with --fix to add missing constraints to B_fixed")
    
    if len(files_with_spurious) > 0 and not fix_mode:
        print(f"[WARNING] Found {len(files_with_spurious)} files with spurious constraints in CL_init")
        print(f"[WARNING] These constraints are NOT in the target oracle model!")
        print(f"[INFO] Run with --fix to remove spurious constraints from C_validated")
    
    if len(files_with_missing) == 0 and len(files_with_spurious) == 0 and len(files_error) == 0:
        print(f"[SUCCESS] All pickle files satisfy: B_fixed + CL_init = Oracle (target model)!")


if __name__ == "__main__":
    main()

