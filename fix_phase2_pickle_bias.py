"""
Fix Phase 2 Pickle: Add Missing Binary Constraints to B_fixed
==============================================================
This script loads a Phase 2 pickle file, checks for missing constraints
from the oracle that should be in the bias, adds them, and saves the
updated pickle file.
"""

import sys
import pickle
import argparse
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
        print("Constructing 9x9 Sudoku with Greater-Than constraints...")
        result = construct_sudoku_greater_than(3, 3, 9)
    elif 'sudoku' in experiment_name.lower():
        print("Constructing 9x9 Sudoku...")
        result = construct_sudoku(3, 3, 9)
    elif 'jsudoku' in experiment_name.lower():
        print("Constructing 9x9 JSudoku...")
        result = construct_jsudoku(grid_size=9)
    elif 'latin' in experiment_name.lower():
        print("Constructing 9x9 Latin Square...")
        result = construct_latin_square(n=9)
    elif 'graph_coloring_register' in experiment_name.lower():
        print("Constructing Graph Coloring (Register)...")
        result = construct_graph_coloring_register()
    elif 'examtt_v1' in experiment_name.lower():
        print("Constructing ExamTT V1...")
        result = construct_examtt_variant1(nsemesters=6, courses_per_semester=5,
                                           slots_per_day=6, days_for_exams=10)
    elif 'examtt_v2' in experiment_name.lower():
        print("Constructing ExamTT V2...")
        result = construct_examtt_variant2(nsemesters=8, courses_per_semester=8,
                                           slots_per_day=10, days_for_exams=10)
    elif 'examtt' in experiment_name.lower():
        print("Constructing ExamTT...")
        result = ces_global(nsemesters=9, courses_per_semester=6,
                           slots_per_day=9, days_for_exams=14)
    elif 'nurse' in experiment_name.lower():
        print("Constructing Nurse Rostering...")
        result = nr_global()
    else:
        print(f"Unknown experiment: {experiment_name}")
        sys.exit(1)
    
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


def fix_phase2_pickle(phase2_pickle_path, experiment_name, output_path=None):
    """
    Fix a Phase 2 pickle file by adding missing binary constraints to B_fixed.
    
    Args:
        phase2_pickle_path: Path to the Phase 2 pickle file
        experiment_name: Name of the experiment/benchmark
        output_path: Optional output path (default: overwrite input)
    """
    
    print(f"\n{'='*80}")
    print(f"Fixing Phase 2 Pickle: {phase2_pickle_path}")
    print(f"{'='*80}\n")
    
    # Load Phase 2 data
    print(f"Loading Phase 2 data...")
    with open(phase2_pickle_path, 'rb') as f:
        phase2_data = pickle.load(f)
    
    C_validated = phase2_data['C_validated']
    phase1_data = phase2_data.get('phase1_data', None)
    
    if phase1_data is None:
        print("\n[ERROR] No Phase 1 data found in Phase 2 pickle!")
        sys.exit(1)
    
    B_fixed = phase1_data.get('B_fixed', [])
    
    print(f"Loaded:")
    print(f"  - C_validated: {len(C_validated)} constraints")
    print(f"  - B_fixed: {len(B_fixed)} constraints")
    
    # Construct oracle
    instance_global, oracle_global = construct_instance(experiment_name)
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
    
    oracle_decomposed = ConstraintOracle(all_binary_constraints)
    oracle_decomposed.variables_list = cpm_array(instance_global.X)
    oracle_decomposed.target_network = all_binary_constraints
    
    print(f"\nOracle target network: {len(oracle_decomposed.constraints)} constraints")
    print(f"  - {len(unique_binary)} != constraints (from AllDifferent)")
    print(f"  - {len(non_global_constraints)} other binary constraints")
    
    # Decompose C_validated to get CL_init
    CL_init = decompose_global_constraints(C_validated)
    print(f"\nCL_init (decomposed C_validated): {len(CL_init)} constraints")
    
    # Find missing constraints
    CL_init_strs = set(str(c) for c in CL_init)
    B_fixed_strs = set(str(c) for c in B_fixed)
    
    missing_constraints = []
    for c in oracle_decomposed.constraints:
        c_str = str(c)
        if c_str not in CL_init_strs and c_str not in B_fixed_strs:
            missing_constraints.append(c)
    
    print(f"\n{'='*80}")
    print(f"Step 2.5: Check for Missing Constraints from Oracle")
    print(f"{'='*80}")
    print(f"Found {len(missing_constraints)} constraints in oracle but not in CL_init or B_fixed")
    
    if len(missing_constraints) > 0:
        print(f"\nMissing constraints:")
        for c in missing_constraints:
            print(f"  - {c}")
        
        print(f"\nAdding {len(missing_constraints)} missing constraints to B_fixed...")
        B_fixed.extend(missing_constraints)
        
        print(f"Updated B_fixed size: {len(B_fixed)}")
        
        # Validate coverage
        B_fixed_strs_updated = set(str(c) for c in B_fixed)
        still_missing = []
        for c in oracle_decomposed.constraints:
            c_str = str(c)
            if c_str not in CL_init_strs and c_str not in B_fixed_strs_updated:
                still_missing.append(c)
        
        if len(still_missing) == 0:
            print(f"\n[SUCCESS] All {len(oracle_decomposed.constraints)} oracle constraints are now covered!")
            print(f"  - In CL_init: {len([c for c in oracle_decomposed.constraints if str(c) in CL_init_strs])}")
            print(f"  - In B_fixed: {len([c for c in oracle_decomposed.constraints if str(c) in B_fixed_strs_updated])}")
        else:
            print(f"\n[WARNING] Still missing {len(still_missing)} constraints!")
        
        # Save updated pickle
        output_path = output_path or phase2_pickle_path
        print(f"\nSaving updated Phase 2 pickle to: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(phase2_data, f)
        
        print(f"[SUCCESS] Phase 2 pickle updated successfully!")
        print(f"\nYou can now run Phase 3 with: {output_path}")
        
    else:
        print(f"\n[INFO] No missing constraints found. B_fixed is complete.")
        print(f"No changes needed to Phase 2 pickle.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fix Phase 2 pickle by adding missing binary constraints to B_fixed'
    )
    parser.add_argument('--phase2_pickle', type=str, required=True,
                       help='Path to Phase 2 pickle file to fix')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment/benchmark name (e.g., sudoku_gt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for fixed pickle (default: overwrite input)')
    
    args = parser.parse_args()
    
    fix_phase2_pickle(args.phase2_pickle, args.experiment, args.output)

