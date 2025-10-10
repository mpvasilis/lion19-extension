"""
Verify all benchmarks have proper GLOBAL mock constraints (not binary).
"""
import cpmpy as cp
from benchmarks_global.sudoku import construct_sudoku
from benchmarks_global.uefa import generate_uefa_instance
from benchmarks_global.vm_allocation import construct_vm_allocation
from benchmarks_global.exam_timetabling import generate_exam_timetabling_instance
from benchmarks_global.nurse_rostering import construct_nurse_rostering


def analyze_constraint_types(name, oracle):
    """Analyze constraint types in benchmark."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Total constraints: {len(oracle.constraints)}")

    # Categorize by arity
    global_constraints = []  # arity > 3
    binary_ternary = []  # arity <= 3

    for c in oracle.constraints:
        c_str = str(c)
        # Count commas to estimate arity (rough but works)
        num_vars = c_str.count('var[') + c_str.count('grid[') + c_str.count('group_')

        if num_vars > 3:
            global_constraints.append(c_str[:100])
        else:
            binary_ternary.append(c_str[:100])

    print(f"Global constraints (arity > 3): {len(global_constraints)}")
    if global_constraints:
        print(f"  Sample: {global_constraints[0]}")

    print(f"Binary/Ternary (arity <= 3): {len(binary_ternary)}")
    if binary_ternary:
        print(f"  Sample: {binary_ternary[0]}")

    # Test SAT
    model = cp.Model(oracle.constraints)
    result = model.solve()
    status = "SAT" if result else "UNSAT"
    print(f"SAT Status: {status}")

    return result


def main():
    print("="*70)
    print("MOCK CONSTRAINT VERIFICATION - GLOBAL vs BINARY")
    print("="*70)

    results = {}

    # Sudoku
    inst, orc = construct_sudoku(3, 3, 9)
    results["Sudoku"] = analyze_constraint_types("SUDOKU", orc)

    # UEFA
    inst, orc = generate_uefa_instance()
    results["UEFA"] = analyze_constraint_types("UEFA", orc)

    # VM Allocation
    pm_data = {
        f"PM{i}": {
            "capacity_cpu": 100,
            "capacity_memory": 128,
            "capacity_disk": 1000,
            "capacity_gpu": 4 if i < 2 else 0
        }
        for i in range(1, 5)
    }
    vm_data = {
        f"VM{i}": {
            "demand_cpu": 10,
            "demand_memory": 16,
            "demand_disk": 100,
            "demand_gpu": 1 if i <= 2 else 0,
            "availability_zone": "AZ1" if i % 2 == 0 else "AZ2",
            "priority": 1 if i <= 3 else 2
        }
        for i in range(1, 7)
    }
    inst, orc = construct_vm_allocation(pm_data, vm_data)
    results["VM Allocation"] = analyze_constraint_types("VM ALLOCATION", orc)

    # Exam Timetabling
    inst, orc = generate_exam_timetabling_instance()
    results["Exam Timetabling"] = analyze_constraint_types("EXAM TIMETABLING", orc)

    # Nurse Rostering
    inst, orc = construct_nurse_rostering()
    results["Nurse Rostering"] = analyze_constraint_types("NURSE ROSTERING", orc)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for benchmark, is_sat in results.items():
        status = "SAT ✓" if is_sat else "UNSAT ✗"
        print(f"{benchmark:20s}: {status}")

    all_sat = all(results.values())
    print("\n" + ("="*70))
    if all_sat:
        print("SUCCESS: All benchmarks are SAT with global mocks")
    else:
        print("FAILURE: Some benchmarks are UNSAT")
    print("="*70)

    return all_sat


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
