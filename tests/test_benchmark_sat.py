"""
Test script to verify all benchmarks are satisfiable after adding mock constraints.
"""

import cpmpy as cp
from benchmarks_global.sudoku import construct_sudoku
from benchmarks_global.uefa import generate_uefa_instance
from benchmarks_global.vm_allocation import construct_vm_allocation
from benchmarks_global.exam_timetabling import generate_exam_timetabling_instance
from benchmarks_global.nurse_rostering import construct_nurse_rostering


def test_benchmark_sat(name, oracle):
    """Test if a benchmark is satisfiable."""
    print(f"\nTesting {name}...")
    model = cp.Model(oracle.constraints)
    result = model.solve()
    status = "SAT" if result else "UNSAT"
    print(f"  {name}: {status}")
    return result


def main():
    print("=" * 70)
    print("BENCHMARK SATISFIABILITY TEST")
    print("=" * 70)

    results = {}

    # Sudoku
    inst, orc = construct_sudoku(3, 3, 9)
    results["Sudoku"] = test_benchmark_sat("Sudoku", orc)

    # UEFA
    inst, orc = generate_uefa_instance()
    results["UEFA"] = test_benchmark_sat("UEFA", orc)

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
    results["VM Allocation"] = test_benchmark_sat("VM Allocation", orc)

    # Exam Timetabling
    inst, orc = generate_exam_timetabling_instance()
    results["Exam Timetabling"] = test_benchmark_sat("Exam Timetabling", orc)

    # Nurse Rostering (use default parameters which are now SAT)
    inst, orc = construct_nurse_rostering()
    results["Nurse Rostering"] = test_benchmark_sat("Nurse Rostering", orc)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for benchmark, is_sat in results.items():
        status = "SAT" if is_sat else "UNSAT (PROBLEM!)"
        print(f"{benchmark:20s}: {status}")

    all_sat = all(results.values())
    print("\n" + ("=" * 70))
    if all_sat:
        print("SUCCESS: All benchmarks are satisfiable")
    else:
        print("FAILURE: Some benchmarks are UNSAT")
    print("=" * 70)

    return all_sat


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
