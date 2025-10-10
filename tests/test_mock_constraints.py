"""
Test script to verify mock over-fitted constraints are added to all benchmarks.
"""

from benchmarks_global.sudoku import construct_sudoku
from benchmarks_global.uefa import generate_uefa_instance
from benchmarks_global.vm_allocation import construct_vm_allocation
from benchmarks_global.exam_timetabling import generate_exam_timetabling_instance
from benchmarks_global.nurse_rostering import construct_nurse_rostering


def test_sudoku():
    print("=" * 70)
    print("TESTING SUDOKU BENCHMARK")
    print("=" * 70)
    inst, orc = construct_sudoku(3, 3, 9)
    total = len(orc.constraints)
    print(f"Total constraints: {total} (expected: 27 original + 2 mock = 29)")

    # Check for diagonal constraints
    main_diag = [c for c in orc.constraints if 'grid[0,0]' in str(c) and 'grid[1,1]' in str(c)]
    anti_diag = [c for c in orc.constraints if 'grid[0,8]' in str(c) and 'grid[1,7]' in str(c)]

    print(f"Main diagonal mock: {'FOUND' if main_diag else 'MISSING'}")
    print(f"Anti-diagonal mock: {'FOUND' if anti_diag else 'MISSING'}")

    if main_diag:
        print(f"  Main diagonal: {main_diag[0]}")
    if anti_diag:
        print(f"  Anti-diagonal: {anti_diag[0]}")

    return len(main_diag) > 0 and len(anti_diag) > 0


def test_uefa():
    print("\n" + "=" * 70)
    print("TESTING UEFA BENCHMARK")
    print("=" * 70)
    inst, orc = generate_uefa_instance()
    total = len(orc.constraints)
    print(f"Total constraints: {total}")

    # Check for mock constraints (first 4 and last 4 teams)
    first_four_mock = [c for c in orc.constraints if 'Team_1' in str(c) and 'Team_10' in str(c)]
    last_four_mock = [c for c in orc.constraints if 'Team_29' in str(c) and 'Team_30' in str(c)]

    print(f"First 4 teams mock: {'FOUND' if first_four_mock else 'MISSING'}")
    print(f"Last 4 teams mock: {'FOUND' if last_four_mock else 'MISSING'}")

    if first_four_mock:
        print(f"  {first_four_mock[0]}")
    if last_four_mock:
        print(f"  {last_four_mock[0]}")

    return len(first_four_mock) > 0 and len(last_four_mock) > 0


def test_vm_allocation():
    print("\n" + "=" * 70)
    print("TESTING VM ALLOCATION BENCHMARK")
    print("=" * 70)

    # Create sample data
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
    total = len(orc.constraints)
    print(f"Total constraints: {total}")

    # Check for mock constraints
    alldiff_mock = [c for c in orc.constraints if 'alldifferent' in str(c).lower() and 'assign_VM1' in str(c)]
    balance_mock = [c for c in orc.constraints if '==' in str(c) and 'cpu_PM1' in str(c)]

    print(f"AllDifferent VMs mock: {'FOUND' if alldiff_mock else 'MISSING'}")
    print(f"CPU balance mock: {'FOUND' if balance_mock else 'MISSING'}")

    if alldiff_mock:
        print(f"  {alldiff_mock[0]}")
    if balance_mock:
        print(f"  {balance_mock[0]}")

    return len(alldiff_mock) > 0 or len(balance_mock) > 0


def test_exam_timetabling():
    print("\n" + "=" * 70)
    print("TESTING EXAM TIMETABLING BENCHMARK")
    print("=" * 70)
    inst, orc = generate_exam_timetabling_instance()
    total = len(orc.constraints)
    print(f"Total constraints: {total}")

    # Check for mock constraints
    # These will be complex expressions, just check count increased
    print(f"Mock constraints should be present (total constraints > expected baseline)")

    # Sample some constraints
    print("Sample constraints:")
    for i, c in enumerate(list(orc.constraints)[-5:]):
        print(f"  {i}: {c}")

    return total > 20  # Should have baseline + mocks


def test_nurse_rostering():
    print("\n" + "=" * 70)
    print("TESTING NURSE ROSTERING BENCHMARK")
    print("=" * 70)
    inst, orc = construct_nurse_rostering(shifts_per_day=3, num_days=7, num_nurses=8,
                                          nurses_per_shift=2, max_workdays=5)
    total = len(orc.constraints)
    print(f"Total constraints: {total}")

    # Check for mock constraints (nurse <= 6 constraints, etc.)
    nurse_limit_mocks = [c for c in orc.constraints if '<=' in str(c) and 'var[0' in str(c)]
    alldiff_mock = [c for c in orc.constraints if 'alldifferent' in str(c).lower()]

    print(f"Nurse limit mocks: {len(nurse_limit_mocks)} constraints")
    print(f"AllDifferent constraints: {len(alldiff_mock)}")

    if nurse_limit_mocks:
        print(f"  Sample: {nurse_limit_mocks[0]}")

    return len(nurse_limit_mocks) > 0


def main():
    print("\n" + "=" * 70)
    print("MOCK CONSTRAINT VERIFICATION TEST SUITE")
    print("=" * 70)

    results = {
        "Sudoku": test_sudoku(),
        "UEFA": test_uefa(),
        "VM Allocation": test_vm_allocation(),
        "Exam Timetabling": test_exam_timetabling(),
        "Nurse Rostering": test_nurse_rostering()
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for benchmark, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{benchmark:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + ("=" * 70))
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
