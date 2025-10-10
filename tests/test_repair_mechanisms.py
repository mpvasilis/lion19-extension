"""
Compare counterexample repair vs culprit score mechanisms.
Run HCAR-Advanced twice: once with counterexample repair, once without.
"""
import sys
import json
from run_hcar_experiments import run_single_experiment, create_method_config

def test_repair_mechanisms(benchmark_name="VM_Allocation"):
    """Compare counterexample repair vs culprit scores."""

    print(f"\n{'='*70}")
    print(f"Testing Repair Mechanisms on {benchmark_name}")
    print(f"{'='*70}\n")

    # Test 1: With counterexample repair
    print("\n--- Test 1: Counterexample-Driven Repair ---")
    config_cex = create_method_config("HCAR-Advanced")
    config_cex.use_counterexample_repair = True
    config_cex.inject_overfitted = True

    result_cex = run_single_experiment(
        benchmark_name=benchmark_name,
        method_name="HCAR-Advanced-CEX",
        config=config_cex,
        num_runs=1,
        use_global_constraints=True
    )

    # Test 2: With culprit scores (no counterexample repair)
    print("\n--- Test 2: Culprit Score-Based Repair ---")
    config_culprit = create_method_config("HCAR-Advanced")
    config_culprit.use_counterexample_repair = False
    config_culprit.inject_overfitted = True

    result_culprit = run_single_experiment(
        benchmark_name=benchmark_name,
        method_name="HCAR-Advanced-Culprit",
        config=config_culprit,
        num_runs=1,
        use_global_constraints=True
    )

    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Metric':<30} {'CEX Repair':<20} {'Culprit Score':<20}")
    print("-" * 70)

    if result_cex and result_culprit:
        metrics = ['s_precision', 's_recall', 'queries_phase2', 'queries_phase3', 'queries_total', 'time']
        labels = {
            's_precision': 'S-Precision (%)',
            's_recall': 'S-Recall (%)',
            'queries_phase2': 'Phase 2 Queries',
            'queries_phase3': 'Phase 3 Queries',
            'queries_total': 'Total Queries',
            'time': 'Time (s)'
        }

        for metric in metrics:
            val_cex = result_cex.get(metric, 0)
            val_culprit = result_culprit.get(metric, 0)

            if metric in ['s_precision', 's_recall']:
                val_cex *= 100
                val_culprit *= 100

            label = labels.get(metric, metric)
            diff = val_cex - val_culprit

            if metric in ['queries_phase2', 'queries_phase3', 'queries_total', 'time']:
                diff_str = f"({diff:+.1f})" if diff != 0 else ""
                print(f"{label:<30} {val_cex:<20.1f} {val_culprit:<20.1f} {diff_str}")
            else:
                print(f"{label:<30} {val_cex:<20.1f} {val_culprit:<20.1f}")

    print("\n" + "="*70)

    # Analyze which mechanism is better
    if result_cex and result_culprit:
        cex_queries = result_cex.get('queries_total', 0)
        culprit_queries = result_culprit.get('queries_total', 0)

        if cex_queries < culprit_queries:
            savings = culprit_queries - cex_queries
            pct = (savings / culprit_queries) * 100 if culprit_queries > 0 else 0
            print(f"\nCounterexample repair SAVES {savings:.0f} queries ({pct:.1f}%)")
        elif cex_queries > culprit_queries:
            extra = cex_queries - culprit_queries
            pct = (extra / cex_queries) * 100 if cex_queries > 0 else 0
            print(f"\nCulprit scores SAVES {extra:.0f} queries ({pct:.1f}%)")
        else:
            print(f"\nBoth mechanisms use SAME number of queries")

    print("="*70 + "\n")

if __name__ == "__main__":
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "VM_Allocation"
    test_repair_mechanisms(benchmark)
