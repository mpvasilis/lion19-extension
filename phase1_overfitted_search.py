import argparse
import os
import pickle
import time
from typing import Dict, List, Optional

from cpmpy import cpm_array

from phase1_passive_learning import (
    construct_instance,
    generate_positive_examples,
    extract_alldifferent_constraints,
    extract_grid_info,
    discover_non_implied_pairs,
    generate_overfitted_alldifferent,
)


DEFAULT_BENCHMARKS = [
    "sudoku",
    "sudoku_gt",
    "examtt",
    "examtt_v1",
    "examtt_v2",
    "nurse",
    "uefa",
    "graph_coloring_register",
    "graph_coloring_scheduling",
    "latin_square",
    "jsudoku",
]


def constraint_to_string_list(constraints: List) -> List[str]:
    """Helper to convert constraints to string form for serialization."""
    return [str(c) for c in constraints]


def search_overfitted_alldifferent(
    benchmark: str,
    num_examples: int = 5,
    target_overfitted: int = 5,
    max_attempts: int = 5000,
    seed: Optional[int] = None,
) -> Dict:
    """
    Run a passive overfitted-search routine for a single benchmark.

    Args:
        benchmark: Benchmark identifier understood by construct_instance.
        num_examples: Number of positive examples to sample from the oracle.
        target_overfitted: Desired number of non-implied AllDifferent constraints to produce.
        max_attempts: Attempt budget passed to the generation routine.
        seed: Optional random seed to improve reproducibility.

    Returns:
        A dictionary with detailed statistics about the search.
    """
    start_time = time.time()

    instance_data = construct_instance(benchmark)
    if len(instance_data) == 3:
        instance, oracle, _ = instance_data
    else:
        instance, oracle = instance_data

    oracle.variables_list = cpm_array(instance.X)

    if seed is not None:
        import random

        random.seed(seed)

    positive_examples = generate_positive_examples(oracle, instance.X, count=num_examples)

    targets = extract_alldifferent_constraints(oracle)
    grid_info = extract_grid_info(instance.X)
    pair_seeds = discover_non_implied_pairs(
        instance.X, positive_examples, targets, grid_info=grid_info
    )

    overfitted = generate_overfitted_alldifferent(
        instance.X,
        positive_examples,
        targets,
        count=target_overfitted,
        max_attempts=max_attempts,
        grid_info=grid_info,
        pair_seeds=pair_seeds,
    )

    duration = time.time() - start_time

    return {
        "benchmark": benchmark,
        "num_examples_requested": num_examples,
        "positive_examples_generated": len(positive_examples),
        "target_constraints": len(targets),
        "overfitted_requested": target_overfitted,
        "overfitted_generated": len(overfitted),
        "generation_duration": duration,
        "pair_seed_count": len(pair_seeds),
        "grid_blocks": len(grid_info.get("blocks", {})) if grid_info else 0,
        "overfitted_constraints": constraint_to_string_list(overfitted),
        "target_constraints_str": constraint_to_string_list(targets),
    }


def run_batch_search(
    benchmarks: List[str],
    num_examples: int,
    target_overfitted: int,
    max_attempts: int,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Execute the overfitted search routine for each benchmark in the list."""
    results = []
    for benchmark in benchmarks:
        print(f"\n=== Benchmark: {benchmark} ===")
        result = search_overfitted_alldifferent(
            benchmark=benchmark,
            num_examples=num_examples,
            target_overfitted=target_overfitted,
            max_attempts=max_attempts,
            seed=seed,
        )
        print(
            f"  Generated {result['overfitted_generated']}/{result['overfitted_requested']} "
            f"non-implied overfitted constraints "
            f"(pair seeds: {result['pair_seed_count']}, blocks: {result['grid_blocks']})"
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Search for non-implied overfitted AllDifferent constraints."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="sudoku_gt",
        help="Comma-separated list of benchmarks or 'all' (default: sudoku_gt)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of positive examples to sample (default: 5)",
    )
    parser.add_argument(
        "--target_overfitted",
        type=int,
        default=5,
        help="Desired number of overfitted constraints to generate (default: 5)",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5000,
        help="Maximum generation attempts (default: 5000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory to store pickle summaries per benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.benchmarks.lower() == "all":
        benchmark_list = DEFAULT_BENCHMARKS
    else:
        benchmark_list = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        if not benchmark_list:
            raise ValueError("No valid benchmarks supplied.")

    results = run_batch_search(
        benchmarks=benchmark_list,
        num_examples=args.num_examples,
        target_overfitted=args.target_overfitted,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for result in results:
            out_path = os.path.join(
                args.output_dir, f"{result['benchmark']}_overfitted_search.pkl"
            )
            with open(out_path, "wb") as handle:
                pickle.dump(result, handle)
            print(f"  Saved summary to: {out_path}")


if __name__ == "__main__":
    main()

