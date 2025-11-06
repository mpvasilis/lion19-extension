"""Verify Phase 1 overfitted constraint counts for each benchmark configuration.

This test expects that the Solution Variance experiments have been executed and
that the corresponding Phase 1 pickle files exist under
`solution_variance_output/{benchmark}_sol{S}_overfitted{InvC}/{benchmark}_phase1.pkl`.

For each benchmark/solution-count pair, we ensure that the metadata records the
expected number of overfitted AllDifferent constraints (`InvC`).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest


# Mapping supplied by experiment specification (solutions -> expected InvC)
EXPECTED_OVERFITTED_COUNTS = {
    "sudoku": {
        2: 20,
        5: 8,
        10: 6,
        50: 3,
    },
    "sudoku_gt": {
        2: 17,
        5: 8,
        20: 5,
        200: 1,
    },
    "jsudoku": {
        2: 41,
        20: 41,
        200: 32,
        500: 25,
    },
    "examtt_v1": {
        2: 25,
        5: 20,
        10: 11,
        50: 5,
    },
    "examtt_v2": {
        2: 44,
        5: 28,
        10: 9,
        50: 4,
    },
}


def _phase1_pickle_path(base_dir: Path, benchmark: str, solutions: int, invc: int) -> Path:
    """Return the expected Phase 1 pickle path for the given configuration."""

    dir_name = f"{benchmark}_sol{solutions}_overfitted{invc}"
    return base_dir / dir_name / f"{benchmark}_phase1.pkl"


def _load_phase1_metadata(pickle_path: Path) -> dict:
    """Load metadata dictionary from a Phase 1 pickle file."""

    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):  # Defensive check: data should be a dict
        raise TypeError(f"Expected dict in {pickle_path}, found {type(data)}")
    metadata = data.get("metadata")
    if metadata is None:
        raise KeyError(f"Missing 'metadata' key in {pickle_path}")
    return metadata


# Dynamically expand to all (benchmark, solution-count, InvC) combinations
TEST_CASES = [
    (benchmark, solutions, expected_invc)
    for benchmark, mapping in EXPECTED_OVERFITTED_COUNTS.items()
    for solutions, expected_invc in mapping.items()
]


@pytest.mark.parametrize("benchmark,solutions,expected_invc", TEST_CASES)
def test_phase1_overfitted_counts(benchmark: str, solutions: int, expected_invc: int) -> None:
    """Ensure metadata records the expected number of overfitted constraints."""

    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / "solution_variance_output"

    pickle_path = _phase1_pickle_path(base_dir, benchmark, solutions, expected_invc)

    if not pickle_path.exists():
        pytest.skip(
            f"Phase 1 output missing for {benchmark} with {solutions} solutions at {pickle_path}. "
            "Run the solution variance experiments before executing this test."
        )

    metadata = _load_phase1_metadata(pickle_path)

    actual_invc = metadata.get("num_overfitted_alldiffs")
    assert actual_invc == expected_invc, (
        "Unexpected count of overfitted constraints for "
        f"{benchmark} (solutions={solutions}). "
        f"Expected {expected_invc}, found {actual_invc}."
    )

    # Sanity check: deduplicated count should never exceed the requested total
    dedup_key = "num_overfitted_alldiffs_dedup"
    if dedup_key in metadata:
        assert metadata[dedup_key] <= expected_invc, (
            f"Deduplicated overfitted count exceeds expected for {benchmark} "
            f"(solutions={solutions}): {metadata[dedup_key]} > {expected_invc}"
        )

