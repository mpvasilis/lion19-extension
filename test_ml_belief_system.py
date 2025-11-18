"""
Test script for ML Belief Initialization System

This script validates that the ML belief initialization system is properly set up
and can perform basic operations without errors.
"""

import sys
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("="*70)
    print("TEST 1: Validating Imports")
    print("="*70)
    
    try:
        print("Importing cpmpy...", end=" ")
        import cpmpy as cp
        print("✓")
        
        print("Importing sklearn...", end=" ")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        print("✓")
        
        print("Importing pandas...", end=" ")
        import pandas as pd
        print("✓")
        
        print("Importing custom modules...", end=" ")
        from probabilistic_belief_initialization import (
            ConstraintFeatureExtractor,
            SyntheticDataGenerator,
            ConstraintClassifier
        )
        print("✓")
        
        print("Importing ml_belief_scorer...", end=" ")
        from ml_belief_scorer import MLBeliefScorer
        print("✓")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_feature_extraction():
    """Test feature extraction on a simple problem."""
    print("="*70)
    print("TEST 2: Feature Extraction")
    print("="*70)
    
    try:
        import cpmpy as cp
        from probabilistic_belief_initialization import ConstraintFeatureExtractor
        
        # Create a simple 4x4 grid
        print("Creating 4x4 grid...", end=" ")
        grid = cp.intvar(1, 4, shape=(4, 4), name="grid")
        print("✓")
        
        # Create some AllDifferent constraints
        print("Creating AllDifferent constraints...", end=" ")
        row_constraint = cp.AllDifferent(grid[0, :])
        col_constraint = cp.AllDifferent(grid[:, 0])
        diag_constraint = cp.AllDifferent([grid[i, i] for i in range(4)])
        print("✓")
        
        # Extract features
        print("Extracting features from row constraint...", end=" ")
        extractor = ConstraintFeatureExtractor(grid)
        features_row = extractor.extract_features(row_constraint)
        print(f"✓ ({len(features_row)} features)")
        
        print("Extracting features from column constraint...", end=" ")
        features_col = extractor.extract_features(col_constraint)
        print(f"✓ ({len(features_col)} features)")
        
        print("Extracting features from diagonal constraint...", end=" ")
        features_diag = extractor.extract_features(diag_constraint)
        print(f"✓ ({len(features_diag)} features)")
        
        # Validate feature values
        print("\nValidating feature values:")
        print(f"  Row constraint - scope_size: {features_row['scope_size']}")
        print(f"  Row constraint - is_complete_row: {features_row['is_complete_row']}")
        print(f"  Col constraint - is_complete_col: {features_col['is_complete_col']}")
        print(f"  Diag constraint - is_main_diagonal: {features_diag['is_main_diagonal']}")
        
        # Basic sanity checks
        assert features_row['scope_size'] == 4, "Row scope size should be 4"
        assert features_row['is_complete_row'] == 1.0, "Row should be detected as complete row"
        assert features_col['is_complete_col'] == 1.0, "Column should be detected as complete column"
        assert features_diag['is_main_diagonal'] == 1.0, "Diagonal should be detected"
        
        print("\n✅ Feature extraction working correctly!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Feature extraction failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark_loading():
    """Test that benchmark modules can be loaded."""
    print("="*70)
    print("TEST 3: Benchmark Loading")
    print("="*70)
    
    benchmarks_to_test = [
        ('sudoku', 'construct_sudoku', {'block_size_row': 2, 'block_size_col': 2, 'grid_size': 4}),
        ('latin_square', 'construct_latin_square', {'n': 4}),
        ('graph_coloring', 'construct_graph_coloring', {'graph_type': 'register'}),
    ]
    
    success_count = 0
    
    for module_name, func_name, params in benchmarks_to_test:
        try:
            print(f"Loading {module_name}.{func_name}...", end=" ")
            module = __import__(f'benchmarks_global.{module_name}', fromlist=[func_name])
            func = getattr(module, func_name)
            
            instance, oracle, overfitted = func(**params)
            
            print(f"✓ (vars: {np.prod(instance.variables.shape) if hasattr(instance.variables, 'shape') else len(instance.variables)}, "
                  f"true: {len(oracle.C_T)}, overfitted: {len(overfitted)})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print(f"\n✅ Successfully loaded {success_count}/{len(benchmarks_to_test)} benchmarks!\n")
    return success_count == len(benchmarks_to_test)


def test_data_generation_small():
    """Test data generation on a small scale."""
    print("="*70)
    print("TEST 4: Small-Scale Data Generation")
    print("="*70)
    
    try:
        from probabilistic_belief_initialization import SyntheticDataGenerator
        
        print("Creating data generator...")
        generator = SyntheticDataGenerator(output_dir="test_ml_data")
        
        # Test on just one benchmark
        print("Testing single benchmark processing...")
        from benchmarks_global.sudoku import construct_sudoku
        
        config = {
            'name': 'test_sudoku_4x4',
            'func': construct_sudoku,
            'params': {'block_size_row': 2, 'block_size_col': 2, 'grid_size': 4},
            'num_solutions': 5  # Very small for testing
        }
        
        print(f"Processing {config['name']}...", end=" ")
        benchmark_data = generator._process_benchmark(config)
        print(f"✓ ({len(benchmark_data)} instances)")
        
        if benchmark_data:
            print(f"\nSample instance features:")
            sample = benchmark_data[0]
            for key in list(sample.keys())[:10]:  # Show first 10 features
                print(f"  {key}: {sample[key]}")
        
        print("\n✅ Data generation works!\n")
        
        # Cleanup
        import shutil
        test_dir = Path("test_ml_data")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Data generation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model_structure():
    """Test that the model can be instantiated and has correct structure."""
    print("="*70)
    print("TEST 5: Model Structure")
    print("="*70)
    
    try:
        from probabilistic_belief_initialization import ConstraintClassifier
        
        print("Creating classifier...", end=" ")
        classifier = ConstraintClassifier(output_dir="test_ml_models")
        print("✓")
        
        print(f"Model type: {type(classifier.model).__name__}")
        print(f"Model parameters:")
        print(f"  n_estimators: {classifier.model.n_estimators}")
        print(f"  max_depth: {classifier.model.max_depth}")
        print(f"  min_samples_leaf: {classifier.model.min_samples_leaf}")
        print(f"  criterion: {classifier.model.criterion}")
        
        # Validate parameters match paper specification
        assert classifier.model.n_estimators == 100, "Should have 100 trees"
        assert classifier.model.max_depth == 10, "Should have max depth 10"
        assert classifier.model.min_samples_leaf == 5, "Should have min samples leaf 5"
        assert classifier.model.criterion == 'gini', "Should use Gini criterion"
        
        print("\n✅ Model structure correct!\n")
        
        # Cleanup
        import shutil
        test_dir = Path("test_ml_models")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model structure test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_ml_scorer_structure():
    """Test that MLBeliefScorer has correct interface."""
    print("="*70)
    print("TEST 6: ML Belief Scorer Structure")
    print("="*70)
    
    try:
        from ml_belief_scorer import MLBeliefScorer
        
        print("Checking MLBeliefScorer class structure...")
        
        # Check that required methods exist
        required_methods = [
            'score_constraint',
            'score_candidates',
            'get_ranked_candidates',
            'initialize_beliefs',
            'get_feature_importance',
            'explain_score'
        ]
        
        for method in required_methods:
            print(f"  Checking method '{method}'...", end=" ")
            assert hasattr(MLBeliefScorer, method), f"Missing method: {method}"
            print("✓")
        
        print("\n✅ ML Belief Scorer structure correct!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ML Belief Scorer test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ML BELIEF INITIALIZATION SYSTEM - VALIDATION TESTS")
    print("="*70 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Feature Extraction", test_feature_extraction),
        ("Benchmark Loading", test_benchmark_loading),
        ("Data Generation", test_data_generation_small),
        ("Model Structure", test_model_structure),
        ("ML Scorer Structure", test_ml_scorer_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The ML Belief Initialization system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python probabilistic_belief_initialization.py")
        print("   (This will train the model, takes 10-30 minutes)")
        print("2. Run: python ml_belief_scorer.py")
        print("   (This will demonstrate the trained model)")
        print("3. Integrate with HCAR pipeline using MLBeliefScorer class")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

