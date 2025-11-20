"""
Generate ML Training Dataset and Train Constraint Classifier

This script generates a comprehensive dataset from all benchmarks in benchmarks_global
and trains the ML model for belief initialization in constraint learning.

Usage:
    python generate_and_train_ml_model.py
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.globalconstraints import AllDifferent

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

from probabilistic_belief_initialization import ConstraintFeatureExtractor, SyntheticDataGenerator, ConstraintClassifier

# Import all benchmarks from benchmarks_global
from benchmarks_global import (
    construct_sudoku,
    construct_sudoku_greater_than,
    construct_jsudoku, construct_jsudoku_4x4, construct_jsudoku_6x6,
    construct_latin_square, construct_latin_square_4x4, construct_latin_square_6x6, construct_latin_square_9x9,
    construct_graph_coloring, construct_graph_coloring_queen5, construct_graph_coloring_register, construct_graph_coloring_scheduling,
    construct_nurse_rostering,
    construct_examtt_simple, construct_examtt_variant1, construct_examtt_variant2,
    construct_uefa,
    construct_vm_allocation
)


class EnhancedDataGenerator(SyntheticDataGenerator):
    """
    Enhanced data generator that uses all benchmarks from benchmarks_global.
    """
    
    def _get_benchmark_configs(self) -> List[Dict[str, Any]]:
        """Define comprehensive benchmark configurations from benchmarks_global."""
        configs = []
        
        print("Configuring benchmarks from benchmarks_global...")
        
        # 1. Sudoku variants (standard)
        print("  - Adding Sudoku variants...")
        for size in [4, 6, 9]:
            block_size = int(np.sqrt(size))
            configs.append({
                'name': f'sudoku_{size}x{size}',
                'func': construct_sudoku,
                'params': {'block_size_row': block_size, 'block_size_col': block_size, 'grid_size': size},
                'num_solutions': 100 if size <= 6 else 50
            })
        
        # 2. Sudoku with Greater Than constraints
        print("  - Adding Sudoku Greater Than variants...")
        for size in [4, 6]:
            block_size = int(np.sqrt(size))
            configs.append({
                'name': f'sudoku_greater_than_{size}x{size}',
                'func': construct_sudoku_greater_than,
                'params': {'block_size_row': block_size, 'block_size_col': block_size, 'grid_size': size},
                'num_solutions': 80 if size <= 4 else 50
            })
        
        # 3. Latin Squares
        print("  - Adding Latin Square variants...")
        # Using specific constructor functions
        configs.append({
            'name': 'latin_square_4x4',
            'func': construct_latin_square_4x4,
            'params': {},
            'num_solutions': 100
        })
        configs.append({
            'name': 'latin_square_6x6',
            'func': construct_latin_square_6x6,
            'params': {},
            'num_solutions': 80
        })
        configs.append({
            'name': 'latin_square_9x9',
            'func': construct_latin_square_9x9,
            'params': {},
            'num_solutions': 50
        })
        
        # Also add generic versions
        for size in [4, 6, 9]:
            configs.append({
                'name': f'latin_square_{size}x{size}',
                'func': construct_latin_square,
                'params': {'n': size},
                'num_solutions': 100 if size <= 6 else 50
            })
        
        # 4. JSudoku (Jigsaw Sudoku)
        print("  - Adding JSudoku variants...")
        configs.append({
            'name': 'jsudoku_4x4',
            'func': construct_jsudoku_4x4,
            'params': {},
            'num_solutions': 100
        })
        configs.append({
            'name': 'jsudoku_6x6',
            'func': construct_jsudoku_6x6,
            'params': {},
            'num_solutions': 80
        })
        # Generic JSudoku
        configs.append({
            'name': 'jsudoku_generic',
            'func': construct_jsudoku,
            'params': {},
            'num_solutions': 80
        })
        
        # 5. Graph Coloring
        print("  - Adding Graph Coloring variants...")
        # Specific constructors
        configs.append({
            'name': 'graph_coloring_queen_5x5',
            'func': construct_graph_coloring_queen5,
            'params': {},
            'num_solutions': 100
        })
        configs.append({
            'name': 'graph_coloring_register',
            'func': construct_graph_coloring_register,
            'params': {},
            'num_solutions': 80
        })
        configs.append({
            'name': 'graph_coloring_scheduling',
            'func': construct_graph_coloring_scheduling,
            'params': {},
            'num_solutions': 80
        })
        # Generic graph coloring with different graph types
        for graph_type in ['queen_5x5', 'queen_6x6', 'register', 'scheduling']:
            configs.append({
                'name': f'graph_coloring_{graph_type}',
                'func': construct_graph_coloring,
                'params': {'graph_type': graph_type},
                'num_solutions': 100 if 'queen_5' in graph_type else 80
            })
        
        # 6. Nurse Rostering
        print("  - Adding Nurse Rostering variants...")
        for num_nurses in [6, 8, 10]:
            configs.append({
                'name': f'nurse_rostering_n{num_nurses}',
                'func': construct_nurse_rostering,
                'params': {'num_nurses': num_nurses, 'num_days': 5, 'shifts_per_day': 2},
                'num_solutions': 50
            })
        
        # 7. Exam Timetabling
        print("  - Adding Exam Timetabling variants...")
        configs.append({
            'name': 'exam_timetabling_simple',
            'func': construct_examtt_simple,
            'params': {'nsemesters': 6, 'courses_per_semester': 4, 'slots_per_day': 9, 'days_for_exams': 10},
            'num_solutions': 30
        })
        configs.append({
            'name': 'exam_timetabling_variant1',
            'func': construct_examtt_variant1,
            'params': {'nsemesters': 6, 'courses_per_semester': 4, 'slots_per_day': 9, 'days_for_exams': 10},
            'num_solutions': 30
        })
        configs.append({
            'name': 'exam_timetabling_variant2',
            'func': construct_examtt_variant2,
            'params': {'nsemesters': 6, 'courses_per_semester': 4, 'slots_per_day': 9, 'days_for_exams': 10},
            'num_solutions': 30
        })
        
        # 8. UEFA (Soccer Tournament Scheduling)
        print("  - Adding UEFA variant...")
        configs.append({
            'name': 'uefa',
            'func': construct_uefa,
            'params': {},
            'num_solutions': 30
        })
        
        # 9. VM Allocation
        print("  - Adding VM Allocation variant...")
        configs.append({
            'name': 'vm_allocation',
            'func': construct_vm_allocation,
            'params': {},
            'num_solutions': 50
        })
        
        print(f"\nTotal benchmarks configured: {len(configs)}")
        return configs


def main():
    """
    Main function to generate dataset and train the ML model.
    """
    print("="*80)
    print(" ML TRAINING: DATASET GENERATION & MODEL TRAINING")
    print("="*80)
    print("\nThis script will:")
    print("  1. Generate training data from all benchmarks in benchmarks_global")
    print("  2. Extract features from true and overfitted constraints")
    print("  3. Train a Random Forest classifier with probability calibration")
    print("  4. Evaluate the model and save it for use in constraint learning")
    print("\n" + "="*80 + "\n")
    
    # =========================================================================
    # STEP 1: Generate Synthetic Training Data
    # =========================================================================
    print("STEP 1: GENERATING SYNTHETIC TRAINING DATA")
    print("-" * 80)
    
    data_generator = EnhancedDataGenerator(output_dir="ml_training_data")
    
    # Generate dataset (use all benchmarks, target 5000+ instances)
    # Note: Some benchmarks might fail, that's okay
    print("\nStarting dataset generation...")
    print("This may take 15-30 minutes depending on your system...\n")
    
    df, metadata = data_generator.generate_dataset(
        num_benchmarks=100,  # Use all configured benchmarks
        total_instances=5000
    )
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"\nTotal instances: {len(df)}")
    print(f"Benchmarks used: {len(metadata['benchmarks'])}")
    print(f"True constraints: {metadata['total_positive']}")
    print(f"Overfitted constraints: {metadata['total_negative']}")
    print(f"Class balance: {100*metadata['total_positive']/len(df):.1f}% positive")
    
    # =========================================================================
    # STEP 2: Prepare Data for Training
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: PREPARING DATA FOR TRAINING")
    print("="*80)
    
    # Remove non-feature columns
    feature_columns = [col for col in df.columns if col not in ['label', 'benchmark']]
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Class distribution:")
    print(f"  - Overfitted (0): {np.sum(y == 0)} ({100*np.mean(y == 0):.1f}%)")
    print(f"  - True constraints (1): {np.sum(y == 1)} ({100*np.mean(y == 1):.1f}%)")
    
    # Check for data quality
    if len(df) < 100:
        print("\n⚠ WARNING: Very few instances generated!")
        print("Some benchmarks may have failed. The model may not be well-trained.")
    
    # Split into train (60%), validation (20%), test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  - Train set: {X_train.shape[0]} instances ({100*len(X_train)/len(X):.1f}%)")
    print(f"  - Validation set: {X_val.shape[0]} instances ({100*len(X_val)/len(X):.1f}%)")
    print(f"  - Test set: {X_test.shape[0]} instances ({100*len(X_test)/len(X):.1f}%)")
    
    # =========================================================================
    # STEP 3: Train the Classifier
    # =========================================================================
    classifier = ConstraintClassifier(output_dir="ml_models")
    classifier.feature_names = feature_columns
    classifier.train(X_train, y_train, X_val, y_val)
    
    # =========================================================================
    # STEP 4: Evaluate on Test Set
    # =========================================================================
    metrics = classifier.evaluate(X_test, y_test)
    
    # =========================================================================
    # STEP 5: Analyze Feature Importance
    # =========================================================================
    importance_df = classifier.analyze_feature_importance(top_n=20)
    
    # =========================================================================
    # STEP 6: Save the Trained Model
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVING TRAINED MODEL")
    print("="*80)
    classifier.save_model('constraint_classifier_calibrated.pkl')
    
    # Save summary report
    report = {
        'dataset': {
            'total_instances': len(df),
            'num_features': len(feature_columns),
            'num_benchmarks': len(metadata['benchmarks']),
            'benchmarks': metadata['benchmarks'],
            'class_distribution': {
                'overfitted': int(np.sum(y == 0)),
                'true_constraints': int(np.sum(y == 1))
            }
        },
        'split': {
            'train': int(X_train.shape[0]),
            'validation': int(X_val.shape[0]),
            'test': int(X_test.shape[0])
        },
        'model': {
            'type': 'Random Forest',
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_leaf': 5,
            'calibration': 'isotonic'
        },
        'performance': {
            'test_accuracy': float(metrics['accuracy']),
            'test_precision': float(metrics['precision']),
            'test_recall': float(metrics['recall']),
            'test_f1': float(metrics['f1']),
            'test_roc_auc': float(metrics['roc_auc'])
        },
        'top_features': [
            {'name': row['feature'], 'importance': float(row['importance'])}
            for _, row in importance_df.head(15).iterrows()
        ],
        'benchmark_stats': metadata['benchmark_stats']
    }
    
    with open('ml_models/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTraining report saved to ml_models/training_report.json")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\n[SUCCESS] Files generated:")
    print("  - ml_training_data/training_data.csv")
    print("  - ml_training_data/metadata.json")
    print("  - ml_models/constraint_classifier_calibrated.pkl")
    print("  - ml_models/feature_importance.csv")
    print("  - ml_models/training_report.json")
    
    print(f"\n[SUCCESS] Model Performance:")
    print(f"  - Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  - Precision: {metrics['precision']:.2%}")
    print(f"  - Recall:    {metrics['recall']:.2%}")
    print(f"  - F1 Score:  {metrics['f1']:.2%}")
    print(f"  - ROC AUC:   {metrics['roc_auc']:.2%}")
    
    if metrics['accuracy'] >= 0.85:
        print("\n[SUCCESS] Model quality: EXCELLENT - Ready for production use!")
    elif metrics['accuracy'] >= 0.75:
        print("\n[SUCCESS] Model quality: GOOD - Should work well for most cases")
    else:
        print("\n[WARNING] Model quality: MODERATE - Consider generating more training data")
    
    print("\n[INFO] Usage:")
    print("  from ml_belief_scorer import MLBeliefScorer")
    print("  scorer = MLBeliefScorer()")
    print("  probability = scorer.score_constraint(candidate, variables)")
    
    print("\n[INFO] Test the model:")
    print("  python ml_belief_scorer.py")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user.\n")
    except Exception as e:
        print(f"\n\n[ERROR] Error during training: {e}\n")
        import traceback
        traceback.print_exc()

