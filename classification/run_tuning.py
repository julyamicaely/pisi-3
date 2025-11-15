"""
Runner script to execute the full Random Forest tuning flow.

Usage:
    python run_tuning.py

This will:
 - Reuse preprocessing from preprocess_data
 - Run tuning (RandomizedSearchCV)
 - Persist best model to classification/models/random_forest_model.joblib
 - Save tuning results to classification/reports/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.model_random_forest import run_rf_tuning
from classification.preprocess_data import preprocess_data
import os


def main():
    print('Starting Random Forest tuning runner...')
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess_data()
    # Run a reasonable default tuning - set n_iter lower for quick runs
    best, info = run_rf_tuning(X_train, y_train, feature_names, random_state=42, n_iter=40, cv_folds=5)
    print('Tuning completed. Best params:')
    print(info['best_params'])


if __name__ == '__main__':
    main()
