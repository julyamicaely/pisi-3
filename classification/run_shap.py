"""
Runner script to generate SHAP interpretability reports for Random Forest.

Usage:
    python run_shap.py

This will load the persisted Random Forest model and produce SHAP visuals
in classification/reports/shap/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.shap_random_forest import generate_shap_reports


def main():
    print('Running SHAP report generation...')
    out = generate_shap_reports()
    print('SHAP outputs saved in:', out)


if __name__ == '__main__':
    main()
