"""
SHAP interpretability utilities for the Random Forest model.

Generates global (summary, beeswarm, bar) and local (decision/force) plots
and saves them under classification/reports/shap/.

Functions are idempotent and re-runnable.
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import shap
except Exception:
    shap = None

from classification.preprocess_data import load_and_preprocess_data


def ensure_shap_installed():
    if shap is None:
        raise ImportError("shap library is required. Install with `pip install shap`")


def generate_shap_reports(model_path=None, sample_index=0, output_dir=None, max_samples=2000):
    """
    Load the Random Forest model and generate SHAP explainability reports.

    Args:
        model_path: path to persisted random_forest_model.joblib (if None, default path used)
        sample_index: index of sample to generate local explanation for
        output_dir: base directory to save reports (defaults to classification/reports/shap/)
        max_samples: maximum number of samples to use for SHAP (default 2000 for performance)
    """
    ensure_shap_installed()

    base = Path(__file__).parent
    if model_path is None:
        model_path = base / 'models' / 'random_forest_model.joblib'
    if output_dir is None:
        output_dir = base / 'reports' / 'shap'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Carregando modelo de {model_path}...")
    model = joblib.load(model_path)
    print(f"   ℹ️  Modelo: {type(model).__name__}")
    print(f"   ℹ️  N° de árvores: {model.n_estimators}")
    
    print("\n📊 Carregando dados pré-processados...")
    X_scaled, X_original, y, feature_names = load_and_preprocess_data()
    print(f"   ℹ️  Total de amostras: {len(X_scaled)}")
    print(f"   ℹ️  N° de features: {len(feature_names)}")
    print(f"   ℹ️  Features: {', '.join(feature_names)}")

    # Use uma amostra para SHAP (performance)
    if len(X_scaled) > max_samples:
        print(f"\n⚠️  Dataset grande ({len(X_scaled)} amostras). Usando amostra de {max_samples} para SHAP...")
        sample_indices = np.random.choice(len(X_scaled), size=max_samples, replace=False)
        X_sample = X_scaled.iloc[sample_indices].copy()
    else:
        X_sample = X_scaled.copy()
    
    print(f"\n📊 Calculando SHAP values para {len(X_sample)} amostras...")
    print(f"   ℹ️  Shape dos dados: {X_sample.shape}")
    explainer = shap.TreeExplainer(model)
    
    # Para Random Forest binário, shap_values pode retornar:
    # - Lista [class_0, class_1] com arrays (n_samples, n_features)
    # - Array 3D (n_samples, n_features, n_classes)
    shap_values_raw = explainer.shap_values(X_sample)
    
    # Detectar formato e extrair classe positiva
    if isinstance(shap_values_raw, list):
        print(f"   ℹ️  Formato: Lista com {len(shap_values_raw)} classes")
        print(f"   ℹ️  Shape SHAP values [classe 0]: {shap_values_raw[0].shape}")
        print(f"   ℹ️  Shape SHAP values [classe 1]: {shap_values_raw[1].shape}")
        shap_values = shap_values_raw[1]  # Classe 1 (com doença)
    elif len(shap_values_raw.shape) == 3:
        print(f"   ℹ️  Formato: Array 3D com shape {shap_values_raw.shape}")
        print(f"   ℹ️  Extraindo classe positiva (índice [:, :, 1])")
        shap_values = shap_values_raw[:, :, 1]  # Classe 1 (com doença)
    else:
        print(f"   ℹ️  Formato: Array 2D com shape {shap_values_raw.shape}")
        shap_values = shap_values_raw
    
    print(f"   ℹ️  Shape final SHAP values: {shap_values.shape}")
    print(f"   ✅ Todas as {shap_values.shape[1]} features serão exibidas nos gráficos!")

    print("\n📊 Gerando visualizações SHAP...")
    
    # Global: summary plot (dot)
    print("   - Summary plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    p = output_dir / 'shap_summary.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {p}")

    # Global: bar plot
    print("   - Bar plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    p = output_dir / 'shap_bar.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {p}")

    # Global: waterfall plot (top features para uma amostra específica)
    print(f"   - Waterfall plot (amostra {sample_index})...")
    try:
        idx = int(sample_index)
        if idx < len(X_sample):
            # Para array 3D, expected_value também pode ser array
            if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > 1:
                expected_val = float(explainer.expected_value[1])
            else:
                expected_val = float(explainer.expected_value)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[idx],
                    base_values=expected_val,
                    data=X_sample.iloc[idx].values,
                    feature_names=feature_names
                ),
                show=False
            )
            plt.tight_layout()
            p = output_dir / f'shap_waterfall_{idx}.png'
            plt.savefig(p, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ {p}")
    except Exception as e:
        print(f"   ⚠️  Waterfall plot não gerado: {e}")

    # Local: force plot para um exemplo (salvar como HTML)
    print(f"   - Force plot (amostra {sample_index})...")
    try:
        idx = int(sample_index)
        if idx < len(X_sample):
            # Para array 3D, expected_value também pode ser array
            if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > 1:
                expected_val = float(explainer.expected_value[1])
            else:
                expected_val = float(explainer.expected_value)
            
            fp = shap.force_plot(
                expected_val,
                shap_values[idx],
                X_sample.iloc[idx],
                feature_names=feature_names,
                matplotlib=False
            )
            html_path = output_dir / f'shap_force_{idx}.html'
            shap.save_html(str(html_path), fp)
            print(f"   ✅ {html_path}")
    except Exception as e:
        print(f"   ⚠️  Force plot não gerado: {e}")

    print(f"\n✅ Todos os relatórios SHAP salvos em: {output_dir}")
    return output_dir


if __name__ == '__main__':
    print('Generating SHAP reports...')
    out = generate_shap_reports()
    print('Saved SHAP reports to', out)
