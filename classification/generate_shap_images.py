"""
🎨 Gerador de Imagens SHAP - Random Forest

Este script gera imagens SHAP uma única vez e salva em arquivos estáticos.
O dashboard depois carrega essas imagens, evitando cálculo pesado toda vez.

Uso:
    python -m classification.generate_shap_images
    
    # Com número customizado de amostras:
    python -m classification.generate_shap_images --samples 500
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_pipeline_and_data():
    """Carrega pipeline e dados para análise SHAP."""
    base_path = Path(__file__).parent
    
    # Carregar pipeline
    pipeline_path = base_path / 'models' / 'random_forest_pipeline.joblib'
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline não encontrado: {pipeline_path}")
    
    print(f"📦 Carregando pipeline de: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    
    # Extrair modelo do pipeline
    model = pipeline.named_steps['classifier']
    scaler = pipeline.named_steps['scaler']
    
    # Carregar dados
    print("📊 Carregando dados de teste...")
    from classification.preprocess_data import load_and_preprocess_data
    X_scaled, X_original, y, feature_names = load_and_preprocess_data()
    
    print(f"✅ Pipeline e dados carregados!")
    print(f"   - Modelo: {type(model).__name__}")
    print(f"   - Features: {len(feature_names)}")
    print(f"   - Amostras: {len(X_scaled)}")
    
    return model, X_scaled, X_original, feature_names


def generate_shap_plots(model, X_scaled, X_original, feature_names, n_samples=500):
    """Gera e salva plots SHAP como imagens estáticas."""
    
    # Criar diretório para imagens
    output_dir = Path(__file__).parent.parent / 'dashboard' / 'assets' / 'shap_images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Gerando imagens SHAP com {n_samples} amostras...")
    print(f"📁 Salvando em: {output_dir}")
    
    # Selecionar amostra aleatória
    sample_size = min(n_samples, len(X_scaled))
    sample_indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
    X_sample = X_scaled.iloc[sample_indices]
    X_sample_original = X_original.iloc[sample_indices]
    
    # Calcular SHAP values
    print("🔮 Calculando SHAP values (pode demorar alguns minutos)...")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    
    # Extrair classe positiva (doença cardiovascular)
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
        base_value = explainer.expected_value[1]
    elif len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[:, :, 1]
        base_value = explainer.expected_value[1]
    else:
        shap_values = shap_values_raw
        base_value = explainer.expected_value
    
    print("✅ SHAP values calculados!")
    
    # Mapeamento de nomes para português
    feature_names_pt = {
        'gender': 'Gênero',
        'ap_hi': 'Pressão Sistólica',
        'ap_lo': 'Pressão Diastólica',
        'smoke': 'Tabagismo',
        'alco': 'Consumo Álcool',
        'active': 'Atividade Física',
        'age_years': 'Idade',
        'bmi': 'IMC',
        'cholesterol_high': 'Colesterol Alto',
        'gluc_high': 'Glicose Alta'
    }
    
    feature_names_translated = [feature_names_pt.get(f, f) for f in feature_names]
    
    # 1. Summary Plot (Beeswarm)
    print("\n📊 Gerando Summary Plot (importância global)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=feature_names_translated,
        show=False,
        max_display=10
    )
    plt.title('SHAP Summary Plot - Importância Global das Features', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvo: shap_summary_plot.png")
    
    # 2. Bar Plot (Feature Importance)
    print("📊 Gerando Bar Plot (importância média)...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, 
        X_sample,
        feature_names=feature_names_translated,
        plot_type="bar",
        show=False,
        max_display=10
    )
    plt.title('Importância Média das Features (|SHAP|)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_bar_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvo: shap_bar_plot.png")
    
    # 3. Dependence Plots para top 3 features
    print("📊 Gerando Dependence Plots (top 3 features)...")
    
    # Calcular importâncias médias
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    
    for idx in top_features_idx:
        feature_name = feature_names[idx]
        feature_name_pt = feature_names_pt.get(feature_name, feature_name)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx,
            shap_values,
            X_sample,
            feature_names=feature_names_translated,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature_name_pt}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        safe_name = feature_name.replace('_', '-')
        plt.savefig(output_dir / f'shap_dependence_{safe_name}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Salvo: shap_dependence_{safe_name}.png")
    
    # 4. Waterfall plot (exemplo individual)
    print("📊 Gerando Waterfall Plot (exemplo individual)...")
    
    # Pegar um exemplo com alta probabilidade
    sample_idx = np.argmax(np.abs(shap_values).sum(axis=1))
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=base_value,
            data=X_sample.iloc[sample_idx].values,
            feature_names=feature_names_translated
        ),
        show=False
    )
    plt.title('SHAP Waterfall Plot - Exemplo de Predição Individual', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_waterfall_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvo: shap_waterfall_example.png")
    
    print(f"\n🎉 Todas as imagens SHAP foram geradas com sucesso!")
    print(f"📁 Localização: {output_dir}")
    print(f"\n📝 Imagens geradas:")
    print(f"   - shap_summary_plot.png (beeswarm)")
    print(f"   - shap_bar_plot.png (importâncias)")
    print(f"   - shap_dependence_*.png (3 plots)")
    print(f"   - shap_waterfall_example.png")
    
    return output_dir


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Gera imagens SHAP estáticas')
    parser.add_argument('--samples', type=int, default=500,
                       help='Número de amostras para SHAP (padrão: 500)')
    args = parser.parse_args()
    
    print("="*80)
    print("🎨 GERADOR DE IMAGENS SHAP - RANDOM FOREST")
    print("="*80)
    
    # Carregar modelo e dados
    model, X_scaled, X_original, feature_names = load_pipeline_and_data()
    
    # Gerar plots
    output_dir = generate_shap_plots(
        model, X_scaled, X_original, feature_names, 
        n_samples=args.samples
    )
    
    print("\n" + "="*80)
    print("✅ CONCLUÍDO!")
    print("="*80)
    print("\nAgora você pode usar essas imagens no dashboard.")
    print("Elas serão carregadas instantaneamente sem recalcular SHAP!")


if __name__ == '__main__':
    main()
