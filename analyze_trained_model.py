"""
Análise rápida do modelo treinado - Feature Importances e Calibração
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Carregar o pipeline
pipeline_path = Path("classification/models/random_forest_pipeline.joblib")
pipeline = joblib.load(pipeline_path)

print("=" * 80)
print("ANÁLISE DO MODELO TREINADO")
print("=" * 80)

# Acessar o Random Forest
rf_model = pipeline.named_steps['classifier']

print(f"\n📊 Hiperparâmetros do Modelo:")
print(f"  n_estimators: {rf_model.n_estimators}")
print(f"  max_depth: {rf_model.max_depth}")
print(f"  min_samples_leaf: {rf_model.min_samples_leaf}")
print(f"  max_features: {rf_model.max_features}")
print(f"  class_weight: {rf_model.class_weight}")

# Feature importances
feature_names = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'smoke', 'alco',
                 'active', 'cholesterol_high', 'gluc_high', 'gender']

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n🎯 Feature Importances (Top 10):")
for i in range(len(feature_names)):
    idx = indices[i]
    print(f"  {i+1}. {feature_names[idx]:20s}: {importances[idx]*100:6.2f}%")

# Verificar se há features sendo ignoradas
ignored_features = [feature_names[i] for i in range(len(feature_names)) 
                    if importances[i] < 0.01]
if ignored_features:
    print(f"\n⚠️  Features praticamente ignoradas (<1%):")
    for feat in ignored_features:
        print(f"   - {feat}")

# Soma das top 3 features
top3_sum = sum([importances[indices[i]] for i in range(3)])
print(f"\n📈 Concentração de Importância:")
print(f"  Top 3 features: {top3_sum*100:.2f}%")
print(f"  Outras 7 features: {(1-top3_sum)*100:.2f}%")

if top3_sum > 0.8:
    print("  ⚠️  PROBLEMA: Modelo muito dependente de poucas features!")
    print("     Outras features sendo quase ignoradas")

# Verificar distribuição de predições nas árvores
print(f"\n🌲 Análise das Árvores:")
print(f"  Número de árvores: {len(rf_model.estimators_)}")

# Pegar uma amostra de predições
from classification.preprocess_data import load_and_preprocess_data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Predições no conjunto de teste
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"\n📊 Distribuição de Predições (Test Set):")
print(f"  Mínimo: {y_pred_proba.min()*100:.2f}%")
print(f"  Q1 (25%): {np.percentile(y_pred_proba, 25)*100:.2f}%")
print(f"  Mediana: {np.median(y_pred_proba)*100:.2f}%")
print(f"  Q3 (75%): {np.percentile(y_pred_proba, 75)*100:.2f}%")
print(f"  Máximo: {y_pred_proba.max()*100:.2f}%")
print(f"  Média: {y_pred_proba.mean()*100:.2f}%")
print(f"  Desvio Padrão: {y_pred_proba.std()*100:.2f}%")
print(f"  Range: {(y_pred_proba.max() - y_pred_proba.min())*100:.2f}%")

# Histograma das predições
print(f"\n📊 Histograma de Predições:")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
hist, _ = np.histogram(y_pred_proba, bins=bins)
total = len(y_pred_proba)

for label, count in zip(bin_labels, hist):
    pct = (count / total) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:10s}: {bar:50s} {pct:5.2f}% ({count:,})")

# Verificar se há problema de calibração
if y_pred_proba.std() < 0.05:
    print(f"\n🚨 PROBLEMA CRÍTICO: Predições com variância muito baixa!")
    print(f"   Modelo está retornando valores quase constantes")
    print(f"   Possíveis causas:")
    print(f"   1. max_depth muito raso (atual: {rf_model.max_depth})")
    print(f"   2. min_samples_leaf muito alto (atual: {rf_model.min_samples_leaf})")
    print(f"   3. Features não têm poder preditivo")
    print(f"   4. Dataset muito desbalanceado e homogêneo")

# Verificar distribuição real vs predita
print(f"\n🎯 Distribuição Real vs Predita:")
print(f"  Real - Positivos: {y_test.mean()*100:.2f}%")
print(f"  Predito - Média: {y_pred_proba.mean()*100:.2f}%")

diff = abs(y_test.mean() - y_pred_proba.mean())
if diff > 0.1:
    print(f"  ⚠️  Diferença de {diff*100:.2f}% entre real e predito!")
    print(f"     Modelo pode estar mal calibrado")

print("\n" + "=" * 80)
print("💡 RECOMENDAÇÕES")
print("=" * 80)

if top3_sum > 0.8:
    print("1. Modelo focado demais em poucas features")
    print("   → Tentar max_features='sqrt' ou 'log2' para forçar diversidade")
    print("   → Reduzir max_depth para evitar overfitting nas top features")

if y_pred_proba.std() < 0.05:
    print("2. Variância de predições muito baixa")
    print("   → Aumentar max_depth (testar 30, 40, None)")
    print("   → Reduzir min_samples_leaf (testar 1)")
    print("   → Verificar se features têm variabilidade suficiente")

if diff > 0.1:
    print("3. Modelo desbalanceado")
    print("   → Usar class_weight='balanced'")
    print("   → Considerar SMOTE ou undersampling")

print("\n" + "=" * 80)
