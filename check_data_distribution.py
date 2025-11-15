"""
Verificar distribuição de classes e features no dataset
"""
import pandas as pd
import numpy as np
from classification.preprocess_data import load_and_preprocess_data

print("=" * 80)
print("ANÁLISE DE DISTRIBUIÇÃO DOS DADOS")
print("=" * 80)

# Carregar e preprocessar
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Reconstruir dataframes com target
df_train = X_train.copy()
df_train['cardio'] = y_train
df_test = X_test.copy()
df_test['cardio'] = y_test

print(f"\n📊 Tamanho dos Datasets:")
print(f"  Treino: {len(df_train):,} amostras")
print(f"  Teste: {len(df_test):,} amostras")
print(f"  Total: {len(df_train) + len(df_test):,} amostras")

# Distribuição da variável alvo
print(f"\n🎯 Distribuição da Classe Alvo (cardio):")
train_dist = df_train['cardio'].value_counts(normalize=True)
test_dist = df_test['cardio'].value_counts(normalize=True)

print(f"  TREINO:")
print(f"    Sem doença (0): {train_dist.get(0, 0)*100:.2f}%")
print(f"    Com doença (1): {train_dist.get(1, 0)*100:.2f}%")
print(f"  TESTE:")
print(f"    Sem doença (0): {test_dist.get(0, 0)*100:.2f}%")
print(f"    Com doença (1): {test_dist.get(1, 0)*100:.2f}%")

# Verificar se há desbalanceamento severo
ratio = train_dist.get(1, 0) / train_dist.get(0, 1) if train_dist.get(0, 0) > 0 else 0
print(f"\n⚖️  Ratio de classes (1/0): {ratio:.3f}")

if ratio > 0.9 and ratio < 1.1:
    print("  ✅ Classes balanceadas")
elif ratio > 0.7 and ratio < 1.3:
    print("  ⚠️  Leve desbalanceamento")
else:
    print("  🚨 DESBALANCEAMENTO SEVERO!")

# Estatísticas das features contínuas
print(f"\n📈 Estatísticas Features Contínuas:")
continuous_features = ['age_years', 'bmi', 'ap_hi', 'ap_lo']

for feature in continuous_features:
    mean_val = df_train[feature].mean()
    std_val = df_train[feature].std()
    min_val = df_train[feature].min()
    max_val = df_train[feature].max()
    
    print(f"\n  {feature}:")
    print(f"    Média: {mean_val:.2f} ± {std_val:.2f}")
    print(f"    Range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"    CV: {(std_val/mean_val)*100:.1f}%")

# Distribuição das features binárias
print(f"\n🔢 Distribuição Features Binárias:")
binary_features = ['smoke', 'alco', 'active', 'cholesterol_high', 'gluc_high', 'gender']

for feature in binary_features:
    if feature in df_train.columns:
        dist = df_train[feature].value_counts(normalize=True)
        print(f"\n  {feature}:")
        for val, pct in dist.items():
            print(f"    {val}: {pct*100:.2f}%")
        
        # Alerta se muito desbalanceado
        if dist.min() < 0.05:
            print(f"    ⚠️  Feature muito desbalanceada! Classe minoritária: {dist.min()*100:.2f}%")

# Correlação com o target
print(f"\n🔗 Correlação com Target (cardio):")
features = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'smoke', 'alco', 
            'active', 'cholesterol_high', 'gluc_high', 'gender']

correlations = []
for feature in features:
    if feature in df_train.columns:
        corr = df_train[feature].corr(df_train['cardio'])
        correlations.append((feature, abs(corr), corr))

correlations.sort(key=lambda x: x[1], reverse=True)

for feature, abs_corr, corr in correlations:
    sign = "📈" if corr > 0 else "📉"
    strength = "Forte" if abs_corr > 0.3 else "Moderada" if abs_corr > 0.1 else "Fraca"
    print(f"  {sign} {feature:20s}: {corr:+.4f} ({strength})")

# DIAGNÓSTICO FINAL
print("\n" + "=" * 80)
print("🔎 DIAGNÓSTICO")
print("=" * 80)

# Check 1: Classes desbalanceadas?
if ratio < 0.4 or ratio > 2.5:
    print("⚠️  PROBLEMA: Classes muito desbalanceadas")
    print("   SOLUÇÃO: Usar class_weight='balanced' ou SMOTE")

# Check 2: Features com baixa correlação?
weak_corrs = [f for f, ac, c in correlations if ac < 0.05]
if weak_corrs:
    print(f"⚠️  ALERTA: {len(weak_corrs)} features com correlação muito fraca")
    print(f"   Features: {', '.join(weak_corrs)}")

# Check 3: Features binárias desbalanceadas?
imbalanced_binary = []
for feature in binary_features:
    if feature in df_train.columns:
        dist = df_train[feature].value_counts(normalize=True)
        if dist.min() < 0.05:
            imbalanced_binary.append(f"{feature} ({dist.min()*100:.1f}%)")

if imbalanced_binary:
    print(f"⚠️  PROBLEMA: Features binárias muito desbalanceadas:")
    for feat in imbalanced_binary:
        print(f"   - {feat}")
    print("   Modelo pode estar ignorando essas features")

# Check 4: Variabilidade das contínuas
low_cv_features = []
for feature in continuous_features:
    cv = (df_train[feature].std() / df_train[feature].mean()) * 100
    if cv < 20:
        low_cv_features.append(f"{feature} (CV={cv:.1f}%)")

if low_cv_features:
    print(f"⚠️  ALERTA: Features com baixa variabilidade:")
    for feat in low_cv_features:
        print(f"   - {feat}")

print("\n" + "=" * 80)
