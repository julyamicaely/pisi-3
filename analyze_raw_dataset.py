"""
Análise do dataset original para entender o problema
"""
import pandas as pd
import numpy as np

# Carregar dados brutos
df = pd.read_csv('EDA/cardio_data.csv')

print("=" * 80)
print("ANÁLISE DO DATASET ORIGINAL")
print("=" * 80)

print(f"\n📊 Informações Gerais:")
print(f"  Total de registros: {len(df):,}")
print(f"  Features: {len(df.columns)}")
print(f"  Colunas: {list(df.columns)}")

# Distribuição do target
print(f"\n🎯 Distribuição da Classe Alvo (cardio):")
cardio_dist = df['cardio'].value_counts(normalize=True)
print(f"  Sem doença (0): {cardio_dist[0]*100:.2f}% ({(cardio_dist[0]*len(df)):,.0f})")
print(f"  Com doença (1): {cardio_dist[1]*100:.2f}% ({(cardio_dist[1]*len(df)):,.0f})")

# Estatísticas básicas
print(f"\n📈 Estatísticas de Features Contínuas:")
continuous = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
for col in continuous:
    if col in df.columns:
        print(f"\n  {col}:")
        print(f"    Média: {df[col].mean():.2f} ± {df[col].std():.2f}")
        print(f"    Min: {df[col].min()}, Max: {df[col].max()}")
        print(f"    Q1: {df[col].quantile(0.25):.2f}, Mediana: {df[col].median():.2f}, Q3: {df[col].quantile(0.75):.2f}")

# Features binárias
print(f"\n🔢 Distribuição de Features Binárias:")
binary = ['gender', 'smoke', 'alco', 'active', 'cholesterol', 'gluc']
for col in binary:
    if col in df.columns:
        dist = df[col].value_counts(normalize=True).sort_index()
        print(f"\n  {col}:")
        for val, pct in dist.items():
            print(f"    {val}: {pct*100:.2f}%")

# Correlação com target
print(f"\n🔗 Correlação com Target (cardio):")
correlations = []
for col in df.columns:
    if col != 'cardio' and col != 'id' and df[col].dtype in ['int64', 'float64']:
        try:
            corr = df[col].corr(df['cardio'])
            correlations.append((col, corr, abs(corr)))
        except:
            pass

correlations.sort(key=lambda x: x[2], reverse=True)

for col, corr, abs_corr in correlations:
    strength = "Forte" if abs_corr > 0.3 else "Moderada" if abs_corr > 0.1 else "Fraca"
    sign = "📈" if corr > 0 else "📉"
    print(f"  {sign} {col:15s}: {corr:+.4f} ({strength})")

# Análise por classe
print(f"\n📊 Comparação entre Classes (Sem doença vs Com doença):")
for col in ['age', 'ap_hi', 'ap_lo', 'weight', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']:
    if col in df.columns:
        mean_0 = df[df['cardio'] == 0][col].mean()
        mean_1 = df[df['cardio'] == 1][col].mean()
        diff_pct = ((mean_1 - mean_0) / mean_0) * 100 if mean_0 != 0 else 0
        
        print(f"\n  {col}:")
        print(f"    Sem doença: {mean_0:.2f}")
        print(f"    Com doença: {mean_1:.2f}")
        print(f"    Diferença: {diff_pct:+.1f}%")

# DIAGNÓSTICO
print("\n" + "=" * 80)
print("🔎 DIAGNÓSTICO")
print("=" * 80)

# Check: Classes balanceadas?
if abs(cardio_dist[0] - 0.5) < 0.05:
    print("✅ Classes bem balanceadas (~50/50)")
else:
    print(f"⚠️  Classes desbalanceadas: {cardio_dist[0]*100:.1f}% / {cardio_dist[1]*100:.1f}%")

# Check: Outliers extremos?
print(f"\n⚠️  Verificando outliers extremos:")
if df['ap_hi'].max() > 300 or df['ap_hi'].min() < 50:
    print(f"   - Pressão Sistólica: Min={df['ap_hi'].min()}, Max={df['ap_hi'].max()} (VALORES IMPOSSÍVEIS!)")
if df['ap_lo'].max() > 200 or df['ap_lo'].min() < 30:
    print(f"   - Pressão Diastólica: Min={df['ap_lo'].min()}, Max={df['ap_lo'].max()} (VALORES IMPOSSÍVEIS!)")
if df['height'].max() > 250 or df['height'].min() < 100:
    print(f"   - Altura: Min={df['height'].min()}, Max={df['height'].max()} (VALORES SUSPEITOS!)")
if df['weight'].max() > 200 or df['weight'].min() < 30:
    print(f"   - Peso: Min={df['weight'].min()}, Max={df['weight'].max()}")

# Check: Features com baixa correlação
weak_corrs = [col for col, _, abs_corr in correlations if abs_corr < 0.05]
if weak_corrs:
    print(f"\n⚠️  Features com correlação muito fraca (<0.05):")
    for col in weak_corrs:
        print(f"   - {col}")

print("\n" + "=" * 80)
