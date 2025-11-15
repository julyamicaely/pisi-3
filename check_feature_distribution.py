"""Verificar distribuição das features binárias"""

from classification.preprocess_data import preprocess_data
import pandas as pd

X_train, X_test, y_train, y_test, scaler, encoders, features = preprocess_data()

print("=" * 70)
print("📊 ANÁLISE DA DISTRIBUIÇÃO DAS FEATURES")
print("=" * 70)

print(f"\nTamanho do treino: {X_train.shape[0]}")
print(f"Tamanho do teste: {X_test.shape[0]}")

# Converter para DataFrame para análise
X_train_df = pd.DataFrame(X_train, columns=features)
X_test_df = pd.DataFrame(X_test, columns=features)

print("\n🔢 Distribuição das features binárias no TREINO (após SMOTE):")
binary_features = ['smoke', 'alco', 'active', 'cholesterol_high', 'gluc_high', 'gender']
for feat in binary_features:
    if feat in X_train_df.columns:
        counts = X_train_df[feat].value_counts()
        print(f"\n{feat}:")
        for val, count in counts.items():
            pct = count / len(X_train_df) * 100
            print(f"   {val}: {count:6d} ({pct:5.1f}%)")

print("\n\n🔢 Distribuição das features binárias no TESTE:")
for feat in binary_features:
    if feat in X_test_df.columns:
        counts = X_test_df[feat].value_counts()
        print(f"\n{feat}:")
        for val, count in counts.items():
            pct = count / len(X_test_df) * 100
            print(f"   {val}: {count:6d} ({pct:5.1f}%)")

print("\n\n📈 Estatísticas das features numéricas no TESTE:")
numeric_features = ['age_years', 'bmi', 'ap_hi', 'ap_lo']
for feat in numeric_features:
    if feat in X_test_df.columns:
        print(f"\n{feat}:")
        print(f"   Min: {X_test_df[feat].min():.2f}")
        print(f"   Max: {X_test_df[feat].max():.2f}")
        print(f"   Mean: {X_test_df[feat].mean():.2f}")
        print(f"   Std: {X_test_df[feat].std():.2f}")
        print(f"   Unique values: {X_test_df[feat].nunique()}")

print("\n" + "=" * 70)
