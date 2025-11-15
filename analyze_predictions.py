"""Analisar distribuição das predições"""

from classification.prediction_service import load_model
from classification.preprocess_data import preprocess_data
import numpy as np

print("=" * 70)
print("📊 ANÁLISE DA DISTRIBUIÇÃO DAS PREDIÇÕES")
print("=" * 70)

# Carregar dados de teste
X_train, X_test, y_train, y_test, scaler, encoders, features = preprocess_data()

# Carregar pipeline
pipeline = load_model()

# Fazer predições em TODOS os dados de teste
probas = pipeline.predict_proba(X_test)[:, 1]

print(f"\n📈 Estatísticas das predições no teste ({len(probas)} amostras):")
print(f"   Mínimo:  {probas.min()*100:.2f}%")
print(f"   Máximo:  {probas.max()*100:.2f}%")
print(f"   Média:   {probas.mean()*100:.2f}%")
print(f"   Mediana: {np.median(probas)*100:.2f}%")
print(f"   Desvio:  {probas.std()*100:.2f}%")

# Contar quantas predições únicas existem
unique_probas = np.unique(probas)
print(f"\n🔢 Número de probabilidades únicas: {len(unique_probas)}")

if len(unique_probas) < 10:
    print(f"   ⚠️ PROBLEMA: Modelo tem apenas {len(unique_probas)} saídas diferentes!")
    print(f"   Valores: {[f'{p*100:.2f}%' for p in unique_probas]}")
else:
    print(f"   ✅ OK: Modelo tem {len(unique_probas)} saídas diferentes")

# Distribuição por bins
print("\n📊 Distribuição das predições:")
bins = [0, 0.3, 0.5, 0.7, 1.0]
labels = ['0-30%', '30-50%', '50-70%', '70-100%']
counts = [np.sum((probas >= bins[i]) & (probas < bins[i+1])) for i in range(len(bins)-1)]

for label, count in zip(labels, counts):
    pct = count / len(probas) * 100
    bar = '█' * int(pct / 2)
    print(f"   {label:10s}: {count:5d} ({pct:5.1f}%) {bar}")

print("\n" + "=" * 70)
