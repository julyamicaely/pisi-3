"""
Análise detalhada de viés no modelo Random Forest
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.prediction_service import load_model, predict_single
import pandas as pd
import numpy as np

# Carregar o modelo
pipeline, metadata = load_model()
print("=" * 80)
print("ANÁLISE DE VIÉS - RANDOM FOREST")
print("=" * 80)

# Casos de teste variados
test_cases = [
    # Baixo risco extremo
    {
        "name": "🟢 Baixíssimo Risco",
        "gender": 2,  # Feminino
        "age_years": 25,
        "bmi": 21,
        "ap_hi": 110,
        "ap_lo": 70,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cholesterol_high": 0,
        "gluc_high": 0
    },
    # Risco moderado
    {
        "name": "🟡 Risco Moderado",
        "gender": 1,  # Masculino
        "age_years": 45,
        "bmi": 27,
        "ap_hi": 130,
        "ap_lo": 85,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cholesterol_high": 0,
        "gluc_high": 0
    },
    # Alto risco extremo
    {
        "name": "🔴 Altíssimo Risco",
        "gender": 1,  # Masculino
        "age_years": 65,
        "bmi": 35,
        "ap_hi": 180,
        "ap_lo": 110,
        "smoke": 1,
        "alco": 1,
        "active": 0,
        "cholesterol_high": 1,
        "gluc_high": 1
    },
    # Teste: apenas mudando gênero (caso 1 -> masculino)
    {
        "name": "🔵 Baixo Risco (Masculino)",
        "gender": 1,  # Masculino
        "age_years": 25,
        "bmi": 21,
        "ap_hi": 110,
        "ap_lo": 70,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cholesterol_high": 0,
        "gluc_high": 0
    },
    # Teste: mudando apenas pressão arterial
    {
        "name": "🟣 Só Hipertensão",
        "gender": 2,  # Feminino
        "age_years": 35,
        "bmi": 23,
        "ap_hi": 170,
        "ap_lo": 105,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cholesterol_high": 0,
        "gluc_high": 0
    },
    # Teste: mudando apenas IMC
    {
        "name": "🟠 Só Obesidade",
        "gender": 2,  # Feminino
        "age_years": 35,
        "bmi": 38,
        "ap_hi": 120,
        "ap_lo": 80,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cholesterol_high": 0,
        "gluc_high": 0
    },
    # Teste: fumante mas saudável em outros aspectos
    {
        "name": "🚬 Só Fumante",
        "gender": 2,  # Feminino
        "age_years": 30,
        "bmi": 22,
        "ap_hi": 115,
        "ap_lo": 75,
        "smoke": 1,  # FUMANTE
        "alco": 0,
        "active": 1,
        "cholesterol_high": 0,
        "gluc_high": 0
    },
    # Teste: álcool mas saudável em outros aspectos
    {
        "name": "🍺 Só Álcool",
        "gender": 1,  # Masculino
        "age_years": 30,
        "bmi": 22,
        "ap_hi": 115,
        "ap_lo": 75,
        "smoke": 0,
        "alco": 1,  # ÁLCOOL
        "active": 1,
        "cholesterol_high": 0,
        "gluc_high": 0
    },
]

results = []
predictions = []

for case in test_cases:
    name = case.pop("name")
    result = predict_single(case)
    prob = result["probability"]
    predictions.append(prob)
    
    print(f"\n{name}")
    print("-" * 80)
    print(f"Gender: {'Feminino' if case['gender'] == 2 else 'Masculino'}, "
          f"Idade: {case['age_years']}, IMC: {case['bmi']}, "
          f"PA: {case['ap_hi']}/{case['ap_lo']}")
    print(f"Fumo: {case['smoke']}, Álcool: {case['alco']}, Ativo: {case['active']}, "
          f"Colesterol: {case['cholesterol_high']}, Glicose: {case['gluc_high']}")
    print(f"\n➡️  PREDIÇÃO: {prob:.2f}%")
    print(f"Top 3 Fatores:")
    for i, factor in enumerate(result["risk_factors"][:3], 1):
        print(f"  {i}. {factor['label']}: {factor['importance']:.1f}%")
    
    results.append({
        "name": name,
        "prob": prob,
        "gender": case["gender"],
        "age": case["age_years"],
        "bmi": case["bmi"],
        "ap_hi": case["ap_hi"],
        "smoke": case["smoke"],
        "alco": case["alco"]
    })

# Análise de variabilidade
print("\n" + "=" * 80)
print("ANÁLISE DE VARIABILIDADE")
print("=" * 80)

predictions = np.array(predictions)
print(f"\n📊 Estatísticas:")
print(f"  Mínimo: {predictions.min():.2f}%")
print(f"  Máximo: {predictions.max():.2f}%")
print(f"  Média: {predictions.mean():.2f}%")
print(f"  Desvio Padrão: {predictions.std():.2f}%")
print(f"  Range: {predictions.max() - predictions.min():.2f}%")
print(f"  Valores únicos: {len(np.unique(predictions))}")

# Análise de viés por gênero
print(f"\n🚹🚺 Análise por Gênero:")
df_results = pd.DataFrame(results)
male_preds = df_results[df_results["gender"] == 1]["prob"]
female_preds = df_results[df_results["gender"] == 2]["prob"]

if len(male_preds) > 0:
    print(f"  Masculino - Média: {male_preds.mean():.2f}%, Std: {male_preds.std():.2f}%")
if len(female_preds) > 0:
    print(f"  Feminino - Média: {female_preds.mean():.2f}%, Std: {female_preds.std():.2f}%")

# Verificar se features binárias estão tendo impacto
print(f"\n🔍 Teste de Impacto de Features Binárias:")
print(f"  Caso base (sem fatores) vs Só Fumante: ", end="")
base_prob = results[0]["prob"]
smoke_prob = [r["prob"] for r in results if "Fumante" in r["name"]][0]
print(f"Δ = {abs(smoke_prob - base_prob):.2f}%")

print(f"  Caso base (sem fatores) vs Só Álcool: ", end="")
alco_prob = [r["prob"] for r in results if "Álcool" in r["name"]][0]
print(f"Δ = {abs(alco_prob - base_prob):.2f}%")

# DIAGNÓSTICO
print("\n" + "=" * 80)
print("🔎 DIAGNÓSTICO")
print("=" * 80)

if predictions.std() < 3:
    print("⚠️  PROBLEMA: Variabilidade muito baixa (<3%)")
    print("   Modelo ainda parece estar ignorando muitas features")
elif predictions.std() < 5:
    print("⚠️  ALERTA: Variabilidade baixa (<5%)")
    print("   Modelo pode estar sub-utilizando algumas features")
else:
    print("✅ Variabilidade adequada (≥5%)")

if predictions.max() - predictions.min() < 10:
    print("⚠️  PROBLEMA: Range muito pequeno (<10%)")
    print("   Casos extremamente diferentes têm predições muito similares")
elif predictions.max() - predictions.min() < 20:
    print("⚠️  ALERTA: Range moderado (<20%)")
else:
    print("✅ Range adequado (≥20%)")

if len(np.unique(predictions)) < 5:
    print("⚠️  PROBLEMA: Poucos valores únicos (<5)")
    print("   Modelo produzindo predições muito similares")
else:
    print(f"✅ Boa diversidade: {len(np.unique(predictions))} valores únicos")

print("\n" + "=" * 80)
