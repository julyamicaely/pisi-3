"""
Teste direto do pipeline para entender o problema
"""
import joblib
import pandas as pd
import numpy as np

# Carregar pipeline
pipeline = joblib.load('classification/models/random_forest_pipeline.joblib')

print("=" * 80)
print("TESTE DIRETO DO PIPELINE")
print("=" * 80)

# Ordem correta das features
feature_order = ['gender', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 
                 'age_years', 'bmi', 'cholesterol_high', 'gluc_high']

# Teste 1: Perfil base
test1 = pd.DataFrame([[2, 120, 80, 0, 0, 1, 40, 25, 0, 0]], columns=feature_order)
# Teste 2: Apenas idade diferente
test2 = pd.DataFrame([[2, 120, 80, 0, 0, 1, 65, 25, 0, 0]], columns=feature_order)
# Teste 3: Apenas pressão diferente
test3 = pd.DataFrame([[2, 180, 110, 0, 0, 1, 40, 25, 0, 0]], columns=feature_order)
# Teste 4: Apenas IMC diferente
test4 = pd.DataFrame([[2, 120, 80, 0, 0, 1, 40, 35, 0, 0]], columns=feature_order)

print("\n📊 Dados de entrada (SEM escalar):")
print("\nTeste 1 (base):        ", test1.values[0])
print("Teste 2 (idade 65):    ", test2.values[0])
print("Teste 3 (PA 180/110):  ", test3.values[0])
print("Teste 4 (IMC 35):      ", test4.values[0])

# Verificar o scaler
scaler = pipeline.named_steps['scaler']
print("\n🔧 Informações do Scaler (RobustScaler):")
print(f"   Center (mediana): {scaler.center_}")
print(f"   Scale (IQR):      {scaler.scale_}")

# Testar scaler manualmente
print("\n📏 Dados APÓS scaling:")
test1_scaled = scaler.transform(test1)
test2_scaled = scaler.transform(test2)
test3_scaled = scaler.transform(test3)
test4_scaled = scaler.transform(test4)

print("\nTeste 1 (base):        ", test1_scaled[0])
print("Teste 2 (idade 65):    ", test2_scaled[0])
print("Teste 3 (PA 180/110):  ", test3_scaled[0])
print("Teste 4 (IMC 35):      ", test4_scaled[0])

# Verificar se há diferenças
print("\n🔍 Diferenças após scaling:")
print(f"   Teste 2 vs 1: {np.any(test2_scaled != test1_scaled)}")
print(f"   Teste 3 vs 1: {np.any(test3_scaled != test1_scaled)}")
print(f"   Teste 4 vs 1: {np.any(test4_scaled != test1_scaled)}")

# Predições
print("\n🎯 PREDIÇÕES:")
prob1 = pipeline.predict_proba(test1)[0, 1] * 100
prob2 = pipeline.predict_proba(test2)[0, 1] * 100
prob3 = pipeline.predict_proba(test3)[0, 1] * 100
prob4 = pipeline.predict_proba(test4)[0, 1] * 100

print(f"\nTeste 1 (base):        {prob1:.2f}%")
print(f"Teste 2 (idade 65):    {prob2:.2f}% (Δ {prob2-prob1:+.2f}%)")
print(f"Teste 3 (PA 180/110):  {prob3:.2f}% (Δ {prob3-prob1:+.2f}%)")
print(f"Teste 4 (IMC 35):      {prob4:.2f}% (Δ {prob4-prob1:+.2f}%)")

# Testar com features binárias
test5 = pd.DataFrame([[2, 120, 80, 1, 0, 1, 40, 25, 0, 0]], columns=feature_order)  # smoke=1
test6 = pd.DataFrame([[2, 120, 80, 0, 0, 1, 40, 25, 1, 0]], columns=feature_order)  # cholesterol=1

prob5 = pipeline.predict_proba(test5)[0, 1] * 100
prob6 = pipeline.predict_proba(test6)[0, 1] * 100

print(f"\nTeste 5 (smoke=1):     {prob5:.2f}% (Δ {prob5-prob1:+.2f}%)")
print(f"Teste 6 (cholesterol=1): {prob6:.2f}% (Δ {prob6-prob1:+.2f}%)")

print("\n" + "=" * 80)
print("DIAGNÓSTICO")
print("=" * 80)

if abs(prob2 - prob1) < 0.1 and abs(prob3 - prob1) < 0.1 and abs(prob4 - prob1) < 0.1:
    print("❌ PROBLEMA CONFIRMADO: Features contínuas NÃO afetam predição!")
    print("   Possíveis causas:")
    print("   1. Scaler com valores incorretos")
    print("   2. Modelo treinado com dados problemáticos")
    print("   3. Features foram todas iguais no treinamento")
else:
    print("✅ Features contínuas ESTÃO afetando predição")

if abs(prob5 - prob1) > 5 or abs(prob6 - prob1) > 5:
    print("✅ Features binárias estão funcionando corretamente")
else:
    print("❌ Features binárias também não estão funcionando!")

print("=" * 80)
