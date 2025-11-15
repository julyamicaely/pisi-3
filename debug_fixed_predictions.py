"""Debug: verificar se modelo está fazendo predições variadas"""

from classification.prediction_service import load_model
import pandas as pd
import numpy as np

pipeline = load_model()

# Criar 5 perfis MUITO diferentes
test_cases = [
    # Caso 1: Jovem saudável
    {'gender': 0, 'ap_hi': 110, 'ap_lo': 70, 'smoke': 0, 'alco': 0, 'active': 1, 
     'age_years': 25, 'bmi': 20.0, 'cholesterol_high': 0, 'gluc_high': 0},
    
    # Caso 2: Meia-idade saudável
    {'gender': 0, 'ap_hi': 120, 'ap_lo': 80, 'smoke': 0, 'alco': 0, 'active': 1,
     'age_years': 45, 'bmi': 24.0, 'cholesterol_high': 0, 'gluc_high': 0},
    
    # Caso 3: Alto risco moderado
    {'gender': 1, 'ap_hi': 150, 'ap_lo': 95, 'smoke': 1, 'alco': 0, 'active': 0,
     'age_years': 60, 'bmi': 30.0, 'cholesterol_high': 1, 'gluc_high': 0},
    
    # Caso 4: Alto risco severo
    {'gender': 1, 'ap_hi': 180, 'ap_lo': 110, 'smoke': 1, 'alco': 1, 'active': 0,
     'age_years': 70, 'bmi': 35.0, 'cholesterol_high': 1, 'gluc_high': 1},
    
    # Caso 5: Jovem com hipertensão
    {'gender': 0, 'ap_hi': 160, 'ap_lo': 100, 'smoke': 0, 'alco': 0, 'active': 1,
     'age_years': 30, 'bmi': 22.0, 'cholesterol_high': 0, 'gluc_high': 0},
]

feature_order = ['gender', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 
                 'age_years', 'bmi', 'cholesterol_high', 'gluc_high']

df = pd.DataFrame(test_cases, columns=feature_order)

print("=" * 70)
print("🔍 TESTE: Modelo está fazendo predições variadas?")
print("=" * 70)

# Fazer predições
probas = pipeline.predict_proba(df)[:, 1]

print("\n📊 Resultados:")
for i, (case, proba) in enumerate(zip(test_cases, probas), 1):
    print(f"\nCaso {i}: {proba*100:.2f}%")
    print(f"   Idade: {case['age_years']}, IMC: {case['bmi']}, PA: {case['ap_hi']}/{case['ap_lo']}")
    print(f"   Fumo: {case['smoke']}, Álcool: {case['alco']}, Ativo: {case['active']}")

# Verificar variabilidade
unique_probas = np.unique(probas)
print(f"\n🔢 Número de valores únicos: {len(unique_probas)}")
print(f"   Min: {probas.min()*100:.2f}%")
print(f"   Max: {probas.max()*100:.2f}%")
print(f"   Range: {(probas.max() - probas.min())*100:.2f}%")

if len(unique_probas) <= 2:
    print("\n⚠️ PROBLEMA CRÍTICO: Modelo retorna apenas 2 valores!")
    print(f"   Valores: {[f'{p*100:.2f}%' for p in unique_probas]}")
else:
    print("\n✅ Modelo está funcionando - múltiplos valores de saída")

# Verificar o classificador
classifier = pipeline.named_steps['classifier']
print(f"\n🔬 Parâmetros do modelo:")
print(f"   n_estimators: {classifier.n_estimators}")
print(f"   max_depth: {classifier.max_depth}")
print(f"   min_samples_leaf: {classifier.min_samples_leaf}")

print("\n" + "=" * 70)
