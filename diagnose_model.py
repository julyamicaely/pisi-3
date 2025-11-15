"""Diagnóstico do problema de predições fixas"""

from classification.prediction_service import load_model
import pandas as pd
import numpy as np

print("=" * 70)
print("🔍 DIAGNÓSTICO: Por que as predições estão fixas?")
print("=" * 70)

# Carregar pipeline
pipeline = load_model()

# Criar diferentes perfis
profiles = [
    {'name': 'Jovem Saudável', 'gender': 0, 'age_years': 25, 'bmi': 20, 'ap_hi': 110, 'ap_lo': 70, 'cholesterol_high': 0, 'gluc_high': 0, 'smoke': 0, 'alco': 0, 'active': 1},
    {'name': 'Meia-idade Normal', 'gender': 0, 'age_years': 50, 'bmi': 25, 'ap_hi': 120, 'ap_lo': 80, 'cholesterol_high': 0, 'gluc_high': 0, 'smoke': 0, 'alco': 0, 'active': 1},
    {'name': 'Idoso Alto Risco', 'gender': 1, 'age_years': 70, 'bmi': 35, 'ap_hi': 180, 'ap_lo': 110, 'cholesterol_high': 1, 'gluc_high': 1, 'smoke': 1, 'alco': 1, 'active': 0},
]

print("\n📊 Testando diferentes perfis:\n")

for profile in profiles:
    name = profile.pop('name')
    df = pd.DataFrame([profile])
    
    # Ver dados ANTES do scaler
    print(f"{name}:")
    print(f"   Input: {profile}")
    
    # Ver dados DEPOIS do scaler
    scaler = pipeline.named_steps['scaler']
    scaled = scaler.transform(df)
    print(f"   Scaled: {scaled[0][:3]}... (primeiros 3)")
    
    # Predição
    proba = pipeline.predict_proba(df)[0, 1]
    print(f"   Probabilidade: {proba*100:.2f}%\n")
    
    profile['name'] = name  # restaurar

# Verificar se o scaler foi treinado
print("\n🔬 Verificando o scaler:")
scaler = pipeline.named_steps['scaler']
print(f"   Center: {scaler.center_[:3]}...")
print(f"   Scale: {scaler.scale_[:3]}...")

# Verificar modelo
print("\n🔬 Verificando o classificador:")
classifier = pipeline.named_steps['classifier']
print(f"   Tipo: {type(classifier).__name__}")
print(f"   N_estimators: {classifier.n_estimators}")
print(f"   N_classes: {classifier.n_classes_}")
print(f"   Feature importances: {classifier.feature_importances_}")

print("\n" + "=" * 70)
