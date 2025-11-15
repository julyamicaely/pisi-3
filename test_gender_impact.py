"""Teste para verificar impacto do gênero nas predições"""

from classification.prediction_service import predict_single

print("=" * 70)
print("🔬 ANÁLISE: Impacto do Gênero nas Predições")
print("=" * 70)

# Perfil 1: Pessoa saudável
print("\n📊 PERFIL 1: Pessoa Saudável (50 anos, IMC normal, sem fatores de risco)")
print("-" * 70)

feminino_saudavel = predict_single({
    'gender': 0,
    'age_years': 50,
    'bmi': 23,
    'ap_hi': 115,
    'ap_lo': 75,
    'cholesterol_high': 0,
    'gluc_high': 0,
    'smoke': 0,
    'alco': 0,
    'active': 1
})

masculino_saudavel = predict_single({
    'gender': 1,
    'age_years': 50,
    'bmi': 23,
    'ap_hi': 115,
    'ap_lo': 75,
    'cholesterol_high': 0,
    'gluc_high': 0,
    'smoke': 0,
    'alco': 0,
    'active': 1
})

print(f"👩 Feminino: {feminino_saudavel['probability']:.2f}%")
print(f"👨 Masculino: {masculino_saudavel['probability']:.2f}%")
print(f"📊 Diferença: {abs(feminino_saudavel['probability'] - masculino_saudavel['probability']):.2f}%")

# Perfil 2: Pessoa com fatores de risco moderados
print("\n📊 PERFIL 2: Fatores de Risco Moderados (60 anos, sobrepeso, hipertensão leve)")
print("-" * 70)

feminino_moderado = predict_single({
    'gender': 0,
    'age_years': 60,
    'bmi': 28,
    'ap_hi': 140,
    'ap_lo': 90,
    'cholesterol_high': 1,
    'gluc_high': 0,
    'smoke': 0,
    'alco': 0,
    'active': 0
})

masculino_moderado = predict_single({
    'gender': 1,
    'age_years': 60,
    'bmi': 28,
    'ap_hi': 140,
    'ap_lo': 90,
    'cholesterol_high': 1,
    'gluc_high': 0,
    'smoke': 0,
    'alco': 0,
    'active': 0
})

print(f"👩 Feminino: {feminino_moderado['probability']:.2f}%")
print(f"👨 Masculino: {masculino_moderado['probability']:.2f}%")
print(f"📊 Diferença: {abs(feminino_moderado['probability'] - masculino_moderado['probability']):.2f}%")

# Perfil 3: Alto risco
print("\n📊 PERFIL 3: Alto Risco (65 anos, obesidade, múltiplos fatores)")
print("-" * 70)

feminino_alto = predict_single({
    'gender': 0,
    'age_years': 65,
    'bmi': 32,
    'ap_hi': 160,
    'ap_lo': 100,
    'cholesterol_high': 1,
    'gluc_high': 1,
    'smoke': 1,
    'alco': 1,
    'active': 0
})

masculino_alto = predict_single({
    'gender': 1,
    'age_years': 65,
    'bmi': 32,
    'ap_hi': 160,
    'ap_lo': 100,
    'cholesterol_high': 1,
    'gluc_high': 1,
    'smoke': 1,
    'alco': 1,
    'active': 0
})

print(f"👩 Feminino: {feminino_alto['probability']:.2f}%")
print(f"👨 Masculino: {masculino_alto['probability']:.2f}%")
print(f"📊 Diferença: {abs(feminino_alto['probability'] - masculino_alto['probability']):.2f}%")

# Análise da importância do gênero
print("\n" + "=" * 70)
print("📈 ANÁLISE:")
print("=" * 70)

import joblib
pipeline = joblib.load('classification/models/random_forest_pipeline.joblib')
classifier = pipeline.named_steps['classifier']
importances = classifier.feature_importances_

features = ['gender', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'age_years', 'bmi', 'cholesterol_high', 'gluc_high']

print("\n🔍 Importância de cada feature no modelo:")
for feature, importance in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"   {feature:20s}: {importance*100:5.2f}%")

print("\n💡 CONCLUSÃO:")
print(f"   • Gênero tem importância de {importances[0]*100:.2f}% no modelo")
print("   • Outros fatores (IMC, pressão, idade) são MUITO mais importantes")
print("   • Por isso a diferença entre F/M é pequena com mesmos fatores")
print("=" * 70)
