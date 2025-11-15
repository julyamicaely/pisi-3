"""
Teste rápido: Verificar se o dashboard está usando o modelo correto
"""
import sys
sys.path.append('c:/projetos/pisi-3-3')

from classification.prediction_service import predict_single

print("="*80)
print("TESTE: Dashboard está usando modelo correto?")
print("="*80)

# Teste 1: Perfil baixo risco
low_risk = {
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
}

# Teste 2: Perfil alto risco
high_risk = {
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
}

result_low = predict_single(low_risk)
result_high = predict_single(high_risk)

print(f"\n🟢 Perfil BAIXO RISCO:")
print(f"   Probabilidade: {result_low['probability']:.2f}%")
print(f"   Classificação: {result_low['risk_label']}")

print(f"\n🔴 Perfil ALTO RISCO:")
print(f"   Probabilidade: {result_high['probability']:.2f}%")
print(f"   Classificação: {result_high['risk_label']}")

diff = result_high['probability'] - result_low['probability']
print(f"\n📊 Diferença: {diff:.2f}%")

print("\n" + "="*80)
print("RESULTADO:")
print("="*80)

if result_low['probability'] < 15 and result_high['probability'] > 70:
    print("✅ MODELO CORRETO CARREGADO!")
    print("   Dashboard DEVE estar funcionando corretamente")
    print(f"   Baixo risco: {result_low['probability']:.2f}% (esperado: ~7-10%)")
    print(f"   Alto risco: {result_high['probability']:.2f}% (esperado: ~80-85%)")
elif result_low['probability'] > 50 and result_high['probability'] > 50:
    print("❌ MODELO ANTIGO (BUGADO) CARREGADO!")
    print("   Dashboard NÃO reflete as correções")
    print("   SOLUÇÃO: Reiniciar o dashboard")
else:
    print("⚠️  MODELO EM ESTADO INTERMEDIÁRIO")
    print("   Pode ser necessário reiniciar o dashboard")

print("="*80)
