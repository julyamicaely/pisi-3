"""
Teste COM force_reload para garantir que usa o modelo mais recente
"""
import sys
sys.path.append('c:/projetos/pisi-3-3')

from classification.prediction_service import load_model, predict_single

print("="*80)
print("TESTE: Forçando reload do modelo")
print("="*80)

# FORÇAR RELOAD
print("\n🔄 Forçando reload do modelo...")
load_model(force_reload=True)

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

# Teste 3: Variação de IDADE
test_age_25 = low_risk.copy()
test_age_25["age_years"] = 25

test_age_65 = low_risk.copy()
test_age_65["age_years"] = 65

# Teste 4: Variação de PRESSÃO
test_pa_110 = low_risk.copy()
test_pa_110["ap_hi"] = 110

test_pa_180 = low_risk.copy()
test_pa_180["ap_hi"] = 180

result_low = predict_single(low_risk)
result_high = predict_single(high_risk)
result_age_25 = predict_single(test_age_25)
result_age_65 = predict_single(test_age_65)
result_pa_110 = predict_single(test_pa_110)
result_pa_180 = predict_single(test_pa_180)

print(f"\n🟢 Perfil BAIXO RISCO:")
print(f"   Probabilidade: {result_low['probability']:.2f}%")

print(f"\n🔴 Perfil ALTO RISCO:")
print(f"   Probabilidade: {result_high['probability']:.2f}%")

print(f"\n📊 TESTE DE IDADE:")
print(f"   25 anos: {result_age_25['probability']:.2f}%")
print(f"   65 anos: {result_age_65['probability']:.2f}%")
print(f"   Diferença: {result_age_65['probability'] - result_age_25['probability']:.2f}%")

print(f"\n📊 TESTE DE PRESSÃO:")
print(f"   PA 110: {result_pa_110['probability']:.2f}%")
print(f"   PA 180: {result_pa_180['probability']:.2f}%")
print(f"   Diferença: {result_pa_180['probability'] - result_pa_110['probability']:.2f}%")

print("\n" + "="*80)
print("DIAGNÓSTICO:")
print("="*80)

idade_varia = abs(result_age_65['probability'] - result_age_25['probability']) > 20
pressao_varia = abs(result_pa_180['probability'] - result_pa_110['probability']) > 50
extremos_separados = abs(result_high['probability'] - result_low['probability']) > 50

if idade_varia and pressao_varia and extremos_separados:
    print("✅ MODELO NOVO (CORRIGIDO) CARREGADO!")
    print("   ✅ Idade influencia corretamente")
    print("   ✅ Pressão influencia corretamente")
    print("   ✅ Casos extremos bem separados")
    print("\n🎉 Dashboard DEVE estar funcionando perfeitamente!")
elif not idade_varia and not pressao_varia:
    print("❌ MODELO ANTIGO (BUGADO) AINDA CARREGADO!")
    print("   ❌ Idade NÃO influencia")
    print("   ❌ Pressão NÃO influencia")
    print("\n⚠️  AÇÃO NECESSÁRIA:")
    print("   1. Parar o dashboard (Ctrl+C no terminal)")
    print("   2. Iniciar novamente: python dashboard/app.py")
else:
    print("⚠️  MODELO PARCIALMENTE FUNCIONAL")
    print(f"   {'✅' if idade_varia else '❌'} Idade {'influencia' if idade_varia else 'NÃO influencia'}")
    print(f"   {'✅' if pressao_varia else '❌'} Pressão {'influencia' if pressao_varia else 'NÃO influencia'}")
    print(f"   {'✅' if extremos_separados else '❌'} Extremos {'separados' if extremos_separados else 'NÃO separados'}")

print("="*80)
