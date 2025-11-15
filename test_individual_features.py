"""
Teste rigoroso de cada feature individual
Verifica se mudanças em CADA variável afetam a predição corretamente
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.prediction_service import predict_single
import pandas as pd

print("=" * 80)
print("TESTE DE SENSIBILIDADE POR FEATURE")
print("=" * 80)

# Perfil BASE (neutro)
base_profile = {
    "gender": 2,  # Feminino
    "age_years": 40,
    "bmi": 25,
    "ap_hi": 120,
    "ap_lo": 80,
    "smoke": 0,
    "alco": 0,
    "active": 1,
    "cholesterol_high": 0,
    "gluc_high": 0
}

base_result = predict_single(base_profile.copy())
base_prob = base_result["probability"]

print(f"\n🎯 PERFIL BASE:")
print(f"   Probabilidade: {base_prob:.2f}%")
print(f"   (Mulher, 40 anos, IMC 25, PA 120/80, sem fatores de risco)\n")

# ============================================================================
# TESTE 1: IDADE
# ============================================================================
print("=" * 80)
print("1️⃣  TESTE: IDADE (esperado: maior idade = maior risco)")
print("=" * 80)

age_tests = [25, 35, 45, 55, 65]
age_results = []

for age in age_tests:
    profile = base_profile.copy()
    profile["age_years"] = age
    result = predict_single(profile)
    prob = result["probability"]
    age_results.append(prob)
    delta = prob - base_prob
    emoji = "🟢" if age < 40 else "🟡" if age == 40 else "🔴"
    print(f"{emoji} {age} anos: {prob:.2f}% (Δ {delta:+.2f}%)")

# Verificar se aumenta com idade
is_increasing = all(age_results[i] <= age_results[i+1] for i in range(len(age_results)-1))
if is_increasing:
    print("✅ CORRETO: Risco aumenta com idade")
else:
    print("❌ ERRO: Risco NÃO aumenta consistentemente com idade!")

# ============================================================================
# TESTE 2: IMC
# ============================================================================
print("\n" + "=" * 80)
print("2️⃣  TESTE: IMC (esperado: maior IMC = maior risco)")
print("=" * 80)

bmi_tests = [18, 22, 25, 30, 35, 40]
bmi_results = []

for bmi in bmi_tests:
    profile = base_profile.copy()
    profile["bmi"] = bmi
    result = predict_single(profile)
    prob = result["probability"]
    bmi_results.append(prob)
    delta = prob - base_prob
    
    if bmi < 18.5:
        cat = "Baixo peso"
        emoji = "🟡"
    elif bmi < 25:
        cat = "Normal"
        emoji = "🟢"
    elif bmi < 30:
        cat = "Sobrepeso"
        emoji = "🟠"
    else:
        cat = "Obesidade"
        emoji = "🔴"
    
    print(f"{emoji} IMC {bmi}: {prob:.2f}% ({cat}, Δ {delta:+.2f}%)")

# Verificar se aumenta com IMC
is_increasing = all(bmi_results[i] <= bmi_results[i+1] for i in range(len(bmi_results)-1))
if is_increasing:
    print("✅ CORRETO: Risco aumenta com IMC")
else:
    print("⚠️  ATENÇÃO: Risco NÃO aumenta linearmente com IMC")

# ============================================================================
# TESTE 3: PRESSÃO ARTERIAL SISTÓLICA
# ============================================================================
print("\n" + "=" * 80)
print("3️⃣  TESTE: PRESSÃO SISTÓLICA (esperado: maior PA = maior risco)")
print("=" * 80)

ap_hi_tests = [90, 110, 120, 130, 140, 160, 180]
ap_hi_results = []

for ap_hi in ap_hi_tests:
    profile = base_profile.copy()
    profile["ap_hi"] = ap_hi
    result = predict_single(profile)
    prob = result["probability"]
    ap_hi_results.append(prob)
    delta = prob - base_prob
    
    if ap_hi < 120:
        cat = "Normal"
        emoji = "🟢"
    elif ap_hi < 130:
        cat = "Elevada"
        emoji = "🟡"
    elif ap_hi < 140:
        cat = "Hipertensão 1"
        emoji = "🟠"
    else:
        cat = "Hipertensão 2"
        emoji = "🔴"
    
    print(f"{emoji} PA {ap_hi}: {prob:.2f}% ({cat}, Δ {delta:+.2f}%)")

# Verificar se aumenta com pressão
is_increasing = all(ap_hi_results[i] <= ap_hi_results[i+1] for i in range(len(ap_hi_results)-1))
if is_increasing:
    print("✅ CORRETO: Risco aumenta com pressão sistólica")
else:
    print("⚠️  ATENÇÃO: Risco NÃO aumenta linearmente com pressão")

# ============================================================================
# TESTE 4: PRESSÃO ARTERIAL DIASTÓLICA
# ============================================================================
print("\n" + "=" * 80)
print("4️⃣  TESTE: PRESSÃO DIASTÓLICA (esperado: maior PA = maior risco)")
print("=" * 80)

ap_lo_tests = [60, 70, 80, 90, 100, 110]
ap_lo_results = []

for ap_lo in ap_lo_tests:
    profile = base_profile.copy()
    profile["ap_lo"] = ap_lo
    result = predict_single(profile)
    prob = result["probability"]
    ap_lo_results.append(prob)
    delta = prob - base_prob
    
    if ap_lo < 80:
        cat = "Normal"
        emoji = "🟢"
    elif ap_lo < 90:
        cat = "Elevada"
        emoji = "🟡"
    else:
        cat = "Hipertensão"
        emoji = "🔴"
    
    print(f"{emoji} PA diast {ap_lo}: {prob:.2f}% ({cat}, Δ {delta:+.2f}%)")

# Verificar se aumenta
is_increasing = all(ap_lo_results[i] <= ap_lo_results[i+1] for i in range(len(ap_lo_results)-1))
if is_increasing:
    print("✅ CORRETO: Risco aumenta com pressão diastólica")
else:
    print("⚠️  ATENÇÃO: Risco NÃO aumenta linearmente com pressão diastólica")

# ============================================================================
# TESTE 5: FEATURES BINÁRIAS
# ============================================================================
print("\n" + "=" * 80)
print("5️⃣  TESTE: FEATURES BINÁRIAS (esperado: presença = maior risco)")
print("=" * 80)

binary_features = {
    "smoke": "Tabagismo",
    "alco": "Álcool",
    "cholesterol_high": "Colesterol Alto",
    "gluc_high": "Glicose Alta"
}

binary_results = []
for feature, label in binary_features.items():
    profile = base_profile.copy()
    profile[feature] = 1
    result = predict_single(profile)
    prob = result["probability"]
    delta = prob - base_prob
    
    emoji = "🔴" if delta > 5 else "🟠" if delta > 2 else "🟡" if delta > 0 else "❌"
    print(f"{emoji} {label:20s}: {prob:.2f}% (Δ {delta:+.2f}%)")
    binary_results.append((label, delta))

# Verificar se TODAS aumentam o risco
all_increase = all(delta > 0 for _, delta in binary_results)
if all_increase:
    print("✅ CORRETO: Todos os fatores de risco aumentam a probabilidade")
else:
    problematic = [label for label, delta in binary_results if delta <= 0]
    print(f"❌ ERRO: Fatores que NÃO aumentam risco: {', '.join(problematic)}")

# ============================================================================
# TESTE 6: ATIVIDADE FÍSICA
# ============================================================================
print("\n" + "=" * 80)
print("6️⃣  TESTE: ATIVIDADE FÍSICA (esperado: ativo = MENOR risco)")
print("=" * 80)

profile_inactive = base_profile.copy()
profile_inactive["active"] = 0
result_inactive = predict_single(profile_inactive)

profile_active = base_profile.copy()
profile_active["active"] = 1
result_active = predict_single(profile_active)

print(f"🔴 Sedentário: {result_inactive['probability']:.2f}%")
print(f"🟢 Ativo:      {result_active['probability']:.2f}%")

if result_active["probability"] < result_inactive["probability"]:
    delta = result_inactive["probability"] - result_active["probability"]
    print(f"✅ CORRETO: Atividade física reduz risco em {delta:.2f}%")
else:
    print("❌ ERRO: Atividade física NÃO está reduzindo risco!")

# ============================================================================
# TESTE 7: GÊNERO
# ============================================================================
print("\n" + "=" * 80)
print("7️⃣  TESTE: GÊNERO (verificar diferença entre masculino/feminino)")
print("=" * 80)

profile_female = base_profile.copy()
profile_female["gender"] = 2
result_female = predict_single(profile_female)

profile_male = base_profile.copy()
profile_male["gender"] = 1
result_male = predict_single(profile_male)

print(f"🚺 Feminino:  {result_female['probability']:.2f}%")
print(f"🚹 Masculino: {result_male['probability']:.2f}%")

diff = abs(result_male["probability"] - result_female["probability"])
if diff < 10:
    print(f"✅ Diferença razoável: {diff:.2f}%")
else:
    print(f"⚠️  Diferença alta: {diff:.2f}%")

# ============================================================================
# TESTE 8: CASOS EXTREMOS
# ============================================================================
print("\n" + "=" * 80)
print("8️⃣  TESTE: CASOS EXTREMOS")
print("=" * 80)

# Caso 1: SUPER SAUDÁVEL
super_healthy = {
    "gender": 2,
    "age_years": 25,
    "bmi": 21,
    "ap_hi": 105,
    "ap_lo": 65,
    "smoke": 0,
    "alco": 0,
    "active": 1,
    "cholesterol_high": 0,
    "gluc_high": 0
}
result_healthy = predict_single(super_healthy)

# Caso 2: SUPER ALTO RISCO
super_risk = {
    "gender": 1,
    "age_years": 65,
    "bmi": 38,
    "ap_hi": 180,
    "ap_lo": 110,
    "smoke": 1,
    "alco": 1,
    "active": 0,
    "cholesterol_high": 1,
    "gluc_high": 1
}
result_risk = predict_single(super_risk)

print(f"🟢 SUPER SAUDÁVEL: {result_healthy['probability']:.2f}%")
print(f"   (Jovem, IMC ideal, PA ótima, sem fatores de risco)")
print(f"\n🔴 SUPER ALTO RISCO: {result_risk['probability']:.2f}%")
print(f"   (Idoso, obeso, hipertenso, todos fatores de risco)")

diff_extreme = result_risk["probability"] - result_healthy["probability"]
print(f"\n📊 Diferença entre extremos: {diff_extreme:.2f}%")

if diff_extreme > 15:
    print(f"✅ EXCELENTE: Grande separação entre casos extremos")
elif diff_extreme > 10:
    print(f"✅ BOM: Boa separação entre casos extremos")
elif diff_extreme > 5:
    print(f"⚠️  RAZOÁVEL: Separação moderada entre casos extremos")
else:
    print(f"❌ PROBLEMA: Separação insuficiente entre casos extremos!")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("📋 RESUMO DA ANÁLISE")
print("=" * 80)

checks = []

# Check 1: Idade
checks.append(("Idade aumenta risco", is_increasing and age_results[-1] - age_results[0] > 5))

# Check 2: IMC
bmi_range = bmi_results[-1] - bmi_results[0]
checks.append(("IMC aumenta risco", bmi_range > 5))

# Check 3: Pressão Sistólica
ap_hi_range = ap_hi_results[-1] - ap_hi_results[0]
checks.append(("Pressão sistólica aumenta risco", ap_hi_range > 5))

# Check 4: Pressão Diastólica
ap_lo_range = ap_lo_results[-1] - ap_lo_results[0]
checks.append(("Pressão diastólica aumenta risco", ap_lo_range > 3))

# Check 5: Fatores binários
checks.append(("Fatores binários aumentam risco", all_increase))

# Check 6: Atividade física
checks.append(("Atividade física reduz risco", result_active["probability"] < result_inactive["probability"]))

# Check 7: Separação extremos
checks.append(("Boa separação entre extremos", diff_extreme > 10))

# Check 8: Range geral
all_probs = age_results + bmi_results + ap_hi_results + ap_lo_results
overall_range = max(all_probs) - min(all_probs)
checks.append(("Range de variação adequado", overall_range > 15))

print()
passed = sum(1 for _, status in checks if status)
total = len(checks)

for check_name, status in checks:
    emoji = "✅" if status else "❌"
    print(f"{emoji} {check_name}")

print(f"\n{'='*80}")
print(f"🎯 RESULTADO FINAL: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
print(f"{'='*80}")

if passed == total:
    print("🎉 PERFEITO! Modelo responde corretamente a todas as variáveis!")
elif passed >= total * 0.8:
    print("✅ BOM! Modelo responde bem à maioria das variáveis")
elif passed >= total * 0.6:
    print("⚠️  RAZOÁVEL. Modelo tem alguns problemas de sensibilidade")
else:
    print("❌ PROBLEMA! Modelo ainda muito enviesado")

print("=" * 80)
