"""Verificar a ordem esperada das features"""

from classification.prediction_service import load_model
import pandas as pd

pipeline = load_model()

# Ver nomes esperados pelo scaler
scaler = pipeline.named_steps['scaler']
print("Feature names esperados pelo scaler:")
print(scaler.feature_names_in_)

print("\n" + "=" * 70)
print("Testando predição com ordem correta:")

# Criar dados na ordem EXATA
correct_order = scaler.feature_names_in_.tolist()
print(f"Ordem correta: {correct_order}")

# Testar dois perfis completamente diferentes
test_data = pd.DataFrame([
    [0, 110, 70, 0, 0, 1, 25, 20.0, 0, 0],  # jovem saudável
    [1, 180, 110, 1, 1, 0, 70, 35.0, 1, 1]  # idoso alto risco
], columns=correct_order)

probas = pipeline.predict_proba(test_data)[:, 1]

print(f"\nJovem saudável: {probas[0]*100:.2f}%")
print(f"Idoso alto risco: {probas[1]*100:.2f}%")
print(f"Diferença: {abs(probas[0] - probas[1])*100:.2f}%")

if probas[0] == probas[1]:
    print("\n⚠️ PROBLEMA CONFIRMADO: Predições idênticas!")
else:
    print("\n✅ Predições diferentes - modelo funciona!")
