"""
Teste específico para verificar se SHAP está carregando o modelo correto
"""
import sys
from pathlib import Path

# Adicionar paths
sys.path.append(str(Path(__file__).parent / "dashboard"))
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("🧪 TESTE: Dashboard Random Forest - SHAP Model Loading")
print("=" * 80)

# Importar a função load_data
from dashboard.pages.random_forest import load_data, rf_data

print("\n📊 Testando carregamento de dados...")
print("-" * 80)

# Verificar dados carregados
print(f"\n✓ Modelo carregado: {rf_data['model'] is not None}")
print(f"✓ Pipeline carregado: {hasattr(rf_data.get('model'), 'named_steps') if rf_data['model'] else False}")
print(f"✓ Dados X_test: {rf_data['X_test'] is not None}")
print(f"✓ Dados y_test: {rf_data['y_test'] is not None}")
print(f"✓ Predições: {rf_data['y_pred'] is not None}")
print(f"✓ Probabilidades: {rf_data['y_proba'] is not None}")
print(f"✓ SHAP values: {rf_data['shap_values'] is not None}")

# Verificar tipo do modelo
if rf_data['model'] is not None:
    model = rf_data['model']
    model_type = type(model).__name__
    print(f"\n🔍 Tipo do modelo: {model_type}")
    
    # Se for pipeline, mostrar steps
    if hasattr(model, 'named_steps'):
        print("\n📦 Pipeline Steps:")
        for step_name, step in model.named_steps.items():
            print(f"  - {step_name}: {type(step).__name__}")
    
    # Verificar se é RandomForestClassifier
    if hasattr(model, 'n_estimators'):
        print(f"\n🌲 RandomForest n_estimators: {model.n_estimators}")
        print(f"🌲 RandomForest max_depth: {model.max_depth}")

# Verificar métricas
if rf_data.get('metrics'):
    print("\n📈 Métricas do modelo:")
    for metric_name, metric_value in rf_data['metrics'].items():
        if isinstance(metric_value, float):
            print(f"  - {metric_name}: {metric_value:.4f}")
        else:
            print(f"  - {metric_name}: {metric_value}")

# Verificar shape dos dados
if rf_data['X_test'] is not None:
    print(f"\n📊 Shape X_test: {rf_data['X_test'].shape}")
    print(f"📊 Shape y_test: {rf_data['y_test'].shape}")
    
    if rf_data['y_pred'] is not None:
        print(f"📊 Shape y_pred: {rf_data['y_pred'].shape}")
    
    if rf_data['y_proba'] is not None:
        print(f"📊 Shape y_proba: {rf_data['y_proba'].shape}")

# Verificar SHAP
if rf_data['shap_values'] is not None:
    print(f"\n🔮 SHAP values shape: {rf_data['shap_values'].shape}")
    print(f"🔮 SHAP base value: {rf_data['shap_base_value']}")
    print(f"🔮 SHAP sample size: {len(rf_data['X_sample'])}")

print("\n" + "=" * 80)
print("✅ Teste concluído!")
print("=" * 80)
