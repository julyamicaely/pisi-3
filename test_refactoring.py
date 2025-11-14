"""
Script de teste rápido para verificar as novas funções implementadas.
Execute: python test_refactoring.py
"""

import sys
from pathlib import Path

# Adicionar paths
sys.path.append(str(Path(__file__).parent))

print("=" * 60)
print("🧪 TESTE DE REFATORAÇÃO - Random Forest Dashboard")
print("=" * 60)

# ==================== TESTE 1: Pré-processamento compartilhado ====================
print("\n1️⃣ Testando load_and_preprocess_data()...")
try:
    from classification.preprocess_data import load_and_preprocess_data
    
    X_scaled, X_original, y, feature_names = load_and_preprocess_data()
    
    print(f"   ✅ Dados carregados com sucesso!")
    print(f"   📊 Shape X_scaled: {X_scaled.shape}")
    print(f"   📊 Shape X_original: {X_original.shape}")
    print(f"   📊 Shape y: {y.shape}")
    print(f"   📋 Features: {feature_names}")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")

# ==================== TESTE 2: Cálculo de métricas ====================
print("\n2️⃣ Testando compute_validation_metrics()...")
try:
    from classification.evaluation import compute_validation_metrics
    import joblib
    
    # Carregar modelo
    model_path = Path(__file__).parent / "classification" / "models" / "random_forest_model.joblib"
    model = joblib.load(model_path)
    
    # Fazer predições
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Calcular métricas
    metrics = compute_validation_metrics(y, y_pred, y_proba)
    
    print(f"   ✅ Métricas calculadas com sucesso!")
    print(f"   📈 Accuracy: {metrics['accuracy']:.3f}")
    print(f"   📈 Precision: {metrics['precision']:.3f}")
    print(f"   📈 Recall: {metrics['recall']:.3f}")
    print(f"   📈 F1-Score: {metrics['f1']:.3f}")
    print(f"   📈 AUC-ROC: {metrics['auc_roc']:.3f}")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")

# ==================== TESTE 3: Componentes de visualização ====================
print("\n3️⃣ Testando componentes de visualização...")
try:
    from dashboard.components.cards import (
        build_roc_curve,
        build_precision_recall_curve,
        build_calibration_curve
    )
    
    # Testar ROC
    roc_component = build_roc_curve(y, y_proba, title="Test ROC")
    print(f"   ✅ Componente ROC criado: {type(roc_component).__name__}")
    
    # Testar PR
    pr_component = build_precision_recall_curve(y, y_proba, title="Test PR")
    print(f"   ✅ Componente PR criado: {type(pr_component).__name__}")
    
    # Testar Calibration
    calib_component = build_calibration_curve(y, y_proba, n_bins=10, title="Test Calib")
    print(f"   ✅ Componente Calibration criado: {type(calib_component).__name__}")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")

# ==================== TESTE 4: Importação do dashboard ====================
print("\n4️⃣ Testando importação do dashboard refatorado...")
try:
    # Adicionar path do dashboard
    sys.path.append(str(Path(__file__).parent / "dashboard"))
    
    # Tentar importar (não executar, só verificar sintaxe)
    from pages import random_forest
    
    print(f"   ✅ Dashboard importado com sucesso!")
    print(f"   📄 Módulo: {random_forest.__name__}")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")

# ==================== RESUMO ====================
print("\n" + "=" * 60)
print("✅ TESTE CONCLUÍDO!")
print("=" * 60)
print("\n📝 Próximos passos:")
print("   1. Execute o dashboard: cd dashboard && python app.py")
print("   2. Acesse: http://localhost:8050/random-forest")
print("   3. Teste as 4 abas:")
print("      - Visão Geral")
print("      - Curvas de Performance")
print("      - Interpretação")
print("      - Análise Exploratória")
print("\n🎯 Verifique os tooltips nas métricas (passe o mouse)!")
