"""
Treino do XGBoost com Validação Cruzada Estratificada (K-Fold).
Calcula métricas médias e desvios padrão, salvando em JSON.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

# Adicionar path do projeto
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Importar função de pré-processamento compartilhada
from preprocess_data import load_and_preprocess_data


def train_xgboost_with_cv(n_splits=5, random_state=42, use_best_params=True):
    """
    Treina XGBoost com validação cruzada estratificada.
    
    Args:
        n_splits: Número de folds (default: 5)
        random_state: Seed para reprodutibilidade
        use_best_params: Se True, tenta carregar melhores parâmetros do Optuna
    
    Returns:
        dict: Resultados completos com métricas e modelo final
    """
    print("=" * 70)
    print("TREINO DO XGBOOST COM VALIDAÇÃO CRUZADA ESTRATIFICADA")
    print("=" * 70)
    
    # Carregar dados pré-processados
    print("\n📊 Carregando dados via pipeline compartilhado...")
    X_scaled, X_original, y, feature_names = load_and_preprocess_data()
    print(f"   ✅ Dados carregados: {X_scaled.shape}")
    print(f"   ✅ Features: {len(feature_names)}")
    
    # Definir parâmetros do modelo
    params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'min_child_weight': 1,
        'random_state': random_state,
        'eval_metric': 'logloss'
    }
    
    # Tentar carregar melhores parâmetros do Optuna
    if use_best_params:
        best_params_path = Path(__file__).resolve().parents[1] / "reports" / "xgboost_best_params.json"
        if best_params_path.exists():
            print(f"\n🔍 Carregando melhores parâmetros do Optuna...")
            with open(best_params_path, 'r') as f:
                best_params_data = json.load(f)
                params.update(best_params_data['best_params'])
                params['random_state'] = random_state  # Manter seed consistente
                print(f"   ✅ Parâmetros otimizados carregados")
        else:
            print(f"\n⚠️ Arquivo de parâmetros não encontrado. Usando padrões.")
    
    print(f"\n📋 Parâmetros do modelo:")
    for param, value in sorted(params.items()):
        print(f"   {param:20s}: {value}")
    
    # Criar modelo
    model = XGBClassifier(**params)
    
    # Definir métricas para CV
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary'),
        'roc_auc': make_scorer(roc_auc_score)
    }
    
    # Configurar validação cruzada estratificada
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    print(f"\n🔄 Executando validação cruzada ({n_splits}-fold estratificada)...")
    print(f"   Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC")
    
    # Executar cross-validation
    cv_results = cross_validate(
        model, X_scaled, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        verbose=1
    )
    
    # Calcular estatísticas
    results = {
        'cv_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_splits': n_splits,
            'random_state': random_state,
            'n_samples': len(X_scaled),
            'n_features': len(feature_names),
            'stratified': True
        },
        'model_params': params,
        'cv_metrics': {}
    }
    
    print("\n" + "=" * 70)
    print("RESULTADOS DA VALIDAÇÃO CRUZADA")
    print("=" * 70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results['cv_metrics'][metric] = {
            'test_mean': float(np.mean(test_scores)),
            'test_std': float(np.std(test_scores)),
            'test_scores': [float(s) for s in test_scores],
            'train_mean': float(np.mean(train_scores)),
            'train_std': float(np.std(train_scores)),
            'train_scores': [float(s) for s in train_scores]
        }
        
        print(f"\n{metric.upper().replace('_', '-')}:")
        print(f"   Test:  {np.mean(test_scores):.4f} (+/- {np.std(test_scores):.4f})")
        print(f"   Train: {np.mean(train_scores):.4f} (+/- {np.std(train_scores):.4f})")
        print(f"   Folds: {[f'{s:.4f}' for s in test_scores]}")
    
    # Treinar modelo final com todos os dados
    print(f"\n🚀 Treinando modelo final com todos os dados...")
    model.fit(X_scaled, y)
    print(f"   ✅ Modelo final treinado!")
    
    # Salvar resultados em JSON
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"xgboost_cv_results_{timestamp}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: {json_path}")
    
    # Salvar modelo final
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "xgboost_model.joblib"
    
    joblib.dump(model, model_path)
    print(f"💾 Modelo final salvo em: {model_path}")
    
    return {
        'model': model,
        'cv_results': results,
        'feature_names': feature_names,
        'X': X_scaled,
        'y': y
    }


if __name__ == "__main__":
    # Executar treino com validação cruzada
    results = train_xgboost_with_cv(n_splits=5, random_state=42, use_best_params=True)
    print("\n✅ Treino com validação cruzada completo!")
