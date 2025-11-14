"""
Otimização de Hiperparâmetros do XGBoost usando Optuna.
Busca os melhores parâmetros e salva resultados em classification/reports/.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

# Adicionar path do projeto
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Importar função de pré-processamento compartilhada
from preprocess_data import load_and_preprocess_data


def objective(trial, X, y):
    """
    Função objetivo para otimização Optuna.
    
    Args:
        trial: Trial do Optuna
        X: Features
        y: Target
    
    Returns:
        float: Score médio de validação cruzada
    """
    # Definir espaço de busca de hiperparâmetros
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # Criar modelo
    model = XGBClassifier(**params)
    
    # Validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return scores.mean()


def optimize_xgboost(n_trials=100, timeout=3600):
    """
    Executa otimização de hiperparâmetros usando Optuna.
    
    Args:
        n_trials: Número de trials (default: 100)
        timeout: Timeout em segundos (default: 3600 = 1h)
    
    Returns:
        dict: Melhores parâmetros e métricas
    """
    print("=" * 70)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS - XGBoost (Optuna)")
    print("=" * 70)
    
    # Carregar dados pré-processados
    print("\n📊 Carregando dados via pipeline compartilhado...")
    X_scaled, X_original, y, feature_names = load_and_preprocess_data()
    print(f"   ✅ Dados carregados: {X_scaled.shape}")
    
    # Criar estudo Optuna
    print(f"\n🔍 Iniciando busca de hiperparâmetros...")
    print(f"   - Trials: {n_trials}")
    print(f"   - Timeout: {timeout}s ({timeout/60:.1f} min)")
    print(f"   - Métrica: AUC-ROC")
    print(f"   - CV: 5-fold Stratified")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Executar otimização
    study.optimize(
        lambda trial: objective(trial, X_scaled, y),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Resultados
    best_params = study.best_params
    best_score = study.best_value
    
    print("\n" + "=" * 70)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("=" * 70)
    print(f"\n🏆 Melhor AUC-ROC (CV): {best_score:.6f}")
    print(f"\n📋 Melhores Hiperparâmetros:")
    for param, value in sorted(best_params.items()):
        print(f"   {param:20s}: {value}")
    
    # Preparar dados para salvamento
    results = {
        'optimization_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_trials': n_trials,
            'timeout_seconds': timeout,
            'n_features': len(feature_names),
            'n_samples': len(X_scaled),
            'metric': 'roc_auc',
            'cv_folds': 5
        },
        'best_score': float(best_score),
        'best_params': best_params,
        'all_trials': []
    }
    
    # Adicionar informações de todos os trials
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'value': float(trial.value) if trial.value is not None else None,
            'params': trial.params,
            'state': str(trial.state)
        }
        results['all_trials'].append(trial_info)
    
    # Salvar resultados em JSON
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"xgboost_optuna_results_{timestamp}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: {json_path}")
    
    # Salvar também arquivo com melhores parâmetros (sobrescreve)
    best_params_path = reports_dir / "xgboost_best_params.json"
    best_params_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_score': float(best_score),
        'best_params': best_params,
        'source': 'optuna_optimization'
    }
    
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Melhores parâmetros salvos em: {best_params_path}")
    
    return results


if __name__ == "__main__":
    # Executar otimização (100 trials ou 1 hora)
    results = optimize_xgboost(n_trials=100, timeout=3600)
    print("\n✅ Otimização completa!")
