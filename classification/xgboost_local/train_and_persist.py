"""
Script principal de treino do XGBoost com persistência completa.
Salva modelo, pipeline, metadados e relatórios.
Convenção consistente com Random Forest.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.model_selection import train_test_split

# Adicionar path do projeto
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Importar funções compartilhadas
from preprocess_data import load_and_preprocess_data
from xgboost_local.report_utils import generate_full_report
from xgboost import XGBClassifier


def clean_old_reports(reports_dir, keep_latest=1):
    """
    Remove relatórios antigos do XGBoost, mantendo apenas os mais recentes.
    
    Args:
        reports_dir (Path): Diretório dos relatórios
        keep_latest (int): Número de relatórios mais recentes a manter (padrão: 1)
    """
    patterns = [
        'xgboost_confusion_matrix_*.png',
        'xgboost_roc_curve_*.png',
        'xgboost_precision_recall_*.png',
        'xgboost_metrics_report_*.txt',
        'xgboost_metrics_*.json',
        'xgboost_feature_importance_*.png',
        'xgboost_feature_importance_*.json'
    ]
    
    print("\n🗑️  Limpando relatórios antigos...")
    removed_count = 0
    
    for pattern in patterns:
        # Buscar arquivos que correspondem ao padrão
        files = sorted(glob.glob(str(reports_dir / pattern)))
        
        # Remover todos exceto os keep_latest mais recentes
        if len(files) > keep_latest:
            files_to_remove = files[:-keep_latest]
            
            for file in files_to_remove:
                try:
                    Path(file).unlink()
                    removed_count += 1
                    print(f"   🗑️  Removido: {Path(file).name}")
                except Exception as e:
                    print(f"   ⚠️  Erro ao remover {Path(file).name}: {e}")
    
    if removed_count > 0:
        print(f"   ✅ {removed_count} arquivo(s) antigo(s) removido(s)")
    else:
        print("   ℹ️  Nenhum arquivo antigo para remover")


def clean_old_model_backups(models_dir, keep_latest=3):
    """
    Remove backups antigos de modelos, mantendo os 3 mais recentes.
    
    Args:
        models_dir (Path): Diretório dos modelos
        keep_latest (int): Número de backups a manter (padrão: 3)
    """
    pattern = 'xgboost_model_*.joblib'
    files = sorted(glob.glob(str(models_dir / pattern)))
    
    if len(files) > keep_latest:
        files_to_remove = files[:-keep_latest]
        removed_count = 0
        
        for file in files_to_remove:
            try:
                Path(file).unlink()
                removed_count += 1
                print(f"   🗑️  Backup removido: {Path(file).name}")
            except Exception as e:
                print(f"   ⚠️  Erro ao remover {Path(file).name}: {e}")
        
        if removed_count > 0:
            print(f"   ✅ {removed_count} backup(s) antigo(s) de modelo removido(s)")



def save_feature_importance(model, feature_names, output_dir):
    """
    Gera e salva gráfico de feature importance.
    
    Args:
        model: Modelo XGBoost treinado
        feature_names: Lista de nomes das features
        output_dir: Diretório de saída
    
    Returns:
        Path: Caminho do arquivo salvo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Obter importâncias
    importances = model.feature_importances_
    
    # Criar DataFrame para ordenar
    import pandas as pd
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Criar figura
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(feature_names))
    
    plt.barh(
        range(len(feature_importance_df)),
        feature_importance_df['importance'],
        color=colors
    )
    
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Importância', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Feature Importance - XGBoost', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, v in enumerate(feature_importance_df['importance']):
        plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"xgboost_feature_importance_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Feature Importance salva: {filepath}")
    
    # Também salvar em JSON
    json_filename = f"xgboost_feature_importance_{timestamp}.json"
    json_path = output_dir / json_filename
    
    importance_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features': feature_importance_df.to_dict('records')
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(importance_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Feature Importance JSON salva: {json_path}")
    
    return filepath


def train_and_save_xgboost(random_state=42, use_best_params=True, use_full_dataset=True):
    """
    Treina XGBoost e salva modelo completo com metadados.
    
    Args:
        random_state: Seed para reprodutibilidade
        use_best_params: Se True, carrega melhores parâmetros do Optuna
        use_full_dataset: Se True, treina com todo o dataset (sem split)
    
    Returns:
        dict: Informações do modelo salvo
    """
    print("=" * 70)
    print("TREINO E PERSISTÊNCIA DO XGBOOST")
    print("=" * 70)
    
    # Carregar dados pré-processados
    print("\n📊 Carregando dados via pipeline compartilhado...")
    X_scaled, X_original, y, feature_names = load_and_preprocess_data()
    print(f"   ✅ Dados carregados: {X_scaled.shape}")
    
    # Decidir se usa split ou dataset completo
    if use_full_dataset:
        print(f"   🎯 Modo: DATASET COMPLETO (sem split)")
        X_train = X_scaled
        y_train = y
        X_test = X_scaled  # Usar mesmo dataset para avaliação
        y_test = y
    else:
        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=random_state,
            stratify=y
        )
        print(f"   📊 Modo: Train/Test Split")
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
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
                params['random_state'] = random_state
                print(f"   ✅ Parâmetros otimizados carregados")
                print(f"   ✅ AUC-ROC esperado: {best_params_data['best_score']:.4f}")
        else:
            print(f"\n⚠️ Arquivo de parâmetros não encontrado. Usando padrões.")
    
    print(f"\n📋 Parâmetros do modelo:")
    for param, value in sorted(params.items()):
        print(f"   {param:20s}: {value}")
    
    # Treinar modelo
    print(f"\n🚀 Treinando XGBoost...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    print(f"   ✅ Treino concluído!")
    
    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Criar diretórios
    models_dir = Path(__file__).resolve().parents[1] / "models"
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Timestamp para todos os arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========== SALVAR MODELO ==========
    print(f"\n💾 Salvando modelo e artefatos...")
    
    # 1. Modelo XGBoost
    model_path = models_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"   ✅ Modelo salvo: {model_path}")
    
    # 2. Modelo com timestamp (backup)
    model_backup_path = models_dir / f"xgboost_model_{timestamp}.joblib"
    joblib.dump(model, model_backup_path)
    print(f"   ✅ Backup do modelo salvo: {model_backup_path}")
    
    # Limpar backups antigos (manter apenas 3 mais recentes)
    clean_old_model_backups(models_dir, keep_latest=3)
    
    # 3. Nomes das features
    features_path = models_dir / "xgboost_features.json"
    features_data = {
        'features': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
        'n_features': len(feature_names)
    }
    with open(features_path, 'w') as f:
        json.dump(features_data, f, indent=2)
    print(f"   ✅ Features salvas: {features_path}")
    
    # 4. Metadados completos
    metadata = {
        'model_info': {
            'model_type': 'XGBoost',
            'library': 'xgboost',
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': timestamp,
            'random_state': random_state,
            'trained_on_full_dataset': use_full_dataset
        },
        'data_info': {
            'n_samples_total': len(X_scaled),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': len(feature_names),
            'features': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
            'preprocessing_pipeline': 'classification.preprocess_data.load_and_preprocess_data()',
            'full_dataset_training': use_full_dataset
        },
        'model_params': params,
        'performance': {
            'test_size': len(X_test),
            'test_accuracy': float((y_pred == y_test).mean())
        },
        'files': {
            'model': str(model_path),
            'model_backup': str(model_backup_path),
            'features': str(features_path),
            'metadata': str(models_dir / f"xgboost_metadata_{timestamp}.json")
        }
    }
    
    metadata_path = models_dir / f"xgboost_metadata_{timestamp}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Metadados salvos: {metadata_path}")
    
    # ========== LIMPAR RELATÓRIOS ANTIGOS ==========
    clean_old_reports(reports_dir, keep_latest=1)
    
    # ========== GERAR RELATÓRIOS ==========
    print(f"\n📊 Gerando relatórios completos...")
    report_files = generate_full_report(
        y_test, y_pred, y_proba[:, 1],
        output_dir=reports_dir,
        model_params=params
    )
    
    # ========== GERAR FEATURE IMPORTANCE ==========
    print(f"\n📊 Gerando gráfico de Feature Importance...")
    feature_importance_path = save_feature_importance(
        model, feature_names, reports_dir
    )
    report_files['feature_importance'] = feature_importance_path
    
    # ========== RESUMO FINAL ==========
    print("\n" + "=" * 70)
    print("RESUMO DA PERSISTÊNCIA")
    print("=" * 70)
    print(f"\n📁 Arquivos salvos em classification/models/:")
    print(f"   - xgboost_model.joblib (modelo principal)")
    print(f"   - xgboost_model_{timestamp}.joblib (backup)")
    print(f"   - xgboost_features.json (lista de features)")
    print(f"   - xgboost_metadata_{timestamp}.json (metadados completos)")
    
    print(f"\n📁 Relatórios salvos em classification/reports/:")
    for report_type, filepath in report_files.items():
        print(f"   - {filepath.name}")
    
    print(f"\n✅ Persistência completa! Modelo pronto para produção.")
    
    return {
        'model': model,
        'metadata': metadata,
        'report_files': report_files,
        'test_data': (X_test, y_test),
        'predictions': (y_pred, y_proba)
    }


if __name__ == "__main__":
    # Treinar e salvar modelo com DATASET COMPLETO
    results = train_and_save_xgboost(
        random_state=42, 
        use_best_params=True,
        use_full_dataset=True  # USAR TODO O DATASET
    )
    print("\n🎉 Processo completo finalizado!")
