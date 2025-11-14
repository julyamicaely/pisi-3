import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from pathlib import Path
import joblib

# Adicionar path do projeto
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Importar função de pré-processamento compartilhada
try:
    from preprocess_data import load_and_preprocess_data
    USE_SHARED_PREPROCESSING = True
    print("✅ Pipeline de pré-processamento compartilhado carregado")
except ImportError:
    print("⚠️ Não foi possível carregar pipeline compartilhado. Usando modo legacy.")
    USE_SHARED_PREPROCESSING = False

DATA_PATH = Path(__file__).resolve().parents[2] / "EDA" / "cardio_data.parquet"


def train_xgboost(use_smote=True, random_state=42):
    """
    Treina modelo XGBoost usando pipeline de pré-processamento compartilhado.
    
    Args:
        use_smote: Se True, aplica SMOTE para balanceamento (default: True)
        random_state: Seed para reprodutibilidade (default: 42)
    
    Returns:
        tuple: (model, X_test, y_test, feature_names, metrics)
    """
    print("=== Treinando modelo XGBoost ===")
    
    # USAR PIPELINE COMPARTILHADO (elimina duplicação!)
    if USE_SHARED_PREPROCESSING:
        try:
            # Carregar dados pré-processados com escalonamento e engenharia de features
            X_scaled, X_original, y, feature_names = load_and_preprocess_data()
            
            # Split estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=0.2, 
                random_state=random_state, 
                stratify=y
            )
            
            print(f"📊 Dados carregados via pipeline compartilhado:")
            print(f"   - Features: {len(feature_names)}")
            print(f"   - Train: {X_train.shape}, Test: {X_test.shape}")
            print(f"   - Escalonamento: ✅ Aplicado")
            print(f"   - Engenharia de features: ✅ Aplicada")
            
        except Exception as e:
            print(f"❌ Erro ao usar pipeline compartilhado: {e}")
            print("Voltando para modo legacy...")
            USE_SHARED_PREPROCESSING = False
    
    # FALLBACK: Modo legacy (caso pipeline não esteja disponível)
    if not USE_SHARED_PREPROCESSING:
        df = pd.read_parquet(DATA_PATH)
        df = df.drop(columns=["id", "bp_category", "bp_category_encoded"], errors="ignore")
        
        X = df[["age_years", "ap_hi", "ap_lo", "cholesterol", "gluc", "weight", "height"]]
        y = df["cardio"]
        feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    
    # Treinar modelo XGBoost com parâmetros otimizados
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=random_state,
        eval_metric="logloss"
    )
    
    print("\n🚀 Iniciando treino do XGBoost...")
    model.fit(X_train, y_train)
    print("✅ Treino concluído!")
    
    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calcular métricas detalhadas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'auc_roc': roc_auc_score(y_test, y_proba[:, 1])
    }
    
    # Gerar relatório
    report = classification_report(y_test, y_pred)
    print("\n📊 Relatório de Classificação:\n")
    print(report)
    print(f"\n🎯 Métricas Resumidas:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Criar diretórios necessários
    results_dir = Path(__file__).resolve().parents[1] / "reports"
    results_dir.mkdir(exist_ok=True)
    
    # Gerar timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Salvar relatório com timestamp
    report_path = results_dir / f"xgboost_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE CLASSIFICAÇÃO - XGBoost\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pipeline: {'Compartilhado (preprocess_data.py)' if USE_SHARED_PREPROCESSING else 'Legacy'}\n")
        f.write(f"Random State: {random_state}\n")
        f.write(f"Número de Features: {len(feature_names)}\n")
        f.write(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}\n\n")
        f.write("-" * 60 + "\n")
        f.write("MÉTRICAS PRINCIPAIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC:   {metrics['auc_roc']:.4f}\n\n")
        f.write("-" * 60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(report)
    
    print(f"\n💾 Relatório salvo em: {report_path}")
    
    return model, X_test, y_test, feature_names, metrics


if __name__ == "__main__":
    model, X_test, y_test, features, metrics = train_xgboost()
    print("\n✅ Treino completo!")
