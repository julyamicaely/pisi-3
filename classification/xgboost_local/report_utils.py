"""
Utilidades para geração de relatórios do XGBoost.
Salva matrizes de confusão, curvas ROC/PR e tabelas de métricas.
Padrão análogo ao Random Forest.
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def save_confusion_matrix(y_true, y_pred, output_dir, prefix="xgboost"):
    """
    Salva matriz de confusão em PNG.
    
    Args:
        y_true: Valores reais
        y_pred: Predições
        output_dir: Diretório de saída
        prefix: Prefixo do arquivo
    
    Returns:
        Path: Caminho do arquivo salvo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calcular matriz
    cm = confusion_matrix(y_true, y_pred)
    
    # Criar figura
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Sem Doença', 'Com Doença'],
        yticklabels=['Sem Doença', 'Com Doença'],
        cbar_kws={'label': 'Contagem'}
    )
    plt.title('Matriz de Confusão - XGBoost', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Predição', fontsize=12)
    plt.tight_layout()
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_confusion_matrix_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Matriz de confusão salva: {filepath}")
    return filepath


def save_roc_curve(y_true, y_proba, output_dir, prefix="xgboost"):
    """
    Salva curva ROC em PNG.
    
    Args:
        y_true: Valores reais
        y_proba: Probabilidades preditas (classe positiva)
        output_dir: Diretório de saída
        prefix: Prefixo do arquivo
    
    Returns:
        Path: Caminho do arquivo salvo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Criar figura
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1E88E5', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC - XGBoost', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_roc_curve_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Curva ROC salva: {filepath}")
    return filepath


def save_precision_recall_curve(y_true, y_proba, output_dir, prefix="xgboost"):
    """
    Salva curva Precision-Recall em PNG.
    
    Args:
        y_true: Valores reais
        y_proba: Probabilidades preditas (classe positiva)
        output_dir: Diretório de saída
        prefix: Prefixo do arquivo
    
    Returns:
        Path: Caminho do arquivo salvo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calcular curva PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    # Criar figura
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#00ACC1', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall (Sensibilidade)', fontsize=12)
    plt.ylabel('Precision (Valor Preditivo Positivo)', fontsize=12)
    plt.title('Curva Precision-Recall - XGBoost', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_precision_recall_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Curva Precision-Recall salva: {filepath}")
    return filepath


def save_metrics_report(y_true, y_pred, y_proba, output_dir, prefix="xgboost", model_params=None):
    """
    Salva relatório completo de métricas em TXT e JSON.
    
    Args:
        y_true: Valores reais
        y_pred: Predições
        y_proba: Probabilidades preditas
        output_dir: Diretório de saída
        prefix: Prefixo do arquivo
        model_params: Parâmetros do modelo (opcional)
    
    Returns:
        tuple: (txt_path, json_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calcular métricas
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='binary')),
        'recall': float(recall_score(y_true, y_pred, average='binary')),
        'f1_score': float(f1_score(y_true, y_pred, average='binary')),
        'roc_auc': float(roc_auc_score(y_true, y_proba))
    }
    
    # Classification report
    class_report = classification_report(y_true, y_pred, target_names=['Sem Doença', 'Com Doença'])
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ======= SALVAR TXT =======
    txt_filename = f"{prefix}_metrics_report_{timestamp}.txt"
    txt_path = output_dir / txt_filename
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE MÉTRICAS - XGBoost\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Amostras: {len(y_true)}\n\n")
        
        if model_params:
            f.write("-" * 70 + "\n")
            f.write("PARÂMETROS DO MODELO\n")
            f.write("-" * 70 + "\n")
            for param, value in sorted(model_params.items()):
                f.write(f"{param:25s}: {value}\n")
            f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("MÉTRICAS PRINCIPAIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.6f}\n")
        f.write(f"Precision: {metrics['precision']:.6f}\n")
        f.write(f"Recall:    {metrics['recall']:.6f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.6f}\n")
        f.write(f"AUC-ROC:   {metrics['roc_auc']:.6f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 70 + "\n")
        f.write(class_report)
        f.write("\n")
    
    print(f"   ✅ Relatório TXT salvo: {txt_path}")
    
    # ======= SALVAR JSON =======
    json_filename = f"{prefix}_metrics_{timestamp}.json"
    json_path = output_dir / json_filename
    
    json_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(y_true),
        'metrics': metrics,
        'model_params': model_params if model_params else {}
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Relatório JSON salvo: {json_path}")
    
    return txt_path, json_path


def generate_full_report(y_true, y_pred, y_proba, output_dir, model_params=None):
    """
    Gera relatório completo: matriz de confusão + curvas + métricas.
    
    Args:
        y_true: Valores reais
        y_pred: Predições
        y_proba: Probabilidades preditas
        output_dir: Diretório de saída
        model_params: Parâmetros do modelo
    
    Returns:
        dict: Caminhos de todos os arquivos salvos
    """
    print("\n📊 Gerando relatórios completos...")
    
    files = {}
    
    # Matriz de confusão
    files['confusion_matrix'] = save_confusion_matrix(y_true, y_pred, output_dir)
    
    # Curva ROC
    files['roc_curve'] = save_roc_curve(y_true, y_proba, output_dir)
    
    # Curva Precision-Recall
    files['pr_curve'] = save_precision_recall_curve(y_true, y_proba, output_dir)
    
    # Relatórios de métricas
    txt_path, json_path = save_metrics_report(y_true, y_pred, y_proba, output_dir, model_params=model_params)
    files['metrics_txt'] = txt_path
    files['metrics_json'] = json_path
    
    print("\n✅ Todos os relatórios gerados com sucesso!")
    
    return files


if __name__ == "__main__":
    print("Este módulo contém funções utilitárias para geração de relatórios.")
    print("Use-o importando as funções em seus scripts de treino.")
