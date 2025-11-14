from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    print("📊 Avaliando modelo...")
    y_pred = model.predict(X_test)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão - Random Forest")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()


def compute_validation_metrics(y_true, y_pred, y_proba=None):
    """
    Calcula métricas de validação abrangentes para classificação binária.
    
    Args:
        y_true (array): Labels verdadeiros
        y_pred (array): Labels preditos
        y_proba (array, optional): Probabilidades da classe positiva
    
    Returns:
        dict: Dicionário com todas as métricas calculadas
    """
    metrics = {
        'accuracy': (y_true == y_pred).mean(),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Adicionar AUC-ROC se probabilidades forem fornecidas
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    # Calcular métricas por classe
    from sklearn.metrics import precision_recall_fscore_support
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    metrics['precision_class_0'] = precision_per_class[0]
    metrics['precision_class_1'] = precision_per_class[1]
    metrics['recall_class_0'] = recall_per_class[0]
    metrics['recall_class_1'] = recall_per_class[1]
    metrics['f1_class_0'] = f1_per_class[0]
    metrics['f1_class_1'] = f1_per_class[1]
    metrics['support_class_0'] = support[0]
    metrics['support_class_1'] = support[1]
    
    return metrics
