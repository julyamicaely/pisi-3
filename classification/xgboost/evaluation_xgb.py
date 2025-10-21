from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Calcula métricas básicas e retorna como dicionário."""
    y_pred = model.predict(X_test)
    return {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred),
        "Revocação": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    }

def cross_validate_model(model, X, y, cv=5):
    """Executa validação cruzada simples."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"Validação cruzada ({cv}-fold): média = {np.mean(scores):.3f} | desvio = {np.std(scores):.3f}")
    return scores
