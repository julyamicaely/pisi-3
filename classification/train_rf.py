import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from classification.model_random_forest import train_random_forest

def main():
    print("🚀 Treinando modelo Random Forest...")

    # Treina o modelo
    model, X_test, y_test = train_random_forest()

    # Faz as previsões
    y_pred = model.predict(X_test)

    # Calcula métricas
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Cria pasta de resultados
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Salva os artefatos
    joblib.dump(model, os.path.join(results_dir, "model_rf.joblib"))
    joblib.dump(y_pred, os.path.join(results_dir, "y_pred_rf.joblib"))
    joblib.dump(y_test, os.path.join(results_dir, "y_test_rf.joblib"))
    joblib.dump(cm, os.path.join(results_dir, "cm_rf.joblib"))
    joblib.dump(report, os.path.join(results_dir, "metrics_rf.joblib"))

    print(f"✅ Random Forest treinado e artefatos salvos em: {results_dir}")

if __name__ == "__main__":
    main()
