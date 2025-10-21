import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from classification.preprocess_data import preprocess_data


def train_random_forest():
    """
    Treina o modelo Random Forest para classificação de doença cardíaca.
    Retorna:
        - model: modelo treinado
        - X_test: conjunto de teste escalonado
        - y_test: rótulos reais do teste
        - feature_names: nomes das colunas originais (para análise de importância)
    """

    print("🔧 Iniciando pré-processamento dos dados...")
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data()

    # Guardar nomes das colunas originais antes do escalonamento
    feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else None

    print("✅ Pré-processamento concluído!")
    print(f"📊 Treinando modelo Random Forest com {X_train.shape[1]} variáveis...")

    # Criar e treinar o modelo
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("✅ Treinamento concluído!")

    # Salvar modelo treinado
    model_dir = os.path.join(os.path.dirname(__file__), "encoders")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "random_forest_model.joblib"))
    print("💾 Modelo salvo em: encoders/random_forest_model.joblib")

    # Gerar previsões
    y_pred = model.predict(X_test)

    # Relatório de classificação
    print("\n📈 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão - Random Forest")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    print("💾 Matriz de confusão salva em: classification/results/confusion_matrix.png")
    plt.show()

    # Retorna modelo, dados e nomes das variáveis
    return model, X_test, y_test, feature_names


if __name__ == "__main__":
    print("⚙️ Executando treino standalone...")
    model, X_test, y_test, features = train_random_forest()
    print(f"🏁 Modelo treinado com {len(features)} features.")
