import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from preprocess_data import preprocess_data


def train_random_forest():
    """
    Treina o modelo Random Forest para classificação de doença cardíaca.
    
    ✅ SEM VAZAMENTO DE DADOS:
      - X_train vem já balanceado com SMOTE
      - X_test está escalonado mas NÃO foi balanceado
      - Scaler foi ajustado apenas em X_train
      - Avaliação é feita em dados limpos do teste
    
    Retorna:
        - model: modelo treinado
        - X_test: conjunto de teste escalonado (sem SMOTE)
        - y_test: rótulos reais do teste (sem SMOTE)
        - feature_names: nomes das colunas originais (para análise de importância)
    """

    print("🔧 Iniciando pré-processamento dos dados (sem vazamento)...")
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess_data()

    print("✅ Pré-processamento concluído!")
    print(f"📊 Treinando Random Forest com {X_train.shape[1]} variáveis...")
    print(f"   Treino: {X_train.shape[0]} amostras (balanceadas com SMOTE)")
    print(f"   Teste:  {X_test.shape[0]} amostras (originais, sem SMOTE)")

    # Criar e treinar o modelo com hiperparâmetros otimizados
    model = RandomForestClassifier(
        n_estimators=200,           # ↑ Mais árvores = melhor generalização
        max_depth=15,               # Limita profundidade para evitar overfitting
        min_samples_split=10,       # Mínimo de amostras para dividir nó
        min_samples_leaf=4,         # Mínimo de amostras em folha
        max_features='sqrt',        # Reduz correlação entre árvores
        bootstrap=True,             # Amostragem com reposição
        class_weight='balanced',    # Balanceia classes automaticamente
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)
    print("✅ Treinamento concluído!")

    # ✅ Salvar modelo treinado no diretório correto (models/, não encoders/)
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_model.joblib")
    joblib.dump(model, model_path)
    print(f"💾 Modelo salvo em: {model_path}")

    # ✅ Gerar previsões no conjunto de teste LIMPO (sem SMOTE)
    y_pred = model.predict(X_test)

    # Relatório de classificação
    print("\n📈 Relatório de Classificação (dados de teste limpos):")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão - Random Forest")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()

    # Salvar em reports/ (não results/)
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"💾 Matriz de confusão salva em: {cm_path}")
    plt.close()  # Fechar figura para evitar sobreposição

    # Retorna modelo, dados de teste limpos e nomes das variáveis
    return model, X_test, y_test, feature_names


if __name__ == "__main__":
    print("⚙️ Executando treino standalone...")
    model, X_test, y_test, features = train_random_forest()
    print(f"🏁 Modelo treinado com {len(features)} features.")
    print("✅ Avaliação realizada em dados limpos, sem vazamento de dados!")
