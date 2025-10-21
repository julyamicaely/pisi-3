import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_feature_importance(model, X, title="Importância das Variáveis", save_path=None):
    """
    Gera e exibe o gráfico de importância das variáveis do modelo Random Forest.

    Parâmetros:
      - model: modelo RandomForest treinado.
      - X: DataFrame, lista de nomes das features ou array.
      - title: título do gráfico.
      - save_path: caminho opcional para salvar o gráfico (em PNG).
    """

    print("📊 Calculando importância das variáveis...")

    # Obter importâncias
    importances = model.feature_importances_

    # Identificar nomes das variáveis
    if isinstance(X, (list, tuple)):  # se for lista de nomes
        feature_names = X
    elif hasattr(X, "columns"):  # se for DataFrame
        feature_names = X.columns
    else:  # caso contrário, gera nomes genéricos
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    # Verificar consistência de tamanhos
    if len(importances) != len(feature_names):
        print("⚠️ Aviso: número de importâncias não bate com número de features.")
        min_len = min(len(importances), len(feature_names))
        importances = importances[:min_len]
        feature_names = feature_names[:min_len]

    # Criar DataFrame de importâncias
    feat_imp = pd.DataFrame({
        "Variável": feature_names,
        "Importância": importances
    }).sort_values(by="Importância", ascending=False)

    # Plotar gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feat_imp,
        y="Variável",
        x="Importância",
        palette="viridis"
    )

    plt.title(title, fontsize=12)
    plt.xlabel("Nível de importância", fontsize=11)
    plt.ylabel("Variável", fontsize=11)
    plt.tight_layout()

    # Salvar ou mostrar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Gráfico salvo em: {save_path}")

    plt.show()

    return feat_imp
