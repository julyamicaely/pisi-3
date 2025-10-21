from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from model_xgboost import train_xgboost
from pathlib import Path

def plot_feature_importance_xgb(model, features, save_path=None):
    importance = model.feature_importances_

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance, y=features, palette="RdBu_r")
    plt.title("Importância das Variáveis - XGBoost")
    plt.xlabel("Importância")
    plt.ylabel("Variável")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"💾 Gráfico salvo em: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    print("=== Gerando gráfico de importância do XGBoost ===")

    model, X_test, y_test, feature_names = train_xgboost()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_path = results_dir / f"xgb_feature_importance_{timestamp}.png"

    plot_feature_importance_xgb(model, feature_names, plot_path)
