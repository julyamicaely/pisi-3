import os
from sklearn.metrics import classification_report
from classification.model_random_forest import train_random_forest
from classification.feature_importance import plot_feature_importance


def main():
    print("=== Iniciando pipeline de classificação ===")

    # 1️⃣ Treinar modelo e obter nomes originais das variáveis
    model, X_test, y_test, feature_names = train_random_forest()
    print("✅ Modelo Random Forest treinado com sucesso!")

    # 2️⃣ Fazer previsões
    print("🔍 Gerando previsões no conjunto de teste...")
    y_pred = model.predict(X_test)

    # 3️⃣ Exibir relatório de classificação
    print("\n📊 Relatório de Classificação:")
    report = classification_report(y_test, y_pred)
    print(report)

    # 4️⃣ Salvar relatório em arquivo
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Relatório de Classificação ===\n\n")
        f.write(report)

    print(f"💾 Relatório salvo em: {report_path}")

    # 5️⃣ Gerar gráfico de importância das variáveis
    print("\n📈 Gerando gráfico de importância das variáveis...")
    feature_plot_path = os.path.join(results_dir, "feature_importance.png")

    # Chamada do gráfico com nomes originais das features
    plot_feature_importance(
        model,
        X=feature_names,  # nomes originais vindos do preprocess
        title="Importância das Variáveis - Random Forest",
        save_path=feature_plot_path
    )

    print(f"💾 Gráfico salvo em: {feature_plot_path}")
    print("\n🎯 Pipeline de classificação finalizado com sucesso!")


if __name__ == "__main__":
    main()
