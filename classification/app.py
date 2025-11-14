import os
from sklearn.metrics import classification_report
from model_random_forest import train_random_forest
from feature_importance import plot_feature_importance


def main():
    print("=== Iniciando pipeline de classificação (SEM VAZAMENTO DE DADOS) ===")

    # 1️⃣ Treinar modelo e obter nomes originais das variáveis
    # ✅ O pipeline agora garante que:
    #    - train_test_split foi feito ANTES do scaler e SMOTE
    #    - Scaler foi ajustado apenas em X_train
    #    - SMOTE foi aplicado apenas em X_train
    #    - Teste permanece limpo e não contaminado
    model, X_test, y_test, feature_names = train_random_forest()
    print("✅ Modelo Random Forest treinado com sucesso!")

    # 2️⃣ Fazer previsões em dados de teste LIMPOS
    print("🔍 Gerando previsões no conjunto de teste (dados limpos)...")
    y_pred = model.predict(X_test)

    # 3️⃣ Exibir relatório de classificação
    print("\n📊 Relatório de Classificação (avaliação confiável):")
    report = classification_report(y_test, y_pred)
    print(report)

    # 4️⃣ Salvar relatório em arquivo (agora em reports/)
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Relatório de Classificação (Pipeline Corrigido - Sem Vazamento) ===\n\n")
        f.write("MUDANÇAS APLICADAS:\n")
        f.write("✅ train_test_split executado ANTES de RobustScaler e SMOTE\n")
        f.write("✅ RobustScaler ajustado apenas em X_train\n")
        f.write("✅ SMOTE aplicado apenas em X_train, y_train\n")
        f.write("✅ Teste avaliado com dados originais (sem SMOTE)\n")
        f.write("✅ Modelo salvo em models/ (não mais em encoders/)\n\n")
        f.write(report)

    print(f"💾 Relatório salvo em: {report_path}")

    # 5️⃣ Gerar gráfico de importância das variáveis
    print("\n📈 Gerando gráfico de importância das variáveis...")
    feature_plot_path = os.path.join(reports_dir, "feature_importance.png")

    # Chamada do gráfico com nomes originais das features
    plot_feature_importance(
        model,
        X=feature_names,  # nomes originais vindos do preprocess
        title="Importância das Variáveis - Random Forest (Pipeline Corrigido)",
        save_path=feature_plot_path
    )

    print(f"💾 Gráfico salvo em: {feature_plot_path}")
    print("\n🎯 Pipeline de classificação finalizado com sucesso!")
    print("✅ AVALIAÇÃO CONFIÁVEL: métricas calculadas em dados de teste limpos, sem vazamento!")


if __name__ == "__main__":
    main()
