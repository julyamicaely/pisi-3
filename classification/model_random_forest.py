import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from classification.preprocess_data import preprocess_data
import json
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time


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

    # This function now runs a tuning flow by default. Use run_tuning.py for a dedicated run.
    print("🔧 Iniciando pré-processamento dos dados (sem vazamento)...")
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess_data()

    print("✅ Pré-processamento concluído!")
    print(f"📊 Dados para treino prontos: {X_train.shape[0]} amostras; teste: {X_test.shape[0]} amostras")

    # If a tuned model exists, load it; otherwise perform a tuning and save best model
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    best_model_path = os.path.join(model_dir, "random_forest_model.joblib")

    if os.path.exists(best_model_path):
        print(f"🔁 Modelo ajustado encontrado em {best_model_path}, carregando...")
        model = joblib.load(best_model_path)
    else:
        print("⚙️ Nenhum modelo ajustado encontrado — iniciando tuning do Random Forest...")
        model, tuning_info = run_rf_tuning(X_train, y_train, feature_names)
        # Save tuning results
        results_json = os.path.join(reports_dir, "random_forest_tuning_results.json")
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(tuning_info, f, indent=2)
        # Save markdown summary
        md_path = os.path.join(reports_dir, "random_forest_tuning_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(tuning_info.get("summary_md", ""))
        print(f"💾 Tuning results saved: {results_json}")
        print(f"💾 Summary saved: {md_path}")

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


def run_rf_tuning(X_train, y_train, feature_names, random_state=42, n_iter=40, cv_folds=5):
    """
    Realiza tuning de RandomForest usando RandomizedSearchCV e StratifiedKFold.

    Retorna o melhor estimador e um dicionário com resultados detalhados.
    """
    from sklearn.ensemble import RandomForestClassifier

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 6, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    base_clf = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rnd = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=random_state,
        verbose=2,
        n_jobs=-1,
        return_train_score=False
    )

    start = time.time()
    rnd.fit(X_train, y_train)
    elapsed = time.time() - start

    best = rnd.best_estimator_
    best_params = rnd.best_params_

    # Cross-validate best estimator to collect metrics per fold
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': 'roc_auc'
    }
    cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

    summary = {
        'best_params': best_params,
        'best_score': float(rnd.best_score_),
        'n_iter': n_iter,
        'cv_folds': cv_folds,
        'random_state': random_state,
        'elapsed_seconds': elapsed,
        'cv_metrics': {k: [float(v) for v in cv_res[k]] for k in cv_res}
    }

    # Create human-readable markdown summary
    md_lines = [
        "# Random Forest Tuning Summary\n",
        f"**Best ROC-AUC (cv)**: {summary['best_score']:.6f}\n",
        f"**Best parameters**: {json.dumps(best_params)}\n",
        f"**Random state**: {random_state}\n",
        f"**Elapsed (s)**: {elapsed:.1f}\n",
        "\n## CV metrics per fold\n"
    ]

    # Collect average metrics
    for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']:
        if metric in cv_res:
            vals = cv_res[metric]
            md_lines.append(f"- **{metric}**: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}\n")

    summary_md = "".join(md_lines)
    summary['summary_md'] = summary_md

    # Persist best model
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best, os.path.join(models_dir, 'random_forest_model.joblib'))

    return best, summary


if __name__ == "__main__":
    print("⚙️ Executando treino standalone...")
    model, X_test, y_test, features = train_random_forest()
    print(f"🏁 Modelo treinado com {len(features)} features.")
    print("✅ Avaliação realizada em dados limpos, sem vazamento de dados!")
