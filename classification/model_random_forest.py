import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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


def train_production_pipeline(use_tuning=False, n_iter=10, cv=3):
    """
    Cria e treina um pipeline de produção completo e portável para Random Forest.
    
    📦 Pipeline contém:
       - RobustScaler (normalização robusta a outliers)
       - RandomForestClassifier (com/sem tuning de hiperparâmetros)
    
    ⚠️ IMPORTANTE: 
       - O pipeline NÃO inclui SMOTE (balanceamento é apenas para treino)
       - O pipeline NÃO inclui remoção de outliers (já feito no preprocess)
       - Use apenas componentes padrão do scikit-learn (portabilidade)
    
    Args:
        use_tuning (bool): Se True, executa RandomizedSearchCV
        n_iter (int): Número de iterações do tuning
        cv (int): Número de folds da validação cruzada
        
    Returns:
        tuple: (pipeline, X_test, y_test, feature_names, tuning_summary)
    """
    
    print("=" * 70)
    print("🏭 CRIANDO PIPELINE DE PRODUÇÃO - RANDOM FOREST")
    print("=" * 70)
    
    # Obter dados pré-processados (SEM escalonamento - pipeline fará isso)
    from classification.preprocess_data import preprocess_data_for_pipeline
    X_train, X_test, y_train, y_test, feature_names = preprocess_data_for_pipeline()
    
    print(f"\n📊 Dados carregados:")
    print(f"   Treino: {X_train.shape[0]} amostras (com SMOTE, SEM escalonamento)")
    print(f"   Teste:  {X_test.shape[0]} amostras (sem SMOTE, SEM escalonamento)")
    print(f"   Features: {len(feature_names)}")
    print(f"   {feature_names}")
    
    # ✅ Criar pipeline com componentes padrão do scikit-learn
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    tuning_summary = None
    
    if use_tuning:
        print(f"\n🔍 Executando RandomizedSearchCV (n_iter={n_iter}, cv={cv})...")
        
        # Grid de hiperparâmetros
        param_distributions = {
            'classifier__n_estimators': [200, 300, 500],
            'classifier__max_depth': [None, 20, 30],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],  # Reduzido para aumentar variabilidade
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__bootstrap': [True],
            'classifier__class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Configurar busca
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=2,
            return_train_score=True
        )
        
        # Treinar
        start_time = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        # Obter melhor pipeline
        pipeline = search.best_estimator_
        
        # Resumo do tuning
        tuning_summary = {
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'cv_folds': cv,
            'n_iterations': n_iter,
            'elapsed_time_minutes': elapsed / 60,
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': search.cv_results_['mean_train_score'].tolist()
            }
        }
        
        print(f"\n✅ Tuning concluído em {elapsed/60:.2f} minutos")
        print(f"   Melhor ROC-AUC (CV): {search.best_score_:.4f}")
        print(f"\n🏆 Melhores hiperparâmetros:")
        for param, value in search.best_params_.items():
            print(f"      {param}: {value}")
    
    else:
        print("\n⚡ Treinando com hiperparâmetros padrão...")
        pipeline.fit(X_train, y_train)
        print("✅ Treinamento concluído!")
    
    # Avaliar no conjunto de teste
    print("\n📈 Avaliando no conjunto de teste...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n📊 Métricas de Teste (dados limpos, sem SMOTE):")
    print(f"   Acurácia:  {test_metrics['accuracy']:.4f}")
    print(f"   Precisão:  {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1-Score:  {test_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    # Salvar pipeline
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    pipeline_path = os.path.join(models_dir, 'random_forest_pipeline.joblib')
    
    joblib.dump(pipeline, pipeline_path)
    print(f"\n💾 Pipeline salvo em: {pipeline_path}")
    print(f"   Tamanho: {os.path.getsize(pipeline_path) / 1024:.2f} KB")
    
    # Salvar metadados
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'test_metrics': test_metrics,
        'tuning_summary': tuning_summary,
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'timestamp': datetime.now().isoformat(),
        'sklearn_version': joblib.__version__
    }
    
    metadata_path = os.path.join(models_dir, 'pipeline_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadados salvos em: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE DE PRODUÇÃO CRIADO COM SUCESSO!")
    print("=" * 70)
    print("\n📦 O pipeline contém:")
    print("   1. RobustScaler (normalização)")
    print("   2. RandomForestClassifier (modelo)")
    print("\n💡 Para usar:")
    print("   pipeline = joblib.load('models/random_forest_pipeline.joblib')")
    print("   probas = pipeline.predict_proba(X)")
    print("=" * 70)
    
    return pipeline, X_test, y_test, feature_names, tuning_summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--pipeline':
        # Criar pipeline de produção
        use_tuning = '--tune' in sys.argv
        
        if use_tuning:
            # Extrair n_iter e cv dos argumentos
            n_iter = 10
            cv = 3
            for arg in sys.argv:
                if arg.startswith('--n_iter='):
                    n_iter = int(arg.split('=')[1])
                elif arg.startswith('--cv='):
                    cv = int(arg.split('=')[1])
            
            print(f"🔍 Modo: Pipeline com tuning (n_iter={n_iter}, cv={cv})")
            pipeline, X_test, y_test, features, summary = train_production_pipeline(
                use_tuning=True, n_iter=n_iter, cv=cv
            )
        else:
            print("⚡ Modo: Pipeline sem tuning (hiperparâmetros padrão)")
            pipeline, X_test, y_test, features, _ = train_production_pipeline(
                use_tuning=False
            )
        
        print(f"\n🏁 Pipeline criado com {len(features)} features.")
    else:
        # Treino tradicional (backward compatibility)
        print("⚙️ Executando treino standalone (modo legado)...")
        model, X_test, y_test, features = train_random_forest()
        print(f"🏁 Modelo treinado com {len(features)} features.")
    
    print("✅ Avaliação realizada em dados limpos, sem vazamento de dados!")
