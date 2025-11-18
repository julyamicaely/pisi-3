# Classification Module - Random Forest Production Pipeline

Este módulo contém o pipeline completo de Machine Learning para classificação de risco cardiovascular usando Random Forest, incluindo:

- ✅ Pré-processamento robusto (sem data leakage)
- ✅ Hyperparameter tuning com RandomizedSearchCV
- ✅ Validação cruzada estratificada (StratifiedKFold)
- ✅ Interpretabilidade com SHAP
- ✅ Persistência de modelos e artefatos

---

## 📁 Estrutura de Arquivos

```
classification/
├── preprocess_data.py           # Pipeline de pré-processamento (reutilizável)
├── model_random_forest.py       # Treinamento e tuning do Random Forest
├── shap_random_forest.py        # Geração de explicações SHAP
├── run_tuning.py                # Runner para executar tuning
├── run_shap.py                  # Runner para gerar relatórios SHAP
├── models/
│   └── random_forest_model.joblib   # Modelo otimizado persistido
├── reports/
│   ├── random_forest_tuning_results.json    # Resultados detalhados do tuning
│   ├── random_forest_tuning_summary.md      # Resumo legível do tuning
│   ├── confusion_matrix.png                 # Matriz de confusão
│   └── shap/                                # Visualizações SHAP
│       ├── shap_summary.png                 # Summary plot (importância global)
│       ├── shap_bar.png                     # Bar plot (importância agregada)
│       ├── shap_beeswarm.png                # Beeswarm plot
│       └── shap_decision_0.png              # Decision plot (exemplo local)
└── scalers/
    └── robust_scaler.joblib     # Scaler persistido
```

---

## 🚀 Como Executar

### Pré-requisitos

Instale as dependências (na raiz do projeto):

```bash
pip install -r requirements.txt
```

### 1. Treinar Modelo (carrega otimizado se existir)

```bash
# Executa train_random_forest() - carrega modelo salvo ou faz tuning
python -c "from classification.model_random_forest import train_random_forest; train_random_forest()"
```

### 2. Executar Hyperparameter Tuning

⚠️ **Atenção**: O tuning pode demorar (depende de `n_iter` e `cv_folds`).

**Opção A: Tuning rápido para teste (n_iter=8, cv=3)**

```bash
python -c "from classification.preprocess_data import preprocess_data; from classification.model_random_forest import run_rf_tuning; import json; X_train,_,y_train,_,_,_,feat = preprocess_data(); best, info = run_rf_tuning(X_train, y_train, feat, random_state=42, n_iter=8, cv_folds=3); json.dump(info, open('classification/reports/random_forest_tuning_results.json', 'w'), indent=2); open('classification/reports/random_forest_tuning_summary.md', 'w').write(info['summary_md']); print('✅ Tuning completo!')"
```

**Opção B: Tuning completo (n_iter=40, cv=5) - usar o runner**

```bash
python classification/run_tuning.py
```

**Outputs:**
- `classification/reports/random_forest_tuning_results.json` - resultados detalhados
- `classification/reports/random_forest_tuning_summary.md` - resumo markdown
- `classification/models/random_forest_model.joblib` - melhor modelo persistido

### 3. Gerar Relatórios SHAP (Interpretabilidade)

```bash
python classification/run_shap.py
```

**Outputs:**
- `classification/reports/shap/shap_summary.png` - Importância global das features
- `classification/reports/shap/shap_bar.png` - Importância média (bar plot)
- `classification/reports/shap/shap_beeswarm.png` - Distribuição de impactos
- `classification/reports/shap/shap_decision_0.png` - Explicação de exemplo específico

---

## 📊 Resultados Atuais (Último Tuning)

**Best ROC-AUC (CV)**: 0.7936  
**Best Parameters**:
```json
{
  "n_estimators": 100,
  "max_depth": 20,
  "min_samples_split": 15,
  "min_samples_leaf": 2,
  "max_features": "log2",
  "bootstrap": true
}
```

**Métricas de Validação Cruzada** (mean ± std):
- Accuracy: 0.7234 ± 0.0041
- Precision: 0.7536 ± 0.0069
- Recall: 0.6638 ± 0.0028
- F1-Score: 0.7058 ± 0.0033
- ROC-AUC: 0.7936 ± 0.0053

---

## 🔧 Personalização de Parâmetros

### Ajustar Tuning

Para modificar o espaço de busca ou número de iterações:

```python
from classification.model_random_forest import run_rf_tuning
from classification.preprocess_data import preprocess_data

X_train, _, y_train, _, _, _, features = preprocess_data()

# Customizar n_iter e cv_folds
best_model, results = run_rf_tuning(
    X_train, 
    y_train, 
    features, 
    random_state=42, 
    n_iter=20,      # ← ajustar número de combinações testadas
    cv_folds=5      # ← ajustar número de folds
)
```

### Ajustar SHAP

Para gerar explicações para outro índice de amostra:

```python
from classification.shap_random_forest import generate_shap_reports

# Gerar explicação local para amostra índice 42
generate_shap_reports(sample_index=42)
```

---

## 🧪 Pipeline de Pré-processamento

O módulo `preprocess_data.py` implementa:

1. ✅ Limpeza de dados inconsistentes (pressão arterial, outliers)
2. ✅ Feature engineering (BMI, age_years, categorical encoding)
3. ✅ Remoção de outliers (IQR method)
4. ✅ Split treino/teste (70/30) com estratificação
5. ✅ Escalonamento (RobustScaler) **apenas em X_train**
6. ✅ Balanceamento (SMOTE) **apenas em X_train**

**Features finais (10)**:
1. `gender` - Gênero (0=feminino, 1=masculino)
2. `ap_hi` - Pressão arterial sistólica
3. `ap_lo` - Pressão arterial diastólica
4. `smoke` - Fumante (0/1)
5. `alco` - Consome álcool (0/1)
6. `active` - Atividade física (0/1)
7. `age_years` - Idade em anos
8. `bmi` - Índice de massa corporal
9. `cholesterol_high` - Colesterol alto (0/1)
10. `gluc_high` - Glicose alta (0/1)

---

## 📈 Integração com Dashboard

O dashboard em `dashboard/` utiliza as funções deste módulo:

```python
from classification.preprocess_data import load_and_preprocess_data
from classification.model_random_forest import train_random_forest

# Carregar dados pré-processados
X_scaled, X_original, y, features = load_and_preprocess_data()

# Carregar modelo treinado
model, X_test, y_test, features = train_random_forest()
```

---

## 🛠️ Troubleshooting

### Erro: ModuleNotFoundError

Execute os scripts **da raiz do projeto** (`pisi-3-3/`):

```bash
cd c:\projetos\pisi-3-3
python classification/run_tuning.py
```

### Tuning muito lento

Reduza `n_iter` e `cv_folds` para testes rápidos:

```python
run_rf_tuning(X_train, y_train, features, n_iter=5, cv_folds=3)
```

### SHAP consome muita memória

O código já usa `shap.sample(X, 100)` para background. Se necessário, calcule SHAP values apenas para um subset:

```python
# Em shap_random_forest.py, modifique:
X_sample = X.sample(n=1000, random_state=42)
shap_values = explainer.shap_values(X_sample)
```

---

## 📝 Manutenção

### Retreinar modelo do zero

1. Delete o modelo existente:
```bash
Remove-Item classification\models\random_forest_model.joblib
```

2. Execute o tuning:
```bash
python classification/run_tuning.py
```

### Atualizar interpretabilidade

Regenere os plots SHAP após treinar novo modelo:

```bash
python classification/run_shap.py
```

---

## 🔗 Referências

- **Scikit-learn**: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- **SHAP**: https://shap.readthedocs.io/en/latest/
- **SMOTE**: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

---

## ✅ Checklist de Produção

- [x] Pipeline de pré-processamento sem data leakage
- [x] Hyperparameter tuning sistemático
- [x] Validação cruzada estratificada
- [x] Persistência de modelos e artefatos
- [x] Interpretabilidade com SHAP
- [x] Documentação completa
- [x] Scripts reutilizáveis e idempotentes
- [x] Outputs organizados em `reports/`

**Modelo pronto para produção!** 🚀