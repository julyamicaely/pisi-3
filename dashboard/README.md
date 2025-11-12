# 📊 Dashboard Colaborativo - Pipeline Cardiovascular

Dashboard multipage com Dash (Plotly) para visualização e análise de modelos de Machine Learning aplicados à classificação e clusterização de risco cardiovascular.

---

## 🎯 Estrutura do Projeto

```
dashboard/
├── app.py                      # Aplicação principal (multipage)
├── styles.py                   # Governança visual (paleta, fontes, templates Plotly)
├── components/
│   ├── __init__.py
│   ├── utils.py               # Helpers para layouts (cards, grids, headers)
│   └── cards.py               # Componentes de visualização (gráficos, tabelas)
├── pages/
│   ├── home.py                # Página inicial
│   ├── random_forest.py       # Análise Random Forest
│   ├── xgboost.py             # Análise XGBoost
│   └── clusterizacao.py       # Análise de clusters K-Means
├── assets/                     # CSS customizado (opcional)
└── README.md                   # Este arquivo
```

---

## 🚀 Como Executar

### 1. Instalar Dependências

```powershell
pip install dash dash-bootstrap-components plotly pandas scikit-learn joblib
```

### 2. Executar Dashboard

```powershell
cd dashboard
python app.py
```

### 3. Acessar

Abra o navegador em: **http://localhost:8050**

---

## 📋 Padrão de Artefatos

Cada modelo deve gerar artefatos seguindo a estrutura abaixo:

### Random Forest
```
classification/
├── models/
│   └── random_forest_model.joblib
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── scalers/
    └── robust_scaler.joblib
```

### XGBoost
```
classification/xgboost_local/
└── results/
    └── xgboost_report_YYYY-MM-DD_HH-MM-SS.txt
```

### Clusterização
```
clusterization/
├── cluster_2_summary.csv
├── cluster_report.txt
├── cluster_distribution.png
├── cluster_comparison.png
├── cluster_boxplots.png
├── cluster_pca.png
└── Cotovelo x Silhouete.png
```

---

## 🎨 Governança Visual

### Paleta de Cores

| Uso | Cor | Hex |
|-----|-----|-----|
| Primária | Verde escuro | `#2F4F4F` |
| Destaque | Turquesa | `#00A699` |
| Sucesso | Verde | `#66BB6A` |
| Alerta | Vermelho suave | `#E57373` |
| Info | Azul | `#42A5F5` |

### Tipografia

- **Família base**: Inter, Arial, sans-serif
- **Headings**: Inter (700 weight)
- **Código**: Fira Code, Courier New

### Espaçamentos

- `xs`: 4px
- `sm`: 8px
- `md`: 16px
- `lg`: 24px
- `xl`: 32px
- `xxl`: 48px

---

## 🧩 Como Adicionar Nova Página

### 1. Criar Arquivo

Crie `dashboard/pages/novo_modelo.py`:

```python
import dash
from dash import html
import dash_bootstrap_components as dbc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from components.utils import make_page_header, make_card

# Registrar página
dash.register_page(
    __name__,
    path="/novo-modelo",
    name="Novo Modelo",
    title="Novo Modelo - Dashboard",
    icon="star-fill",
)

def layout():
    return html.Div([
        make_page_header("Novo Modelo", "Descrição do modelo", icon="star-fill"),
        make_card("Conteúdo", html.P("Adicione suas análises aqui")),
    ])
```

### 2. Adicionar Artefatos

Salve artefatos seguindo o padrão:

```
modelo_dir/
├── results/
│   ├── metrics.json
│   └── classification_report.txt
└── figures/
    └── feature_importance.png
```

### 3. Recarregar Dashboard

A página aparecerá automaticamente no menu lateral!

---

## 📦 JSON Padrão de Métricas (Opcional)

Para integração futura, use este formato:

```json
{
  "model": "random_forest",
  "timestamp": "2025-11-12T14:30:00",
  "metrics": {
    "accuracy": 0.84,
    "precision": 0.84,
    "recall": 0.84,
    "f1_score": 0.84
  },
  "figures": {
    "feature_importance": "classification/reports/feature_importance.png",
    "confusion_matrix": "classification/reports/confusion_matrix.png"
  },
  "report_path": "classification/reports/classification_report.txt",
  "model_path": "classification/models/random_forest_model.joblib"
}
```

---

## 🛠️ Componentes Reutilizáveis

### Utils (`components/utils.py`)

```python
from components.utils import (
    make_card,              # Card padrão
    make_metric_card,       # Card de métrica com valor
    build_metric_grid,      # Grid de métricas
    make_page_header,       # Header de página
    make_tabs,              # Abas
    make_alert,             # Alert
    make_empty_state,       # Placeholder vazio
)
```

### Cards (`components/cards.py`)

```python
from components.cards import (
    build_confusion_matrix,     # Matriz de confusão
    build_feature_importance,   # Importância de features
    build_classification_report_card,  # Relatório formatado
    build_scatter_plot,         # Dispersão
    build_histogram,            # Histograma
    build_box_plot,             # Box plot
    build_pie_chart,            # Pizza
)
```

---

## 🎓 Boas Práticas

### ✅ Sempre Fazer

1. Usar helpers centralizados (`make_card`, `make_metric_card`, etc.)
2. Seguir paleta de cores definida em `styles.py`
3. Prefixar IDs com nome da página (`rf-graph-1`, `xgb-table-1`)
4. Usar `make_empty_state` quando não houver dados
5. Adicionar ícones Bootstrap (`icon="tree-fill"`)

### ❌ Evitar

1. Criar estilos inline customizados (use `styles.py`)
2. Cores hardcoded (use `PALETTE`)
3. Funções com I/O direto (mantenha UI e lógica separadas)
4. Duplicação de código de layout

---

## 📊 Callbacks e Interatividade

Para adicionar interatividade (filtros, dropdowns):

```python
from dash import callback, Input, Output

@callback(
    Output("rf-graph", "figure"),
    Input("rf-filter", "value")
)
def update_graph(filter_value):
    # Lógica de atualização
    return figure
```

**Importante**: Use IDs prefixados para evitar colisões entre páginas!

---

## 🔧 Troubleshooting

### Erro: "ModuleNotFoundError: dash_bootstrap_components"

```powershell
pip install dash-bootstrap-components
```

### Imagens não aparecem

Verifique se os caminhos estão corretos:
- Caminhos relativos devem ser a partir de `dashboard/`
- Use `/assets/../pasta/imagem.png` para acessar fora de assets

### Página não aparece no menu

Verifique se:
1. Arquivo está em `dashboard/pages/`
2. Tem `dash.register_page(__name__, ...)`
3. Tem função `layout()`

---

## 📈 Roadmap

- [ ] Filtros globais (idade, sexo, período)
- [ ] Comparação lado a lado de modelos
- [ ] Export de relatórios em PDF
- [ ] API REST para integração
- [ ] Cache de artefatos para performance

---

## 👥 Contribuindo

1. Crie nova página em `pages/`
2. Use componentes de `components/`
3. Siga estilos de `styles.py`
4. Teste localmente
5. Documente artefatos esperados

---

## 📝 Licença

Este projeto faz parte do pipeline de ML cardiovascular - Novembro 2025

---

**Desenvolvido com ❤️ usando Dash (Plotly)**
