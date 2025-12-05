"""
Dashboard Interativo - XGBoost Classifier
Layout em scroll vertical com seções progressivas e filtros independentes
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Importar estilos
import sys
sys.path.append(str(Path(__file__).parent.parent))
from styles import PALETTE, SPACING, create_plotly_template
from components.cards import build_roc_curve, build_precision_recall_curve, build_calibration_curve

# Importar módulos do projeto
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from classification.xgboost_local.model_xgboost import train_xgboost
    from classification.evaluation import compute_validation_metrics
    USE_SHARED_FUNCTIONS = True
except ImportError:
    print("⚠️ Funções compartilhadas não disponíveis.")
    USE_SHARED_FUNCTIONS = False

# Registrar página
dash.register_page(
    __name__,
    path="/xgboost",
    name="XGBoost",
    title="XGBoost - Dashboard Interativo",
    icon="lightning-charge-fill"
)

# ================== CARREGAR DADOS ==================
_xgb_data_cache = None

def load_data():
    """Carrega modelo XGBoost e dataset (com cache)."""
    global _xgb_data_cache
    
    # Retornar cache se já carregado
    if _xgb_data_cache is not None:
        print("🔄 Usando cache do XGBoost (não retreinando)")
        return _xgb_data_cache
    
    print("🚀 Carregando dados XGBoost pela primeira vez...")
    
    data = {
        "model": None,
        "X_test": None,
        "X_test_original": None,
        "y_test": None,
        "y_pred": None,
        "y_proba": None,
        "feature_names": None,
        "metrics": {}
    }
    
    try:
        # Treinar/carregar modelo XGBoost
        model, X_test, y_test, feature_names, metrics = train_xgboost(use_smote=False)
        
        # Converter para DataFrame se necessário
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        data["model"] = model
        data["X_test"] = X_test
        data["X_test_original"] = X_test.copy()  # Para filtros
        data["y_test"] = y_test
        data["y_pred"] = model.predict(X_test)
        data["y_proba"] = model.predict_proba(X_test)
        data["feature_names"] = feature_names
        data["metrics"] = metrics
        
        print("✅ Dados XGBoost carregados com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados XGBoost: {e}")
        import traceback
        traceback.print_exc()
    
    _xgb_data_cache = data
    return data

def get_data():
    """Obtém dados (carrega se necessário)."""
    return load_data()

def get_data():
    """Obtém dados (carrega se necessário)."""
    return load_data()

# NÃO carregar dados na importação - apenas sob demanda
# xgb_data = load_data()  # ❌ REMOVIDO

# ================== TRADUÇÕES ==================
FEATURE_TRANSLATIONS = {
    'age_years': 'Idade (anos)',
    'ap_hi': 'Pressão Sistólica',
    'ap_lo': 'Pressão Diastólica',
    'cholesterol': 'Colesterol',
    'gluc': 'Glicose',
    'weight': 'Peso (kg)',
    'height': 'Altura (cm)',
    'bmi': 'IMC',
    'pulse_pressure': 'Pressão de Pulso',
    'map': 'Pressão Arterial Média',
    'smoke': 'Fumante',
    'alco': 'Álcool',
    'active': 'Atividade Física',
    'gender': 'Gênero'
}

# ================== LAYOUT ==================
def layout():
    """Layout principal do dashboard XGBoost."""
    
    data = get_data()
    
    if data["model"] is None:
        return html.Div([
            dbc.Alert(
                "❌ Erro ao carregar modelo XGBoost. Verifique os logs.",
                color="danger",
                className="m-4"
            )
        ])
    
    metrics = data["metrics"]
    
    # Hero Section SIMPLIFICADO - teste
    hero = html.Div([
        html.H1("XGBoost Classifier"),
        html.P("Gradient Boosting otimizado"),
        html.Div([
            html.Div([
                html.H2(f"{metrics.get('precision', 0):.1%}"),
                html.P("Precision"),
            ], style={"display": "inline-block", "margin": "10px"}),
            html.Div([
                html.H2(f"{metrics.get('recall', 0):.1%}"),
                html.P("Recall"),
            ], style={"display": "inline-block", "margin": "10px"}),
            html.Div([
                html.H2(f"{metrics.get('f1', 0):.1%}"),
                html.P("F1-Score"),
            ], style={"display": "inline-block", "margin": "10px"}),
            html.Div([
                html.H2(f"{metrics.get('roc_auc', 0):.1%}"),
                html.P("AUC-ROC"),
            ], style={"display": "inline-block", "margin": "10px"}),
        ])
    ], style={
        "background": "linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%)",
        "padding": "40px",
        "borderRadius": "16px",
        "marginBottom": "20px",
        "color": "white"
    })
    
    return html.Div([
        hero,
        html.H3("Tabs de Análise", style={"marginTop": "20px"}),
        dbc.Tabs([
            dbc.Tab(label="📊 Overview", tab_id="tab-overview"),
            dbc.Tab(label="📈 Performance", tab_id="tab-curves"),
            dbc.Tab(label="🎯 Features", tab_id="tab-features"),
            dbc.Tab(label="🔍 EDA", tab_id="tab-eda"),
        ], id="xgb-tabs", active_tab="tab-overview"),
        html.Div(id="xgb-tab-content", style={"marginTop": "20px"})
    ], style={"padding": "20px"})


# ================== CALLBACK: TAB CONTENT ==================
@callback(
    Output("xgb-tab-content", "children"),
    Input("xgb-tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Renderiza conteúdo da tab ativa."""
    
    # Carregar dados UMA ÚNICA VEZ aqui
    data = get_data()
    
    if active_tab == "tab-overview":
        return html.Div([
            html.H4("Matriz de Confusão"),
            html.Label("Filtrar por gênero:"),
            dcc.Dropdown(
                id="xgb-gender-filter",
                options=[
                    {"label": "Todos", "value": "all"},
                    {"label": "Masculino", "value": 1},
                    {"label": "Feminino", "value": 2}
                ],
                value="all",
                clearable=False,
                style={"marginBottom": "20px"}
            ),
            dcc.Graph(id="xgb-confusion-matrix")
        ])
    
    elif active_tab == "tab-curves":
        if data["y_test"] is None or data["y_proba"] is None:
            return dbc.Alert("⚠️ Dados não disponíveis", color="warning")
        
        y_test = data["y_test"]
        y_proba = data["y_proba"][:, 1]
        
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'AUC = {roc_auc:.3f}',
            line=dict(color=PALETTE["primary"], width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Baseline',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=False
        ))
        fig_roc.update_layout(
            title=f"Curva ROC (AUC: {roc_auc:.3f})",
            xaxis_title='Taxa de Falsos Positivos',
            yaxis_title='Taxa de Verdadeiros Positivos',
            height=400,
            template="plotly_white",
            showlegend=True
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name='Precision-Recall',
            line=dict(color=PALETTE["success"], width=3)
        ))
        fig_pr.update_layout(
            title="Curva Precision-Recall",
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400,
            template="plotly_white"
        )
        
        return html.Div([
            html.H5("Performance Curves", style={"marginBottom": "20px"}),
            html.Div([
                dcc.Graph(figure=fig_roc, config={'displayModeBar': False}, style={"display": "inline-block", "width": "49%"}),
                dcc.Graph(figure=fig_pr, config={'displayModeBar': False}, style={"display": "inline-block", "width": "49%", "marginLeft": "2%"})
            ])
        ], style={"padding": "20px"})
    
    elif active_tab == "tab-features":
        feature_count = len(data["feature_names"])
        return html.Div([
            html.H5("Importância das Features"),
            html.Label("Número de features:"),
            dcc.Slider(
                id="xgb-n-features-slider",
                min=5,
                max=min(feature_count, 15),  # Limitar a 15 para evitar gráficos gigantes
                value=10,
                marks={i: str(i) for i in range(5, min(feature_count, 16), 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            dcc.Graph(id="xgb-feature-importance", style={"marginTop": "20px"})
        ])
    
    elif active_tab == "tab-eda":
        feature_options = [
            {"label": FEATURE_TRANSLATIONS.get(f, f), "value": f}
            for f in data["feature_names"]
        ]
        
        return html.Div([
            html.H5("Scatter Plot"),
            dbc.Row([
                dbc.Col([
                    html.Label("Eixo X:"),
                    dcc.Dropdown(
                        id="xgb-scatter-x",
                        options=feature_options,
                        value="age_years",
                        clearable=False
                    )
                ], md=6),
                dbc.Col([
                    html.Label("Eixo Y:"),
                    dcc.Dropdown(
                        id="xgb-scatter-y",
                        options=feature_options,
                        value="ap_hi",
                        clearable=False
                    )
                ], md=6)
            ]),
            dcc.Graph(id="xgb-scatter-plot", style={"marginTop": "20px"}),
            
            html.H5("Distribuição de Feature", style={"marginTop": "40px"}),
            html.Label("Selecionar feature:"),
            dcc.Dropdown(
                id="xgb-dist-feature",
                options=feature_options,
                value="age_years",
                clearable=False
            ),
            dcc.Graph(id="xgb-distribution", style={"marginTop": "20px"})
        ])
    
    return html.P("Tab não encontrada")


# ================== CALLBACKS DOS GRÁFICOS ==================

# Confusion Matrix
@callback(
    Output("xgb-confusion-matrix", "figure"),
    Input("xgb-gender-filter", "value"),
    prevent_initial_call=False
)
def update_confusion_matrix(gender_filter):
    """Atualiza matriz de confusão com filtro de gênero."""
    
    data = get_data()
    X_test = data["X_test_original"]
    y_test = data["y_test"]
    y_pred = data["y_pred"]
    
    # Aplicar filtro
    if gender_filter != "all" and "gender" in X_test.columns:
        mask = X_test["gender"] == int(gender_filter)
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
    else:
        y_test_filtered = y_test
        y_pred_filtered = y_pred
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_test_filtered, y_pred_filtered)
    
    # Criar heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Sem Doença", "Com Doença"],
        y=["Sem Doença", "Com Doença"],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale="Blues",
        showscale=False
    ))
    
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predito",
        yaxis_title="Real",
        height=400,
        template="plotly_white"
    )
    
    return fig


# Feature Importance
@callback(
    Output("xgb-feature-importance", "figure"),
    Input("xgb-n-features-slider", "value"),
    prevent_initial_call=False
)
def update_feature_importance(n_features):
    """Atualiza gráfico de importância de features."""
    
    data = get_data()
    model = data["model"]
    feature_names = data["feature_names"]
    
    # Obter importâncias
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    # Traduzir nomes
    translated_names = [FEATURE_TRANSLATIONS.get(feature_names[i], feature_names[i]) for i in indices]
    
    # Criar gráfico
    fig = go.Figure([
        go.Bar(
            x=importances[indices],
            y=translated_names,
            orientation='h',
            marker_color=PALETTE["accent"]
        )
    ])
    
    fig.update_layout(
        title=f"Top {n_features} Features Mais Importantes",
        xaxis_title="Importância",
        yaxis_title="Feature",
        height=max(400, n_features * 35),  # Altura dinâmica mas limitada
        template="plotly_white",
        yaxis=dict(autorange="reversed")
    )
    
    return fig


# Scatter Plot
@callback(
    Output("xgb-scatter-plot", "figure"),
    [Input("xgb-scatter-x", "value"),
     Input("xgb-scatter-y", "value")],
    prevent_initial_call=False
)
def update_scatter(x_feature, y_feature):
    """Atualiza scatter plot."""
    
    data = get_data()
    X_test = data["X_test_original"]
    y_test = data["y_test"]
    
    df_plot = pd.DataFrame({
        'x': X_test[x_feature],
        'y': X_test[y_feature],
        'Diagnóstico': ['Com Doença' if y == 1 else 'Sem Doença' for y in y_test]
    })
    
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='Diagnóstico',
        labels={
            'x': FEATURE_TRANSLATIONS.get(x_feature, x_feature),
            'y': FEATURE_TRANSLATIONS.get(y_feature, y_feature)
        },
        color_discrete_map={
            'Sem Doença': PALETTE["success"],
            'Com Doença': PALETTE["warn"]
        },
        opacity=0.6
    )
    
    fig.update_layout(
        height=450,
        template="plotly_white"
    )
    
    return fig


# Distribution
@callback(
    Output("xgb-distribution", "figure"),
    Input("xgb-dist-feature", "value"),
    prevent_initial_call=False
)
def update_distribution(feature):
    """Atualiza histograma de distribuição."""
    
    data = get_data()
    X_test = data["X_test_original"]
    y_test = data["y_test"]
    
    df_plot = pd.DataFrame({
        'feature': X_test[feature],
        'Diagnóstico': ['Com Doença' if y == 1 else 'Sem Doença' for y in y_test]
    })
    
    fig = px.histogram(
        df_plot,
        x='feature',
        color='Diagnóstico',
        barmode='overlay',
        labels={'feature': FEATURE_TRANSLATIONS.get(feature, feature)},
        color_discrete_map={
            'Sem Doença': PALETTE["success"],
            'Com Doença': PALETTE["warn"]
        },
        opacity=0.7
    )
    
    fig.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title=FEATURE_TRANSLATIONS.get(feature, feature),
        yaxis_title="Frequência"
    )
    
    return fig
# Descomentar após identificar o problema

# # ================== CALLBACK: TAB CONTENT ==================
