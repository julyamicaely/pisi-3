"""
Dashboard - Comparação de Modelos
Comparativo de performance entre os modelos treinados.
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# Importar estilos
sys.path.append(str(Path(__file__).parent.parent))
from styles import PALETTE, SHADOWS, GRADIENTS, CARD_STYLE, SIZES

# Registrar página
dash.register_page(
    __name__,
    path="/model-comparison",
    name="Comparação de Modelos",
    icon="bar-chart-steps"
)

# ====== Dados dos Modelos (Treino/Validação) ======
models_data = [
    {
        "Model": "Random Forest",
        "Accuracy": 0.73,
        "Precision": 0.72,
        "Recall": 0.73,
        "F1-Score": 0.72,
        "AUC-ROC": 0.79,
        "Color": PALETTE["primary"]
    },
    {
        "Model": "Naive Bayes",
        "Accuracy": 0.68,
        "Precision": 0.68,
        "Recall": 0.68,
        "F1-Score": 0.67,
        "AUC-ROC": 0.74, # Estimado
        "Color": PALETTE["accent"]
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.73,
        "Precision": 0.75,
        "Recall": 0.69,
        "F1-Score": 0.72,
        "AUC-ROC": 0.79,
        "Color": PALETTE["secondary"]
    }
]

df_models = pd.DataFrame(models_data)

# ====== Dados dos Modelos (Teste - Novos) ======
test_models_data = [
    {
        "Model": "Random Forest",
        "Accuracy": 0.735,
        "Precision": 0.699,
        "Recall": 0.653,
        "F1-Score": 0.675,
        "AUC-ROC": 0.806
    },
    {
        "Model": "Naive Bayes",
        "Accuracy": 0.722,
        "Precision": 0.726,
        "Recall": 0.547,
        "F1-Score": 0.624,
        "AUC-ROC": 0.764
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.731,
        "Precision": 0.738,
        "Recall": 0.559,
        "F1-Score": 0.636,
        "AUC-ROC": 0.778
    }
]
df_test_models = pd.DataFrame(test_models_data)

# Transformar para formato longo para facilitar plotagem agrupada
df_long = df_models.melt(id_vars=["Model", "Color"], 
                         value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
                         var_name="Metric", value_name="Score")

# Matrizes de Confusão (Normalizadas em %)
cm_data = {
    "Random Forest": [[0.46, 0.12], [0.16, 0.26]],
    "Naive Bayes":   [[0.40, 0.11], [0.21, 0.27]],
    "XGBoost":       [[0.47, 0.10], [0.17, 0.26]]
}

# ====== Componentes Auxiliares ======
def make_kpi_card(title, model_name, value, color, icon="trophy-fill"):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.H6(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                    html.H4(model_name, className=f"text-{color} mb-0", style={"fontWeight": "700"}),
                    html.Small(value, className="text-muted")
                ]),
                html.Div([
                    html.I(className=f"bi bi-{icon}", 
                           style={"fontSize": "2rem", "color": PALETTE.get(color, "#000"), "opacity": "0.2"})
                ])
            ], className="d-flex justify-content-between align-items-center")
        ])
    ], style=CARD_STYLE, className="h-100 border-0 shadow-sm")

# ====== Layout da Página ======
layout = html.Div([
    
    # Cabeçalho
    html.Div([
        html.H1("Comparação de Modelos", style=SIZES["h1"], className="text-primary mb-2"),
        html.P("Análise comparativa de performance entre os algoritmos de classificação.", 
               className="text-muted"),
    ], className="mb-4"),

    # KPIs de Destaque
    dbc.Row([
        dbc.Col(make_kpi_card("Melhor Acurácia", "XGBoost / RF", "~73%", "success"), width=12, md=4),
        dbc.Col(make_kpi_card("Melhor Precisão", "XGBoost", "75%", "info", "bullseye"), width=12, md=4),
        dbc.Col(make_kpi_card("Melhor Recall", "Random Forest", "73%", "warning", "arrow-repeat"), width=12, md=4),
    ], className="mb-4"),

    # Seção de Gráficos (Sem Tabs para garantir visibilidade)
    html.Div([
        html.H4("Visão Geral de Performance", className="mb-3 text-primary"),
        dbc.Row([
            # Gráfico Radar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Radar de Performance", className="bg-white border-0 fw-bold"),
                    dbc.CardBody([
                        dcc.Graph(id="radar-chart", config={"displayModeBar": False}, style={"height": "400px"})
                    ])
                ], style=CARD_STYLE, className="shadow-sm border-0 h-100")
            ], width=12, lg=6),
            
            # Gráfico de Barras
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Comparativo por Métrica", className="bg-white border-0 fw-bold"),
                    dbc.CardBody([
                        dcc.Graph(id="bar-chart", config={"displayModeBar": False}, style={"height": "400px"})
                    ])
                ], style=CARD_STYLE, className="shadow-sm border-0 h-100")
            ], width=12, lg=6),
        ], className="g-4 mb-5"),

        html.H4("Matrizes de Confusão (Normalizadas)", className="mb-3 text-primary"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Random Forest", className="bg-white border-0 fw-bold text-center"),
                    dbc.CardBody(dcc.Graph(id="cm-rf", config={"displayModeBar": False}, style={"height": "250px"}))
                ], style=CARD_STYLE, className="shadow-sm border-0")
            ], width=12, md=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("XGBoost", className="bg-white border-0 fw-bold text-center"),
                    dbc.CardBody(dcc.Graph(id="cm-xgb", config={"displayModeBar": False}, style={"height": "250px"}))
                ], style=CARD_STYLE, className="shadow-sm border-0")
            ], width=12, md=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Naive Bayes", className="bg-white border-0 fw-bold text-center"),
                    dbc.CardBody(dcc.Graph(id="cm-nb", config={"displayModeBar": False}, style={"height": "250px"}))
                ], style=CARD_STYLE, className="shadow-sm border-0")
            ], width=12, md=4),
        ], className="g-4 mb-2"),
        html.P("Valores normalizados (%). Eixo Y: Real, Eixo X: Predito.", className="text-center text-muted small mb-5"),

        html.H4("Dados Detalhados", className="mb-3 text-primary"),
        
        # Tabela de Teste (Nova)
        dbc.Card([
            dbc.CardHeader("Performance no Teste (Dados Hold-out)", className="bg-white border-0 fw-bold text-success"),
            dbc.CardBody([
                dbc.Table.from_dataframe(df_test_models, striped=True, bordered=True, hover=True, responsive=True, className="mb-0")
            ])
        ], style=CARD_STYLE, className="shadow-sm border-0 mb-4"),

        # Tabela de Treino (Antiga)
        dbc.Card([
            dbc.CardHeader("Performance no Treino/Validação", className="bg-white border-0 fw-bold text-muted"),
            dbc.CardBody([
                dbc.Table.from_dataframe(df_models.drop(columns=["Color"]), striped=True, bordered=True, hover=True, responsive=True, className="mb-0")
            ])
        ], style=CARD_STYLE, className="shadow-sm border-0 mb-4")
    ]),

], className="p-4")

# ====== Callbacks ======
@callback(
    [Output("radar-chart", "figure"),
     Output("bar-chart", "figure"),
     Output("cm-rf", "figure"),
     Output("cm-xgb", "figure"),
     Output("cm-nb", "figure")],
    Input("radar-chart", "id") # Dummy input
)
def update_charts(_):
    # 1. Radar Chart
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    fig_radar = go.Figure()

    for model in models_data:
        fig_radar.add_trace(go.Scatterpolar(
            r=[model[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=model["Model"],
            line_color=model["Color"],
            opacity=0.6
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.5, 0.85], tickfont=dict(size=10), tickangle=0),
            angularaxis=dict(tickfont=dict(size=12, family="Inter, sans-serif", color="#333"), rotation=90)
        ),
        margin=dict(t=40, b=40, l=60, r=60),
        legend=dict(orientation="h", y=-0.15, font=dict(size=12)),
        template="plotly_white"
    )

    # 2. Bar Chart
    fig_bar = px.bar(df_long, x="Metric", y="Score", color="Model", barmode="group",
                     color_discrete_map={m["Model"]: m["Color"] for m in models_data},
                     text_auto=".2f")
    
    fig_bar.update_traces(textangle=0, textposition="outside", cliponaxis=False)

    fig_bar.update_layout(
        plot_bgcolor="white",
        margin=dict(t=30, b=50, l=20, r=20),
        legend=dict(orientation="h", y=1.1, font=dict(size=12)),
        yaxis=dict(range=[0.5, 0.85], gridcolor="#f0f0f0"),
        xaxis=dict(title=None, tickfont=dict(size=12)),
        font=dict(family="Inter, sans-serif")
    )

    # 3. Confusion Matrices
    def make_cm_heatmap(matrix, color_scale):
        z = matrix
        x = ['Negativo', 'Positivo']
        y = ['Negativo', 'Positivo']
        
        # Annotations
        annotations = []
        for i in range(2):
            for j in range(2):
                annotations.append(dict(
                    x=x[j], y=y[i],
                    text=f"{z[i][j]:.0%}",
                    font=dict(color="white" if z[i][j] > 0.3 else "black"),
                    showarrow=False
                ))

        fig = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            colorscale=color_scale,
            showscale=False
        ))
        
        fig.update_layout(
            annotations=annotations,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Predito",
            yaxis_title="Real",
            yaxis=dict(autorange="reversed") # Para bater com convenção visual de matriz
        )
        return fig

    fig_rf = make_cm_heatmap(cm_data["Random Forest"], "Blues")
    fig_xgb = make_cm_heatmap(cm_data["XGBoost"], "Greens")
    fig_nb = make_cm_heatmap(cm_data["Naive Bayes"], "Reds")

    return fig_radar, fig_bar, fig_rf, fig_xgb, fig_nb
