"""
Página inicial do dashboard.
Visão geral do projeto e links rápidos para diferentes análises.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
from pathlib import Path

# Adicionar paths para imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.utils import make_page_header, make_card
from styles import PALETTE, SPACING

# Registrar como página inicial
dash.register_page(
    __name__,
    path="/",
    name="Home",
    title="Dashboard - Home",
    icon="house-fill",
)


# ================== LAYOUT DA PÁGINA ==================
def layout():
    """Cria layout da página inicial."""
    
    # Header
    page_header = html.Div([
        html.H1("🏥 Dashboard de Machine Learning", style={"fontSize": "36px", "fontWeight": "700", "color": PALETTE["dark"]}),
        html.P(
            "Pipeline de classificação e análise de risco cardiovascular",
            style={"fontSize": "18px", "color": PALETTE["muted"], "marginBottom": "0"}
        ),
    ], style={
        "textAlign": "center",
        "padding": f"{SPACING['xxl']} {SPACING['lg']}",
        "backgroundColor": "white",
        "borderRadius": "12px",
        "marginBottom": SPACING["xl"],
        "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
    })
    
    # Cards de navegação rápida
    nav_cards = dbc.Row([
        # Random Forest
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-tree-fill", style={"fontSize": "48px", "color": PALETTE["primary"]}),
                        html.H4("Random Forest", className="mt-3 mb-2"),
                        html.P("Modelo de classificação com ensemble de árvores de decisão", className="text-muted mb-3"),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-right me-2"), "Ver Análise"],
                            href="/random-forest",
                            color="primary",
                            className="w-100"
                        ),
                    ], style={"textAlign": "center"})
                ])
            ], style={"height": "100%", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "transition": "transform 0.2s"})
        ], width=12, md=6, lg=4, className="mb-4"),
        
        # XGBoost
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-lightning-charge-fill", style={"fontSize": "48px", "color": PALETTE["accent"]}),
                        html.H4("XGBoost", className="mt-3 mb-2"),
                        html.P("Gradient Boosting otimizado para alta performance", className="text-muted mb-3"),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-right me-2"), "Ver Análise"],
                            href="/xgboost",
                            color="info",
                            className="w-100"
                        ),
                    ], style={"textAlign": "center"})
                ])
            ], style={"height": "100%", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
        ], width=12, md=6, lg=4, className="mb-4"),
        
        # Naive Bayes
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-clipboard-data-fill", style={"fontSize": "48px", "color": PALETTE["info"]}),
                        html.H4("Naive Bayes", className="mt-3 mb-2"),
                        html.P("Classificador probabilístico baseado no Teorema de Bayes", className="text-muted mb-3"),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-right me-2"), "Ver Análise"],
                            href="/naive-bayes",
                            color="secondary",
                            className="w-100"
                        ),
                    ], style={"textAlign": "center"})
                ])
            ], style={"height": "100%", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
        ], width=12, md=6, lg=4, className="mb-4"),

        # Clusterização
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="bi bi-diagram-3-fill", style={"fontSize": "48px", "color": PALETTE["success"]}),
                        html.H4("Clusterização", className="mt-3 mb-2"),
                        html.P("Segmentação de pacientes com K-Means não supervisionado", className="text-muted mb-3"),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-right me-2"), "Ver Análise"],
                            href="/clusterizacao",
                            color="success",
                            className="w-100"
                        ),
                    ], style={"textAlign": "center"})
                ])
            ], style={"height": "100%", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
        ], width=12, md=6, lg=4, className="mb-4"),
    ])
    
    # Sobre o projeto
    about_section = dbc.Row([
        dbc.Col([
            make_card(
                "Sobre o Projeto",
                html.Div([
                    html.P("Este dashboard apresenta os resultados de diferentes modelos de Machine Learning aplicados à classificação de risco cardiovascular."),
                    html.H6("Modelos Disponíveis:", className="mt-3 mb-2"),
                    html.Ul([
                        html.Li([html.Strong("Random Forest: "), "Classificação robusta com ensemble de árvores"]),
                        html.Li([html.Strong("XGBoost: "), "Gradient Boosting otimizado para máxima precisão"]),
                        html.Li([html.Strong("Naive Bayes: "), "Classificador probabilístico para dados mistos"]),
                        html.Li([html.Strong("K-Means: "), "Clusterização para descobrir padrões ocultos"]),
                    ]),
                    html.H6("Características:", className="mt-3 mb-2"),
                    html.Ul([
                        html.Li("✅ Pipeline sem vazamento de dados (data leakage)"),
                        html.Li("✅ Pré-processamento com RobustScaler e SMOTE"),
                        html.Li("✅ Avaliação em dados de teste limpos"),
                        html.Li("✅ Visualizações interativas e relatórios detalhados"),
                    ]),
                ]),
                icon="info-circle"
            ),
        ], width=12, lg=6),
        
        dbc.Col([
            make_card(
                "Como Usar",
                html.Div([
                    html.H6("1. Executar Pipelines:", className="mb-2"),
                    html.Ul([
                        html.Li([html.Code("python classification/app.py", style={"fontSize": "12px"}), " - Random Forest"]),
                        html.Li([html.Code("python classification/xgboost_local/app_xgboost.py", style={"fontSize": "12px"}), " - XGBoost"]),
                        html.Li([html.Code("python classification/naive_bayes/train_naive_bayes.py", style={"fontSize": "12px"}), " - Naive Bayes"]),
                        html.Li([html.Code("python clusterization/n2_clusters.py", style={"fontSize": "12px"}), " - Clusterização"]),
                    ], style={"fontSize": "14px"}),
                    
                    html.H6("2. Visualizar Resultados:", className="mb-2 mt-3"),
                    html.P("Navegue pelas páginas usando o menu lateral ou os cards acima.", style={"fontSize": "14px"}),
                    
                    html.H6("3. Exportar e Compartilhar:", className="mb-2 mt-3"),
                    html.P("Todos os artefatos (modelos, relatórios, gráficos) são salvos automaticamente.", style={"fontSize": "14px"}),
                ]),
                icon="play-circle"
            ),
        ], width=12, lg=6),
    ], className="mb-4")
    
    # Estatísticas do projeto
    stats_section = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.I(className="bi bi-file-earmark-code", style={"fontSize": "32px", "color": PALETTE["primary"]}),
                    html.H3("4", className="mt-2 mb-0"),
                    html.P("Modelos de ML", className="text-muted mb-0 small"),
                ], style={"textAlign": "center"})
            ])
        ], width=6, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.I(className="bi bi-graph-up", style={"fontSize": "32px", "color": PALETTE["accent"]}),
                    html.H3("15+", className="mt-2 mb-0"),
                    html.P("Features", className="text-muted mb-0 small"),
                ], style={"textAlign": "center"})
            ])
        ], width=6, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.I(className="bi bi-people", style={"fontSize": "32px", "color": PALETTE["success"]}),
                    html.H3("70k+", className="mt-2 mb-0"),
                    html.P("Pacientes", className="text-muted mb-0 small"),
                ], style={"textAlign": "center"})
            ])
        ], width=6, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.I(className="bi bi-shield-check", style={"fontSize": "32px", "color": PALETTE["info"]}),
                    html.H3("Sim", className="mt-2 mb-0"),
                    html.P("Sem Vazamento", className="text-muted mb-0 small"),
                ], style={"textAlign": "center"})
            ])
        ], width=6, md=3),
    ])
    
    # Footer
    footer = html.Div([
        html.Hr(style={"margin": f"{SPACING['xl']} 0"}),
        html.P([
            "Dashboard desenvolvido com ",
            html.I(className="bi bi-heart-fill", style={"color": PALETTE["warn"]}),
            " usando Dash (Plotly) | ",
            html.A("Documentação", href="#", className="text-decoration-none"),
            " | ",
            html.A("GitHub", href="#", className="text-decoration-none"),
        ], className="text-center text-muted small"),
    ])
    
    # Layout completo
    return html.Div([
        page_header,
        nav_cards,
        about_section,
        stats_section,
        footer,
    ])
