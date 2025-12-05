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
    
    # Header com gradiente
    page_header = html.Div([
        html.Div([
            html.H1([
                html.I(className="bi bi-heart-pulse-fill me-3", style={"fontSize": "42px", "color": "white"}),
                "Dashboard de Machine Learning"
            ], className="mb-3", style={
                "fontSize": "42px", 
                "fontWeight": "700",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "color": "white"
            }),
            html.P(
                "Pipeline de classificação e análise de risco cardiovascular",
                className="text-white",
                style={"fontSize": "18px", "marginBottom": "0", "opacity": "0.95"}
            ),
        ], style={"position": "relative", "zIndex": "1"})
    ], style={
        "textAlign": "center",
        "padding": f"{SPACING['xxl']} {SPACING['lg']}",
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "borderRadius": "16px",
        "marginBottom": SPACING["xl"],
        "boxShadow": "0 12px 24px rgba(102, 126, 234, 0.3), 0 6px 12px rgba(118, 75, 162, 0.2)",
        "position": "relative",
        "overflow": "hidden",
    })
    
    # Cards de navegação rápida com glassmorphism
    nav_cards = dbc.Row([
        # Random Forest
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-tree-fill icon-pulse", style={
                                "fontSize": "56px", 
                                "background": "linear-gradient(135deg, #667eea, #764ba2)",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                                "backgroundClip": "text"
                            }),
                        ], style={"marginBottom": "16px"}),
                        html.Div([
                            html.H4("Random Forest", className="mb-2", style={"fontWeight": "700"}, id="rf-title"),
                            dbc.Tooltip(
                                "Ensemble de múltiplas árvores de decisão que vota para classificação final. "
                                "Robusto contra overfitting e funciona bem com dados desbalanceados.",
                                target="rf-title",
                                placement="top",
                            ),
                        ]),
                        html.P("Modelo de classificação com ensemble de árvores de decisão", 
                               className="text-muted mb-3", 
                               style={"fontSize": "14px", "lineHeight": "1.5"}),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-right me-2"), "Ver Análise"],
                            href="/random-forest",
                            style={
                                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                "border": "none",
                                "fontWeight": "600",
                                "padding": "10px 24px",
                                "borderRadius": "8px"
                            },
                            className="w-100"
                        ),
                    ], style={"textAlign": "center"})
                ])
            ], className="glass-card", style={
                "height": "100%", 
                "borderRadius": "16px",
                "border": "1px solid rgba(255,255,255,0.3)"
            })
        ], width=12, md=6, lg=4, className="mb-4"),
        
        # XGBoost
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-lightning-charge-fill icon-pulse", style={
                                "fontSize": "56px",
                                "background": "linear-gradient(135deg, #4facfe, #00f2fe)",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                                "backgroundClip": "text"
                            }),
                        ], style={"marginBottom": "16px"}),
                        html.Div([
                            html.H4("XGBoost", className="mb-2", style={"fontWeight": "700"}, id="xgb-title"),
                            dbc.Tooltip(
                                "Extreme Gradient Boosting: algoritmo de boosting sequencial que otimiza "
                                "erros residuais. Alta precisão e eficiência computacional.",
                                target="xgb-title",
                                placement="top",
                            ),
                        ]),
                        html.P("Gradient Boosting otimizado para alta performance", 
                               className="text-muted mb-3",
                               style={"fontSize": "14px", "lineHeight": "1.5"}),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-right me-2"), "Ver Análise"],
                            href="/xgboost",
                            style={
                                "background": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
                                "border": "none",
                                "fontWeight": "600",
                                "padding": "10px 24px",
                                "borderRadius": "8px"
                            },
                            className="w-100"
                        ),
                    ], style={"textAlign": "center"})
                ])
            ], className="glass-card", style={
                "height": "100%",
                "borderRadius": "16px",
                "border": "1px solid rgba(255,255,255,0.3)"
            })
        ], width=12, md=6, lg=4, className="mb-4"),
        
        # Naive Bayes
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-clipboard-data-fill icon-pulse", style={
                                "fontSize": "56px",
                                "background": "linear-gradient(135deg, #fa709a, #fee140)",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                                "backgroundClip": "text"
                            }),
                        ], style={"marginBottom": "16px"}),
                        html.Div([
                            html.H4("Naive Bayes", className="mb-2", style={"fontWeight": "700"}, id="nb-title"),
                            dbc.Tooltip(
                                "Classificador probabilístico baseado no Teorema de Bayes. "
                                "Rápido, eficiente e funciona bem com dados de alta dimensionalidade.",
                                target="nb-title",
                                placement="top",
                            ),
                        ]),
                        html.P("Classificador probabilístico baseado no Teorema de Bayes", 
                               className="text-muted mb-3",
                               style={"fontSize": "14px", "lineHeight": "1.5"}),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-right me-2"), "Ver Análise"],
                            href="/naive-bayes",
                            style={
                                "background": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
                                "border": "none",
                                "fontWeight": "600",
                                "padding": "10px 24px",
                                "borderRadius": "8px"
                            },
                            className="w-100"
                        ),
                    ], style={"textAlign": "center"})
                ])
            ], className="glass-card", style={
                "height": "100%",
                "borderRadius": "16px",
                "border": "1px solid rgba(255,255,255,0.3)"
            })
        ], width=12, md=6, lg=4, className="mb-4"),

        # Clusterização
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-diagram-3-fill icon-pulse", style={
                                "fontSize": "56px",
                                "background": "linear-gradient(135deg, #11998e, #38ef7d)",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                                "backgroundClip": "text"
                            }),
                        ], style={"marginBottom": "16px"}),
                        html.Div([
                            html.H4("Clusterização", className="mb-2", style={"fontWeight": "700"}, id="cluster-title"),
                            dbc.Tooltip(
                                "K-Means: algoritmo de aprendizado não supervisionado que agrupa pacientes "
                                "por similaridade. Descobre padrões ocultos nos dados.",
                                target="cluster-title",
                                placement="top",
                            ),
                        ]),
                        html.P("Segmentação de pacientes com K-Means não supervisionado", 
                               className="text-muted mb-3",
                               style={"fontSize": "14px", "lineHeight": "1.5"}),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-right me-2"), "Ver Análise"],
                            href="/clusterizacao",
                            style={
                                "background": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
                                "border": "none",
                                "fontWeight": "600",
                                "padding": "10px 24px",
                                "borderRadius": "8px"
                            },
                            className="w-100"
                        ),
                    ], style={"textAlign": "center"})
                ])
            ], className="glass-card", style={
                "height": "100%",
                "borderRadius": "16px",
                "border": "1px solid rgba(255,255,255,0.3)"
            })
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
    
    # Footer com atalhos de teclado
    footer = html.Div([
        html.Hr(style={"margin": f"{SPACING['xl']} 0"}),
        
        # Atalhos de teclado
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("⌨️ Atalhos de Teclado", className="mb-3", style={"fontWeight": "600"}),
                    html.Div([
                        dbc.Badge("Home", color="light", text_color="dark", className="me-2"),
                        html.Span("Voltar ao topo", className="text-muted small"),
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("End", color="light", text_color="dark", className="me-2"),
                        html.Span("Ir ao final", className="text-muted small"),
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("Ctrl + K", color="light", text_color="dark", className="me-2"),
                        html.Span("Buscar (em breve)", className="text-muted small"),
                    ], className="mb-2"),
                ], style={
                    "background": "rgba(102, 126, 234, 0.05)",
                    "padding": "20px",
                    "borderRadius": "12px",
                    "border": "1px solid rgba(102, 126, 234, 0.2)"
                })
            ], md=6, lg=4, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.H6("🎯 Navegação Rápida", className="mb-3", style={"fontWeight": "600"}),
                    html.Div([
                        html.A([
                            html.I(className="bi bi-tree me-2"),
                            "Random Forest"
                        ], href="/random-forest", className="d-block text-decoration-none mb-2"),
                        html.A([
                            html.I(className="bi bi-lightning-charge me-2"),
                            "XGBoost"
                        ], href="/xgboost", className="d-block text-decoration-none mb-2"),
                        html.A([
                            html.I(className="bi bi-clipboard-data me-2"),
                            "Naive Bayes"
                        ], href="/naive-bayes", className="d-block text-decoration-none mb-2"),
                    ])
                ], style={
                    "background": "rgba(67, 160, 71, 0.05)",
                    "padding": "20px",
                    "borderRadius": "12px",
                    "border": "1px solid rgba(67, 160, 71, 0.2)"
                })
            ], md=6, lg=4, className="mb-4"),
        ]),
        
        html.P([
            "Dashboard desenvolvido com ",
            html.I(className="bi bi-heart-fill", style={"color": PALETTE["warn"]}),
            " usando Dash (Plotly) | ",
            html.A("Documentação", href="#", className="text-decoration-none"),
            " | ",
            html.A("GitHub", href="#", className="text-decoration-none"),
        ], className="text-center text-muted small mt-4"),
    ])
    
    # Layout completo
    return html.Div([
        page_header,
        nav_cards,
        about_section,
        stats_section,
        footer,
    ])
