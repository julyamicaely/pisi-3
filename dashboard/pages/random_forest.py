"""
Dashboard Interativo - Random Forest Classifier
Layout em scroll vertical com seções progressivas e filtros independentes
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Importar estilos
import sys
sys.path.append(str(Path(__file__).parent.parent))
from styles import PALETTE, create_plotly_template
from components.cards import build_roc_curve, build_precision_recall_curve, build_calibration_curve

# Importar módulos do projeto
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from classification.preprocess_data import load_and_preprocess_data
    from classification.evaluation import compute_validation_metrics
    USE_SHARED_FUNCTIONS = True
except ImportError:
    print("⚠️ Funções compartilhadas não disponíveis. Usando modo legacy.")
    USE_SHARED_FUNCTIONS = False

# Registrar página
dash.register_page(
    __name__,
    path="/random-forest",
    name="Random Forest",
    title="Random Forest - Dashboard Interativo",
    icon="tree-fill"
)

# ================== CARREGAR DADOS ==================
def load_data():
    """
    Carrega modelo e dataset. 
    Usa função compartilhada quando disponível para evitar duplicação.
    """
    base_path = Path(__file__).parent.parent.parent
    
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
        # Carregar modelo
        model_path = base_path / "classification" / "models" / "random_forest_model.joblib"
        if model_path.exists():
            data["model"] = joblib.load(model_path)
        
        # USAR FUNÇÃO COMPARTILHADA se disponível (elimina duplicação!)
        if USE_SHARED_FUNCTIONS and data["model"] is not None:
            try:
                X_scaled, X_original, y, feature_names = load_and_preprocess_data()
                
                data["X_test"] = X_scaled
                data["X_test_original"] = X_original
                data["y_test"] = y
                data["feature_names"] = feature_names
                
                # Fazer predições
                data["y_pred"] = data["model"].predict(X_scaled)
                data["y_proba"] = data["model"].predict_proba(X_scaled)
                
                # Calcular métricas usando função compartilhada
                data["metrics"] = compute_validation_metrics(
                    y, 
                    data["y_pred"], 
                    data["y_proba"][:, 1]
                )
                
                print("✅ Usando funções compartilhadas de pré-processamento")
                return data
            except Exception as e:
                print(f"⚠️ Erro ao usar função compartilhada: {e}. Usando método legacy.")
        
        # FALLBACK: Método original (legacy)
        dataset_path = base_path / "EDA" / "cardio_data.parquet"
        scaler_path = base_path / "classification" / "scalers" / "robust_scaler.joblib"
        
        if dataset_path.exists() and scaler_path.exists() and data["model"] is not None:
            # Carregar dados reais
            df = pd.read_parquet(dataset_path)
            
            # Carregar scaler usado no treinamento
            scaler = joblib.load(scaler_path)
            
            # Preparar features necessárias
            # Converter cholesterol e gluc para binário (normal=0, alto=1 ou 2)
            df['cholesterol_high'] = (df['cholesterol'] > 1).astype(int)
            df['gluc_high'] = (df['gluc'] > 1).astype(int)
            
            # Ajustar gender (dataset tem 1=feminino, 2=masculino; modelo espera 0/1)
            df['gender'] = df['gender'] - 1
            
            # Selecionar features na ordem correta do modelo
            feature_order = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                           'smoke', 'alco', 'active', 'age_years', 'bmi', 
                           'cholesterol_high', 'gluc_high']
            
            X_test = df[feature_order].copy()
            
            # ✅ APLICAR SCALER (modelo foi treinado com dados escalonados!)
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=feature_order,
                index=X_test.index
            )
            
            data["X_test"] = X_test_scaled  # Dados ESCALADOS para o modelo
            data["X_test_original"] = X_test  # Dados ORIGINAIS para filtros
            data["y_test"] = df['cardio'].values  # Labels REAIS do Kaggle
            data["feature_names"] = feature_order
            
            # Fazer predições do modelo com os dados REAIS ESCALONADOS
            data["y_pred"] = data["model"].predict(data["X_test"])
            data["y_proba"] = data["model"].predict_proba(data["X_test"])
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
    
    return data

rf_data = load_data()

# Calcular métricas globais para exibição
accuracy = 0.0
precision = 0.0
recall = 0.0
f1_score = 0.0
auc_roc = 0.0

if rf_data['y_test'] is not None and rf_data['y_pred'] is not None:
    accuracy = np.mean(rf_data['y_test'] == rf_data['y_pred']) * 100
    
    # Usar métricas da função compartilhada se disponível
    if rf_data.get('metrics'):
        precision = rf_data['metrics'].get('precision', 0) * 100  # Converter para porcentagem
        recall = rf_data['metrics'].get('recall', 0) * 100
        f1_score = rf_data['metrics'].get('f1', 0) * 100
        auc_roc = rf_data['metrics'].get('auc_roc', 0)  # AUC já está em escala 0-1

# ================== HELPERS ==================

def make_section_header(icon, title, subtitle):
    """Cria cabeçalho de seção com estilo vibrante."""
    return html.Div([
        html.Div([
            html.H3([
                html.I(className=f"bi bi-{icon} me-3", 
                      style={"color": PALETTE['accent'], "fontSize": "36px"}),
                title
            ], className="mb-2", style={"fontWeight": "700", "color": PALETTE['dark']}),
            html.P(subtitle, className="text-muted", style={"fontSize": "16px"}),
        ], className="text-center py-4")
    ], style={
        "borderBottom": f"4px solid {PALETTE['gradient_start']}",
        "marginBottom": "30px",
        "background": f"linear-gradient(135deg, {PALETTE['light']} 0%, #ffffff 100%)"
    })

# ================== LAYOUT ==================

layout = dbc.Container([
    
    # ========== HERO SECTION ==========
    html.Div([
        html.Div([
            html.H1("🌲 Random Forest Classifier", 
                   className="display-4 fw-bold text-white mb-3"),
            html.P("Avaliação Completa do Modelo de Risco Cardiovascular", 
                  className="lead text-white-50 mb-4"),
            
            # MÉTRICAS GLOBAIS DE AVALIAÇÃO
            dbc.Row([
                # Precision
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-bullseye", 
                                  style={"fontSize": "24px", "color": "white", "marginBottom": "8px"}),
                            html.P("Precision", className="text-white-50 mb-1", 
                                  style={"fontSize": "14px"}),
                            html.H2(f"{precision:.1f}%", 
                                   className="text-white mb-0 fw-bold"),
                        ], className="text-center p-3", 
                           style={"backgroundColor": "rgba(255,255,255,0.15)", 
                                  "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"}),
                        html.Span([
                            html.I(className="bi bi-info-circle", id="tooltip-precision",
                                  style={"color": "rgba(255,255,255,0.7)", "cursor": "pointer", "marginLeft": "8px"}),
                            dbc.Tooltip("Proporção de diagnósticos positivos corretos. Alta precision = poucos falsos positivos.",
                                      target="tooltip-precision")
                        ])
                    ])
                ], md=3),
                
                # Recall
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-heart-pulse", 
                                  style={"fontSize": "24px", "color": "white", "marginBottom": "8px"}),
                            html.P("Recall", className="text-white-50 mb-1", 
                                  style={"fontSize": "14px"}),
                            html.H2(f"{recall:.1f}%", 
                                   className="text-white mb-0 fw-bold"),
                        ], className="text-center p-3", 
                           style={"backgroundColor": "rgba(255,255,255,0.15)", 
                                  "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"}),
                        html.Span([
                            html.I(className="bi bi-info-circle", id="tooltip-recall",
                                  style={"color": "rgba(255,255,255,0.7)", "cursor": "pointer", "marginLeft": "8px"}),
                            dbc.Tooltip("Capacidade de identificar todos os casos positivos. Alto recall = poucos falsos negativos.",
                                      target="tooltip-recall")
                        ])
                    ])
                ], md=3),
                
                # F1-Score
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-graph-up", 
                                  style={"fontSize": "24px", "color": "white", "marginBottom": "8px"}),
                            html.P("F1-Score", className="text-white-50 mb-1", 
                                  style={"fontSize": "14px"}),
                            html.H2(f"{f1_score:.1f}%", 
                                   className="text-white mb-0 fw-bold"),
                        ], className="text-center p-3", 
                           style={"backgroundColor": "rgba(255,255,255,0.15)", 
                                  "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"}),
                        html.Span([
                            html.I(className="bi bi-info-circle", id="tooltip-f1",
                                  style={"color": "rgba(255,255,255,0.7)", "cursor": "pointer", "marginLeft": "8px"}),
                            dbc.Tooltip("Média harmônica entre precision e recall. Balanceia ambas as métricas.",
                                      target="tooltip-f1")
                        ])
                    ])
                ], md=3),
                
                # AUC-ROC
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-activity", 
                                  style={"fontSize": "24px", "color": "white", "marginBottom": "8px"}),
                            html.P("AUC-ROC", className="text-white-50 mb-1", 
                                  style={"fontSize": "14px"}),
                            html.H2(f"{auc_roc:.3f}", 
                                   className="text-white mb-0 fw-bold"),
                        ], className="text-center p-3", 
                           style={"backgroundColor": "rgba(255,255,255,0.15)", 
                                  "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"}),
                        html.Span([
                            html.I(className="bi bi-info-circle", id="tooltip-auc",
                                  style={"color": "rgba(255,255,255,0.7)", "cursor": "pointer", "marginLeft": "8px"}),
                            dbc.Tooltip("Área sob a curva ROC. Mede capacidade discriminativa (0.5=aleatório, 1.0=perfeito).",
                                      target="tooltip-auc")
                        ])
                    ])
                ], md=3),
            ], className="mt-4")
        ], className="container py-5")
    ], style={
        "background": f"linear-gradient(135deg, {PALETTE['gradient_start']} 0%, {PALETTE['gradient_end']} 100%)",
        "marginBottom": "40px",
        "borderRadius": "0 0 30px 30px",
        "boxShadow": "0 10px 40px rgba(0,0,0,0.2)"
    }),
    
    # ========== TABS DE NAVEGAÇÃO ==========
    dbc.Tabs([
        # ========== TAB 1: OVERVIEW ==========
        dbc.Tab(
            label="📊 Overview",
            tab_id="tab-overview",
            children=[
                html.Div([
                    make_section_header("graph-up", "Visão Geral do Desempenho", 
                                        "Métricas principais e matriz de confusão"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Label("🎯 Filtrar por Gênero:", className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='confusion-gender-filter',
                            options=[
                                {'label': '👥 Todos', 'value': 'all'},
                                {'label': '👩 Feminino', 'value': 0},
                                {'label': '👨 Masculino', 'value': 1}
                            ],
                            value='all',
                            clearable=False,
                            style={"marginBottom": "20px"}
                        ),
                    ]),
                    dcc.Graph(id='confusion-matrix-viz', config={'displayModeBar': True})
                ])
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none"})
        ], md=12, className="mb-5"),
    ])
                ], className="p-4")
            ]
        ),
        
        # ========== TAB 2: CURVAS DE DESEMPENHO ==========
        dbc.Tab(
            label="📈 Performance Curves",
            tab_id="tab-curves",
            children=[
                html.Div([
                    make_section_header("activity", "Curvas de Desempenho", 
                                        "Análise detalhada da capacidade preditiva do modelo"),
    
    dbc.Row([
        # Curva ROC
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="bi bi-graph-up me-2", style={"color": PALETTE['accent']}),
                        "Curva ROC"
                    ], className="mb-3 fw-bold", style={"fontSize": "18px"}),
                    html.P(
                        "Receiver Operating Characteristic",
                        className="text-muted mb-2", 
                        style={"fontSize": "13px", "fontStyle": "italic"}
                    ),
                    html.P([
                        "Mostra o trade-off entre ",
                        html.Strong("Taxa de Verdadeiros Positivos (TPR)"),
                        " e ",
                        html.Strong("Taxa de Falsos Positivos (FPR)"),
                        ". Quanto mais próxima do canto superior esquerdo, melhor o modelo."
                    ], className="text-muted small mb-4"),
                    html.Div(
                        build_roc_curve(rf_data['y_test'], rf_data['y_proba'][:, 1]) if rf_data['y_test'] is not None else html.Div("Dados não disponíveis"),
                        style={"marginTop": "20px"}
                    )
                ], style={"padding": "25px"})
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none", "height": "100%"})
        ], md=6, className="mb-5"),
        
        # Curva Precision-Recall
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="bi bi-bullseye me-2", style={"color": PALETTE['accent']}),
                        "Curva Precision-Recall"
                    ], className="mb-3 fw-bold", style={"fontSize": "18px"}),
                    html.P(
                        "Equilíbrio entre Precisão e Cobertura",
                        className="text-muted mb-2", 
                        style={"fontSize": "13px", "fontStyle": "italic"}
                    ),
                    html.P([
                        "Mostra o equilíbrio entre ",
                        html.Strong("Precision (acurácia dos positivos)"),
                        " e ",
                        html.Strong("Recall (cobertura dos positivos)"),
                        ". Ideal para datasets desbalanceados."
                    ], className="text-muted small mb-4"),
                    html.Div(
                        build_precision_recall_curve(rf_data['y_test'], rf_data['y_proba'][:, 1]) if rf_data['y_test'] is not None else html.Div("Dados não disponíveis"),
                        style={"marginTop": "20px"}
                    )
                ], style={"padding": "25px"})
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none", "height": "100%"})
        ], md=6, className="mb-5"),
    ], className="mb-4"),
    
    # Curva de Calibração (largura completa)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="bi bi-sliders me-2", style={"color": PALETTE['accent']}),
                        "Curva de Calibração & Brier Score"
                    ], className="mb-3 fw-bold", style={"fontSize": "18px"}),
                    html.P(
                        "Avaliação da Qualidade das Probabilidades Preditas",
                        className="text-muted mb-2", 
                        style={"fontSize": "13px", "fontStyle": "italic"}
                    ),
                    html.P([
                        "Avalia se as probabilidades preditas correspondem às frequências reais. ",
                        "Uma curva próxima à diagonal indica boa calibração. ",
                        html.Strong("Brier Score "),
                        "mede o erro médio quadrático das probabilidades (quanto menor, melhor)."
                    ], className="text-muted small mb-4"),
                    html.Div(
                        build_calibration_curve(rf_data['y_test'], rf_data['y_proba'][:, 1]) if rf_data['y_test'] is not None else html.Div("Dados não disponíveis"),
                        style={"marginTop": "20px"}
                    )
                ], style={"padding": "25px"})
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none"})
        ], md=12, className="mb-5"),
    ])
                ], className="p-4")
            ]
        ),
        
        # ========== TAB 3: INTERPRETAÇÃO (FEATURE IMPORTANCE) ==========
        dbc.Tab(
            label="⭐ Interpretation",
            tab_id="tab-interpretation",
            children=[
                html.Div([
                    make_section_header("stars", "Importância das Features", 
                                        "Quais variáveis mais influenciam o modelo?"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Label("📊 Número de Features:", className="fw-bold mb-2"),
                        dcc.Slider(
                            id='n-features-slider',
                            min=5,
                            max=12,
                            step=1,
                            value=10,
                            marks={i: str(i) for i in range(5, 13)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], className="mb-3"),
                    dcc.Graph(id='feature-importance-viz', config={'displayModeBar': True})
                ])
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none"})
        ], md=12, className="mb-5"),
    ])
                ], className="p-4")
            ]
        ),
        
        # ========== TAB 4: ANÁLISE EXPLORATÓRIA (EDA) ==========
        dbc.Tab(
            label="🔍 EDA",
            tab_id="tab-eda",
            children=[
                html.Div([
                    make_section_header("binoculars", "Análise Exploratória", 
                                        "Compare features e descubra padrões"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎛️ Configurações de Análise", className="mb-3 fw-bold"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Eixo X:", className="fw-bold small"),
                            dcc.Dropdown(
                                id='scatter-feature-x',
                                options=[{'label': col, 'value': col} for col in rf_data['feature_names']],
                                value='ap_hi',
                                clearable=False
                            ),
                        ], md=6),
                        dbc.Col([
                            html.Label("Eixo Y:", className="fw-bold small"),
                            dcc.Dropdown(
                                id='scatter-feature-y',
                                options=[{'label': col, 'value': col} for col in rf_data['feature_names']],
                                value='ap_lo',
                                clearable=False
                            ),
                        ], md=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Faixa Etária:", className="fw-bold small"),
                            dcc.RangeSlider(
                                id='scatter-age-filter',
                                min=30,
                                max=70,
                                step=5,
                                value=[30, 70],
                                marks={i: f'{i}' for i in range(30, 71, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], md=12),
                    ]),
                ])
            ], className="mb-3", style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none"})
        ], md=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='scatter-plot-viz', config={'displayModeBar': True})
                ])
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none"})
        ], md=12, className="mb-5"),
    ]),
    
    # ========== SEÇÃO 4: DISTRIBUIÇÕES ==========
    make_section_header("bar-chart", "Distribuições por Diagnóstico", 
                        "Como as features se comportam em cada classe?"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Label("📈 Selecione a Feature:", className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='dist-feature-select',
                            options=[{'label': col, 'value': col} for col in rf_data['feature_names']],
                            value='ap_hi',
                            clearable=False
                        ),
                    ], className="mb-3"),
                    dcc.Graph(id='distribution-viz', config={'displayModeBar': True})
                ])
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none"})
        ], md=12, className="mb-5"),
    ])
                ], className="p-4")
            ]
        )
    ], id="main-tabs", active_tab="tab-overview", className="mb-4"),
    
    # Footer
    html.Div([
        html.Hr(style={"borderTop": f"2px solid {PALETTE['muted']}"}),
        html.P("Dashboard desenvolvido com Dash & Plotly | Random Forest Classifier", 
               className="text-center text-muted small py-3")
    ])
    
], fluid=True, className="px-4 py-4", style={"backgroundColor": "#fafbfc"})


# ================== CALLBACKS ==================

@callback(
    Output('confusion-matrix-viz', 'figure'),
    Input('confusion-gender-filter', 'value')
)
def update_confusion_matrix(gender):
    """Atualiza matriz de confusão baseado no filtro de gênero."""
    # Usar dados ORIGINAIS (não escalados) para filtros
    df = rf_data['X_test_original'].copy()
    df['prediction'] = rf_data['y_pred']
    df['actual'] = rf_data['y_test']
    
    if gender != 'all':
        df = df[df['gender'] == gender]
    
    cm = confusion_matrix(df['actual'], df['prediction'])
    
    # Criar heatmap colorido
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predito: Sem Doença', 'Predito: Com Doença'],
        y=['Real: Sem Doença', 'Real: Com Doença'],
        colorscale=[[0, PALETTE['success']], [0.5, PALETTE['chart_2']], [1, PALETTE['warn']]],
        text=cm,
        texttemplate='<b>%{text}</b>',
        textfont={"size": 24, "color": "white"},
        showscale=True,
        hovertemplate='%{y}<br>%{x}<br><b>Quantidade: %{z}</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Matriz de Confusão",
            font=dict(size=22, color=PALETTE['dark'], family="Inter")
        ),
        height=500,
        template=create_plotly_template()
    )
    
    return fig


@callback(
    Output('feature-importance-viz', 'figure'),
    Input('n-features-slider', 'value')
)
def update_feature_importance(n_features):
    """Atualiza gráfico de importância das features."""
    importances = rf_data['model'].feature_importances_
    feature_df = pd.DataFrame({
        'feature': rf_data['feature_names'],
        'importance': importances
    }).sort_values('importance', ascending=False).head(n_features)
    
    # Traduzir nomes
    translations = {
        'ap_hi': 'Pressão Sistólica', 'ap_lo': 'Pressão Diastólica',
        'age_years': 'Idade', 'bmi': 'IMC', 'weight': 'Peso',
        'height': 'Altura', 'cholesterol_high': 'Colesterol Alto',
        'gluc_high': 'Glicose Alta', 'gender': 'Gênero',
        'smoke': 'Fumante', 'alco': 'Álcool', 'active': 'Ativo'
    }
    feature_df['feature_pt'] = feature_df['feature'].map(translations)
    
    # Cores vibrantes por importância
    colors = px.colors.sequential.Plasma_r
    
    fig = go.Figure(go.Bar(
        x=feature_df['importance'],
        y=feature_df['feature_pt'],
        orientation='h',
        marker=dict(
            color=feature_df['importance'],
            colorscale=colors,
            showscale=True,
            colorbar=dict(title="Importância", thickness=15)
        ),
        text=feature_df['importance'].apply(lambda x: f'{x:.4f}'),
        textposition='outside',
        textfont=dict(size=12, color=PALETTE['dark']),
        hovertemplate='<b>%{y}</b><br>Importância: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Top {n_features} Features Mais Importantes",
            font=dict(size=20, color=PALETTE['dark'])
        ),
        xaxis_title="Importância (Gini Index)",
        height=max(400, n_features * 40),
        template=create_plotly_template(),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


@callback(
    Output('scatter-plot-viz', 'figure'),
    [Input('scatter-feature-x', 'value'),
     Input('scatter-feature-y', 'value'),
     Input('scatter-age-filter', 'value')]
)
def update_scatter(feature_x, feature_y, age_range):
    """Atualiza scatter plot interativo."""
    # Usar dados ORIGINAIS (não escalados) para visualização e filtros
    df = rf_data['X_test_original'].copy()
    df['prediction'] = rf_data['y_pred']
    df['actual'] = rf_data['y_test']
    df['probability'] = rf_data['y_proba'][:, 1]
    
    df = df[(df['age_years'] >= age_range[0]) & (df['age_years'] <= age_range[1])]
    
    fig = px.scatter(
        df,
        x=feature_x,
        y=feature_y,
        color=df['actual'].map({0: 'Sem Doença', 1: 'Com Doença'}),
        size='probability',
        hover_data=['age_years', 'gender', 'bmi'],
        title=f'Relação: {feature_x} vs {feature_y}',
        color_discrete_map={
            'Sem Doença': '#1E88E5',  # Azul vibrante
            'Com Doença': '#FF6B6B'   # Vermelho coral
        },
        template=create_plotly_template()
    )
    
    fig.update_traces(marker=dict(line=dict(width=0.8, color='white'), opacity=0.8))
    fig.update_layout(
        height=550,
        title=dict(font=dict(size=20, color=PALETTE['dark'])),
        legend=dict(
            title="Diagnóstico", 
            font=dict(size=14),
            orientation='h',
            yanchor='top',
            y=1.1,
            xanchor='center',
            x=0.5
        )
    )
    
    return fig


@callback(
    Output('distribution-viz', 'figure'),
    Input('dist-feature-select', 'value')
)
def update_distribution(feature):
    """Atualiza distribuição da feature selecionada."""
    # Usar dados ORIGINAIS (não escalados) para visualização
    df = rf_data['X_test_original'].copy()
    df['actual'] = rf_data['y_test']
    
    fig = go.Figure()
    
    # SEM DOENÇA - AZUL VIBRANTE (bem visível)
    fig.add_trace(go.Histogram(
        x=df[df['actual'] == 0][feature],
        name='Sem Doença',
        marker=dict(
            color='rgba(30, 136, 229, 0.6)',  # Azul vibrante com 60% opacidade
            line=dict(color='rgba(30, 136, 229, 1)', width=1.5)
        ),
        nbinsx=40,
        hovertemplate='<b>Sem Doença</b><br>%{x}<br>Frequência: %{y}<extra></extra>'
    ))
    
    # COM DOENÇA - VERMELHO CORAL (bem visível)
    fig.add_trace(go.Histogram(
        x=df[df['actual'] == 1][feature],
        name='Com Doença',
        marker=dict(
            color='rgba(255, 107, 107, 0.6)',  # Vermelho coral com 60% opacidade
            line=dict(color='rgba(255, 107, 107, 1)', width=1.5)
        ),
        nbinsx=40,
        hovertemplate='<b>Com Doença</b><br>%{x}<br>Frequência: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Distribuição de {feature} por Diagnóstico',
            font=dict(size=20, color=PALETTE['dark'])
        ),
        xaxis_title=feature,
        yaxis_title='Frequência',
        barmode='overlay',
        height=450,
        template=create_plotly_template(),
        legend=dict(
            font=dict(size=14),
            orientation='h',
            yanchor='top',
            y=1.1,
            xanchor='center',
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


# Fim dos callbacks
