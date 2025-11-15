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
    from classification.prediction_service import predict_single
    USE_SHARED_FUNCTIONS = True
    PREDICTION_SERVICE_AVAILABLE = True
except ImportError:
    print("⚠️ Funções compartilhadas não disponíveis. Usando modo legacy.")
    USE_SHARED_FUNCTIONS = False
    PREDICTION_SERVICE_AVAILABLE = False

# Importar SHAP (opcional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP não disponível. Instale com: pip install shap")
    SHAP_AVAILABLE = False

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
        "metrics": {},
        "shap_values": None,
        "shap_base_value": None
    }
    
    try:
        # Carregar PIPELINE (modelo novo corrigido)
        pipeline_path = base_path / "classification" / "models" / "random_forest_pipeline.joblib"
        pipeline = None
        
        if pipeline_path.exists():
            pipeline = joblib.load(pipeline_path)
            # Extrair o classificador do pipeline para uso no SHAP
            data["model"] = pipeline.named_steps['classifier']
            print(f"✅ Pipeline carregado! Modelo: {type(data['model']).__name__}")
        else:
            # Fallback para modelo antigo
            model_path = base_path / "classification" / "models" / "random_forest_model.joblib"
            if model_path.exists():
                data["model"] = joblib.load(model_path)
                print("⚠️ Usando modelo antigo (sem pipeline)")
        
        # USAR FUNÇÃO COMPARTILHADA se disponível (elimina duplicação!)
        if USE_SHARED_FUNCTIONS and pipeline is not None:
            try:
                # Pipeline precisa de dados NÃO escalonados
                X_scaled, X_original, y, feature_names = load_and_preprocess_data()
                
                data["X_test_original"] = X_original
                data["y_test"] = y
                data["feature_names"] = feature_names
                
                # Fazer predições COM O PIPELINE (usa dados originais)
                data["y_pred"] = pipeline.predict(X_original)
                data["y_proba"] = pipeline.predict_proba(X_original)
                
                # Para SHAP: precisa dos dados APÓS o scaler do pipeline
                scaler = pipeline.named_steps['scaler']
                X_scaled_by_pipeline = pd.DataFrame(
                    scaler.transform(X_original),
                    columns=feature_names,
                    index=X_original.index
                )
                data["X_test"] = X_scaled_by_pipeline
                
                # Calcular métricas usando função compartilhada
                data["metrics"] = compute_validation_metrics(
                    y, 
                    data["y_pred"], 
                    data["y_proba"][:, 1]
                )
                
                print("✅ Usando pipeline com funções compartilhadas de pré-processamento")
            except Exception as e:
                print(f"⚠️ Erro ao usar função compartilhada com pipeline: {e}")
                import traceback
                traceback.print_exc()
                
        elif USE_SHARED_FUNCTIONS and data["model"] is not None:
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
        
        if dataset_path.exists():
            # Carregar dados reais
            df = pd.read_parquet(dataset_path)
            
            # Preparar features necessárias
            # Converter cholesterol e gluc para binário (normal=0, alto=1 ou 2)
            df['cholesterol_high'] = (df['cholesterol'] > 1).astype(int)
            df['gluc_high'] = (df['gluc'] > 1).astype(int)
            
            # Ajustar gender (dataset tem 1=feminino, 2=masculino; modelo espera 0/1)
            df['gender'] = df['gender'] - 1
            
            # Selecionar features na ordem correta do modelo (SEM height/weight)
            feature_order = ['gender', 'ap_hi', 'ap_lo', 
                           'smoke', 'alco', 'active', 'age_years', 'bmi', 
                           'cholesterol_high', 'gluc_high']
            
            X_test_original = df[feature_order].copy()
            data["X_test_original"] = X_test_original
            data["y_test"] = df['cardio'].values
            data["feature_names"] = feature_order
            
            # Lógica diferente para PIPELINE vs MODELO ANTIGO
            if pipeline is not None:
                # PIPELINE: passa dados NÃO escalonados (pipeline faz o scaling)
                scaler = pipeline.named_steps['scaler']
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test_original),
                    columns=feature_order,
                    index=X_test_original.index
                )
                data["X_test"] = X_test_scaled
                
                # Predições com pipeline (usa dados originais)
                data["y_pred"] = pipeline.predict(X_test_original)
                data["y_proba"] = pipeline.predict_proba(X_test_original)
                
                # Calcular métricas
                if USE_SHARED_FUNCTIONS:
                    data["metrics"] = compute_validation_metrics(
                        data["y_test"],
                        data["y_pred"],
                        data["y_proba"][:, 1]
                    )
                
                print("✅ Usando pipeline com método legacy de carregamento de dados")
                
            elif data["model"] is not None:
                # MODELO ANTIGO: precisa de scaler separado
                scaler_path = base_path / "classification" / "scalers" / "robust_scaler.joblib"
                
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                    X_test_scaled = pd.DataFrame(
                        scaler.transform(X_test_original),
                        columns=feature_order,
                        index=X_test_original.index
                    )
                    data["X_test"] = X_test_scaled
                    
                    # Predições com modelo antigo (usa dados escalonados)
                    data["y_pred"] = data["model"].predict(X_test_scaled)
                    data["y_proba"] = data["model"].predict_proba(X_test_scaled)
                    
                    # Calcular métricas
                    if USE_SHARED_FUNCTIONS:
                        data["metrics"] = compute_validation_metrics(
                            data["y_test"],
                            data["y_pred"],
                            data["y_proba"][:, 1]
                        )
                    
                    print("✅ Usando modelo antigo com método legacy")
                else:
                    print(f"⚠️ Scaler não encontrado: {scaler_path}")
            else:
                print("⚠️ Nenhum modelo ou pipeline disponível")
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
    
    # Calcular SHAP values (apenas uma amostra para performance)
    # DESABILITADO: Muito lento no carregamento inicial
    # Para habilitar, descomente o bloco abaixo
    """
    if SHAP_AVAILABLE and data["model"] is not None and data["X_test"] is not None:
        try:
            print("📊 Calculando SHAP values para 2000 amostras...")
            sample_size = min(2000, len(data["X_test"]))
            sample_indices = np.random.choice(len(data["X_test"]), size=sample_size, replace=False)
            X_sample = data["X_test"].iloc[sample_indices]
            
            explainer = shap.TreeExplainer(data["model"])
            shap_values_raw = explainer.shap_values(X_sample)
            
            # Extrair classe positiva
            if isinstance(shap_values_raw, list):
                data["shap_values"] = shap_values_raw[1]
                data["shap_base_value"] = explainer.expected_value[1]
            elif len(shap_values_raw.shape) == 3:
                data["shap_values"] = shap_values_raw[:, :, 1]
                data["shap_base_value"] = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                data["shap_values"] = shap_values_raw
                data["shap_base_value"] = explainer.expected_value
            
            data["X_sample"] = X_sample
            data["X_sample_original"] = data["X_test_original"].iloc[sample_indices]
            data["y_sample"] = data["y_test"][sample_indices]
            data["sample_indices"] = sample_indices
            print(f"✅ SHAP values calculados para {sample_size} amostras")
        except Exception as e:
            print(f"⚠️ Erro ao calcular SHAP values: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP não está instalado. Instale com: pip install shap")
        else:
            print("⚠️ Modelo ou dados não disponíveis para SHAP")
    """
    print("ℹ️ SHAP desabilitado para carregamento rápido do dashboard")
    
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
    
    # ========== SEÇÃO DE PREDIÇÃO INTERATIVA ==========
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H3([
                    html.I(className="bi bi-heart-pulse me-3", style={"color": PALETTE['accent']}),
                    "🩺 Predição de Risco Cardiovascular"
                ], className="mb-4 fw-bold text-center"),
                
                html.P("Insira os dados do paciente para prever o risco de doença cardiovascular:",
                      className="text-muted text-center mb-4"),
                
                dbc.Row([
                    # Coluna 1: Dados Demográficos
                    dbc.Col([
                        html.H5("👤 Dados Demográficos", className="mb-3 fw-bold"),
                        
                        html.Label("Idade (anos):", className="fw-bold mb-1"),
                        dbc.Input(id="input-age", type="number", value=45, min=29, max=65, 
                                 className="mb-3", placeholder="Ex: 45"),
                        
                        html.Label("Gênero:", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="input-gender",
                            options=[
                                {'label': '👩 Feminino', 'value': 0},
                                {'label': '👨 Masculino', 'value': 1}
                            ],
                            value=0,
                            clearable=False,
                            className="mb-3"
                        ),
                        
                        html.Label("IMC (kg/m²):", className="fw-bold mb-1"),
                        dbc.Input(id="input-bmi", type="number", value=25.0, min=15, max=50, step=0.1,
                                 className="mb-3", placeholder="Ex: 25.0"),
                    ], md=4),
                    
                    # Coluna 2: Pressão e Hábitos
                    dbc.Col([
                        html.H5("💉 Pressão Arterial", className="mb-3 fw-bold"),
                        
                        html.Label("Pressão Sistólica (ap_hi):", className="fw-bold mb-1"),
                        dbc.Input(id="input-ap-hi", type="number", value=120, min=90, max=200,
                                 className="mb-3", placeholder="Ex: 120"),
                        
                        html.Label("Pressão Diastólica (ap_lo):", className="fw-bold mb-1"),
                        dbc.Input(id="input-ap-lo", type="number", value=80, min=60, max=130,
                                 className="mb-3", placeholder="Ex: 80"),
                        
                        html.H5("🏃 Estilo de Vida", className="mb-3 fw-bold mt-3"),
                        
                        dbc.Checklist(
                            id="input-active",
                            options=[{'label': ' Pratica atividade física', 'value': 1}],
                            value=[1],
                            className="mb-2"
                        ),
                    ], md=4),
                    
                    # Coluna 3: Fatores de Risco
                    dbc.Col([
                        html.H5("⚠️ Fatores de Risco", className="mb-3 fw-bold"),
                        
                        dbc.Checklist(
                            id="input-smoke",
                            options=[{'label': ' Fumante', 'value': 1}],
                            value=[],
                            className="mb-2"
                        ),
                        
                        dbc.Checklist(
                            id="input-alco",
                            options=[{'label': ' Consome álcool', 'value': 1}],
                            value=[],
                            className="mb-3"
                        ),
                        
                        html.Label("Colesterol:", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="input-cholesterol",
                            options=[
                                {'label': '✅ Normal', 'value': 0},
                                {'label': '⚠️ Alto', 'value': 1}
                            ],
                            value=0,
                            clearable=False,
                            className="mb-3"
                        ),
                        
                        html.Label("Glicose:", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="input-glucose",
                            options=[
                                {'label': '✅ Normal', 'value': 0},
                                {'label': '⚠️ Alto', 'value': 1}
                            ],
                            value=0,
                            clearable=False,
                            className="mb-3"
                        ),
                    ], md=4),
                ]),
                
                # Botão de Predição
                html.Div([
                    dbc.Button(
                        [html.I(className="bi bi-play-fill me-2"), "🩺 Calcular Risco"],
                        id="btn-predict",
                        color="primary",
                        size="lg",
                        className="mt-3 px-5",
                        style={"fontSize": "18px", "fontWeight": "bold"}
                    )
                ], className="text-center"),
                
                # Resultado da Predição
                html.Div(id="prediction-result", className="mt-4")
            ], className="p-4")
        ], style={"boxShadow": "0 6px 20px rgba(0,0,0,0.15)", "border": "none", "borderRadius": "15px"})
    ], className="mb-5"),
    
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
        
        # ========== TAB 4: SHAP INTERPRETABILITY ==========
        dbc.Tab(
            label="🔬 SHAP Analysis",
            tab_id="tab-shap",
            children=[
                html.Div([
                    make_section_header("lightbulb", "SHAP - Explainability AI", 
                                        "Interpretabilidade avançada com SHapley Additive exPlanations"),
    
    # Galeria de Visualizações SHAP (imagens estáticas)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="bi bi-images me-2", style={"color": PALETTE['accent']}),
                        "Galeria de Visualizações SHAP"
                    ], className="mb-3 fw-bold", style={"fontSize": "20px"}),
                    html.P([
                        "Selecione a visualização para explorar diferentes aspectos da interpretabilidade do modelo."
                    ], className="text-muted small mb-3"),
                    
                    # Botões de navegação
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="bi bi-bar-chart-fill me-2"),
                            "Summary Plot"
                        ], id="btn-shap-summary", color="primary", outline=True, className="me-2"),
                        dbc.Button([
                            html.I(className="bi bi-graph-up me-2"),
                            "Feature Importance"
                        ], id="btn-shap-bar", color="primary", outline=True, className="me-2"),
                        dbc.Button([
                            html.I(className="bi bi-water me-2"),
                            "Waterfall (Amostra)"
                        ], id="btn-shap-waterfall", color="primary", outline=True),
                    ], className="mb-4 d-flex flex-wrap", style={"gap": "10px"}),
                    
                    # Container para a imagem
                    html.Div(id='shap-image-container', children=[
                        html.Img(
                            src='/assets/shap_summary.png',
                            style={
                                'width': '100%',
                                'maxWidth': '1000px',
                                'height': 'auto',
                                'display': 'block',
                                'margin': '0 auto',
                                'border': '1px solid #ddd',
                                'borderRadius': '8px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
                            },
                            id='shap-image'
                        )
                    ], className="text-center mb-3"),
                    
                    # Descrição dinâmica
                    html.Div(id='shap-description', children=[
                        dbc.Alert([
                            html.I(className="bi bi-lightbulb text-warning me-2"),
                            html.Strong("Summary Plot: "),
                            "Visualização global mostrando todas as features ordenadas por importância. ",
                            "Cada ponto representa uma amostra, a cor indica o valor da feature (vermelho=alto, azul=baixo), ",
                            "e a posição horizontal mostra o impacto SHAP na predição. ",
                            "Features no topo têm maior impacto global."
                        ], color="light", className="mb-0")
                    ])
                ])
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none"})
        ], md=12, className="mb-5"),
    ])
                ], className="p-4")
            ]
        ),
        
        # ========== TAB 5: ANÁLISE EXPLORATÓRIA (EDA) ==========
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

# Callback de Predição Interativa
@callback(
    Output('prediction-result', 'children'),
    Input('btn-predict', 'n_clicks'),
    [
        Input('input-age', 'value'),
        Input('input-gender', 'value'),
        Input('input-bmi', 'value'),
        Input('input-ap-hi', 'value'),
        Input('input-ap-lo', 'value'),
        Input('input-smoke', 'value'),
        Input('input-alco', 'value'),
        Input('input-active', 'value'),
        Input('input-cholesterol', 'value'),
        Input('input-glucose', 'value'),
    ],
    prevent_initial_call=True
)
def predict_cardiovascular_risk(n_clicks, age, gender, bmi, ap_hi, ap_lo, 
                               smoke, alco, active, cholesterol, glucose):
    """Realiza predição de risco cardiovascular."""
    if not PREDICTION_SERVICE_AVAILABLE:
        return dbc.Alert("⚠️ Serviço de predição não disponível", color="warning")
    
    try:
        # Preparar dados - checkboxes retornam lista
        patient_data = {
            'age_years': age,
            'gender': gender,
            'bmi': bmi,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'smoke': 1 if (smoke and len(smoke) > 0) else 0,
            'alco': 1 if (alco and len(alco) > 0) else 0,
            'active': 1 if (active and len(active) > 0) else 0,
            'cholesterol_high': cholesterol,
            'gluc_high': glucose
        }
        
        # Fazer predição
        result = predict_single(patient_data)
        probability = result['probability']  # Já vem em porcentagem
        risk_class = result['class']
        
        # Obter importâncias das features do resultado
        feature_contributions = result.get('feature_contributions', {})
        
        # Determinar cor, ícone e mensagem
        if probability < 30:
            color = "success"
            icon = "bi-heart-pulse-fill"
            risk_text = "BAIXO"
            message = "✅ O paciente apresenta baixo risco de doença cardiovascular."
            emoji = "💚"
        elif probability < 70:
            color = "warning"
            icon = "bi-exclamation-triangle-fill"
            risk_text = "MODERADO"
            message = "⚠️ O paciente apresenta risco moderado. Recomenda-se acompanhamento médico."
            emoji = "💛"
        else:
            color = "danger"
            icon = "bi-heart-fill"
            risk_text = "ALTO"
            message = "🚨 O paciente apresenta alto risco. É fundamental buscar avaliação médica."
            emoji = "❤️"
        
        # Criar lista de fatores mais relevantes
        feature_names_pt = {
            'ap_hi': 'Pressão Sistólica',
            'ap_lo': 'Pressão Diastólica',
            'bmi': 'IMC',
            'age_years': 'Idade',
            'cholesterol_high': 'Colesterol Alto',
            'gluc_high': 'Glicose Alta',
            'active': 'Atividade Física',
            'smoke': 'Tabagismo',
            'alco': 'Consumo de Álcool',
            'gender': 'Gênero'
        }
        
        # Ordenar features por importância (top 3)
        if feature_contributions:
            top_features = sorted(feature_contributions.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True)[:3]
            features_list = html.Ul([
                html.Li([
                    html.Strong(feature_names_pt.get(feat, feat)),
                    f": {value:.1f}% de contribuição"
                ], className="mb-1") 
                for feat, value in top_features
            ], className="text-start", style={"fontSize": "14px"})
        else:
            features_list = html.P("Detalhes não disponíveis", 
                                   className="text-muted small")
        
        return dbc.Card([
            dbc.CardBody([
                # Header com resultado
                html.Div([
                    html.H1(emoji, className="mb-3", style={"fontSize": "64px"}),
                    html.H2(f"{probability:.1f}%", 
                           className="mb-2 fw-bold",
                           style={"fontSize": "48px", "color": PALETTE['accent']}),
                    html.H4(f"Risco {risk_text}", 
                           className="mb-3",
                           style={"color": PALETTE['dark'], "fontWeight": "700"}),
                    html.P(message, className="mb-4", 
                          style={"fontSize": "16px"})
                ], className="text-center py-3"),
                
                # Separador
                html.Hr(style={"borderTop": f"2px solid {PALETTE['light']}"}),
                
                # Fatores mais relevantes
                html.Div([
                    html.H5([
                        html.I(className="bi bi-bar-chart-fill me-2"),
                        "📊 Principais Fatores de Influência"
                    ], className="mb-3 fw-bold text-center"),
                    features_list
                ], className="mt-3")
            ], className="p-4")
        ], color=color, outline=True, style={
            "borderWidth": "3px",
            "boxShadow": "0 6px 20px rgba(0,0,0,0.15)",
            "borderRadius": "15px"
        })
        
    except Exception as e:
        return dbc.Alert(f"❌ Erro na predição: {str(e)}", color="danger")

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


# ================== CALLBACKS SHAP ==================

@callback(
    [Output('shap-image', 'src'),
     Output('shap-description', 'children'),
     Output('btn-shap-summary', 'color'),
     Output('btn-shap-bar', 'color'),
     Output('btn-shap-waterfall', 'color')],
    [Input('btn-shap-summary', 'n_clicks'),
     Input('btn-shap-bar', 'n_clicks'),
     Input('btn-shap-waterfall', 'n_clicks')],
    prevent_initial_call=False
)
def update_shap_gallery(summary_clicks, bar_clicks, waterfall_clicks):
    """Atualiza galeria de imagens SHAP baseado no botão clicado."""
    ctx = dash.callback_context
    
    # Determinar qual botão foi clicado
    if not ctx.triggered:
        button_id = 'btn-shap-summary'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Configuração de cada visualização
    configs = {
        'btn-shap-summary': {
            'src': '/assets/shap_summary.png',
            'desc': dbc.Alert([
                html.I(className="bi bi-lightbulb text-warning me-2"),
                html.Strong("Summary Plot: "),
                "Visualização global mostrando ",
                html.Strong("todas as 10 features"),
                " ordenadas por importância. ",
                "Cada ponto representa uma amostra, a cor indica o valor da feature (vermelho=alto, azul=baixo), ",
                "e a posição horizontal mostra o impacto SHAP na predição. ",
                html.Br(),
                html.Br(),
                "📌 ",
                html.Strong("Como interpretar: "),
                "Features no topo têm maior impacto global no modelo. ",
                "Pontos vermelhos à direita indicam que valores altos da feature aumentam a probabilidade de doença cardiovascular."
            ], color="light", className="mb-0")
        },
        'btn-shap-bar': {
            'src': '/assets/shap_bar.png',
            'desc': dbc.Alert([
                html.I(className="bi bi-lightbulb text-warning me-2"),
                html.Strong("Feature Importance (SHAP): "),
                "Importância média absoluta de cada feature. ",
                "Mostra ",
                html.Strong("quanto cada variável contribui em média"),
                " para as predições do modelo, independente da direção (positiva ou negativa). ",
                html.Br(),
                html.Br(),
                "📌 ",
                html.Strong("Como interpretar: "),
                "Barras maiores = maior influência global no modelo. ",
                "Compare com a Feature Importance tradicional (Gini) na aba 'Interpretation' para ver as diferenças entre os métodos."
            ], color="light", className="mb-0")
        },
        'btn-shap-waterfall': {
            'src': '/assets/shap_waterfall_0.png',
            'desc': dbc.Alert([
                html.I(className="bi bi-lightbulb text-warning me-2"),
                html.Strong("Waterfall Plot (Explicação Local): "),
                "Explica ",
                html.Strong("uma predição individual"),
                " mostrando como cada feature contribuiu para o resultado daquele paciente específico. ",
                "Inicia no valor base (predição média) e adiciona/subtrai contribuições até chegar na predição final. ",
                html.Br(),
                html.Br(),
                "📌 ",
                html.Strong("Como interpretar: "),
                "Barras vermelhas empurram a predição para 'Com Doença', ",
                "barras azuis empurram para 'Sem Doença'. ",
                "O valor final f(x) é a predição do modelo para esta amostra específica. ",
                "Ideal para explicar decisões individuais em contextos clínicos."
            ], color="light", className="mb-0")
        }
    }
    
    # Obter configuração da visualização selecionada
    config = configs.get(button_id, configs['btn-shap-summary'])
    
    # Definir cores dos botões (primary para selecionado, outline para outros)
    colors = {
        'summary': 'primary' if button_id == 'btn-shap-summary' else 'secondary',
        'bar': 'primary' if button_id == 'btn-shap-bar' else 'secondary',
        'waterfall': 'primary' if button_id == 'btn-shap-waterfall' else 'secondary'
    }
    
    return config['src'], config['desc'], colors['summary'], colors['bar'], colors['waterfall']


# Fim dos callbacks
