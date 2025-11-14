"""
Página de análise do modelo XGBoost.
Exibe métricas, gráficos de importância e comparações.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
import os
from pathlib import Path

# Adicionar paths para imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.utils import (
    make_page_header, make_card, build_metric_grid,
    make_tabs, make_alert, make_empty_state
)
from components.cards import build_classification_report_card
from styles import PALETTE, SPACING

# Registrar página
dash.register_page(
    __name__,
    path="/xgboost",
    name="XGBoost",
    title="XGBoost - Dashboard",
    icon="lightning-charge-fill",
)

# ================== CAMINHOS DOS ARTEFATOS ==================
BASE_DIR = Path(__file__).parent.parent.parent
XGBOOST_DIR = BASE_DIR / "classification" / "xgboost_local"
RESULTS_DIR = XGBOOST_DIR / "results"


# ================== FUNÇÕES DE CARREGAMENTO ==================
def load_xgboost_artifacts():
    """Carrega artefatos do modelo XGBoost."""
    artifacts = {
        "reports": [],
        "error": None,
    }
    
    try:
        # Listar todos os relatórios disponíveis
        if RESULTS_DIR.exists():
            reports = list(RESULTS_DIR.glob("xgboost_report_*.txt"))
            artifacts["reports"] = sorted([str(r.name) for r in reports], reverse=True)
    
    except Exception as e:
        artifacts["error"] = str(e)
    
    return artifacts


def load_latest_report():
    """Carrega o relatório mais recente do XGBoost."""
    try:
        if RESULTS_DIR.exists():
            reports = sorted(RESULTS_DIR.glob("xgboost_report_*.txt"), reverse=True)
            if reports:
                with open(reports[0], "r", encoding="utf-8") as f:
                    return f.read()
    except:
        pass
    return None


# ================== LAYOUT DA PÁGINA ==================
def layout():
    """Cria layout da página XGBoost."""
    
    # Carregar artefatos
    artifacts = load_xgboost_artifacts()
    report_text = load_latest_report()
    
    # Header da página
    page_header = make_page_header(
        "XGBoost Classifier",
        "Modelo de classificação com Gradient Boosting extremamente otimizado",
        icon="lightning-charge-fill"
    )
    
    # Alertas
    alerts = []
    if artifacts["error"]:
        alerts.append(make_alert(
            f"⚠️ Erro ao carregar artefatos: {artifacts['error']}",
            color="warning"
        ))
    
    if not artifacts["reports"]:
        alerts.append(make_alert(
            "ℹ️ Nenhum relatório encontrado. Execute o modelo XGBoost primeiro em classification/xgboost_local/",
            color="info"
        ))
    
    # Seção de métricas (placeholder - pode ser expandido)
    metrics_section = html.Div([
        html.H4("Informações do Modelo", className="mb-3"),
        make_card(
            "Características do XGBoost",
            html.Ul([
                html.Li("Gradient Boosting otimizado para velocidade e performance"),
                html.Li("Regularização L1 e L2 para prevenir overfitting"),
                html.Li("Suporta dados categóricos nativamente"),
                html.Li("Paralelização eficiente para treinamento rápido"),
            ]),
            icon="info-circle"
        ),
    ])
    
    # Relatórios disponíveis
    if report_text:
        reports_section = build_classification_report_card(report_text)
    else:
        reports_section = make_empty_state(
            "Nenhum relatório gerado ainda. Execute o pipeline XGBoost.",
            icon="file-earmark-text"
        )
    
    # Lista de relatórios históricos
    if artifacts["reports"]:
        history_items = [
            html.Li([
                html.I(className="bi bi-file-text me-2"),
                report
            ])
            for report in artifacts["reports"][:10]  # Mostrar últimos 10
        ]
        
        history_card = make_card(
            "Histórico de Execuções",
            html.Div([
                html.P(f"Total de execuções: {len(artifacts['reports'])}", className="text-muted mb-3"),
                html.Ul(history_items, style={"fontSize": "14px"}),
            ]),
            icon="clock-history"
        )
    else:
        history_card = make_empty_state("Nenhum histórico disponível")
    
    # Dashboard interativo (link para app_xgboost.py)
    dashboard_info = make_card(
        "Dashboard Interativo com Dash",
        html.Div([
            html.P("O XGBoost possui um dashboard interativo separado com visualizações dinâmicas:"),
            html.Ul([
                html.Li("Análise de importância de features"),
                html.Li("Métricas detalhadas por classe"),
                html.Li("Visualizações interativas com Plotly"),
            ]),
            html.Hr(),
            html.P([
                "Execute: ",
                html.Code("python classification/xgboost_local/app_xgboost.py", style={"backgroundColor": "#f5f5f5", "padding": "4px 8px", "borderRadius": "4px"}),
            ], className="mb-0"),
        ]),
        icon="speedometer2"
    )
    
    # Instruções
    instructions = make_card(
        "Como usar",
        html.Ol([
            html.Li("Navegue até classification/xgboost_local/"),
            html.Li("Execute: python app_xgboost.py"),
            html.Li("Acesse o dashboard interativo em http://localhost:8050"),
            html.Li("Ou visualize os relatórios estáticos acima"),
        ]),
        icon="book"
    )
    
    # Layout completo
    return html.Div([
        page_header,
        *alerts,
        metrics_section,
        html.Hr(style={"margin": f"{SPACING['lg']} 0"}),
        dbc.Row([
            dbc.Col(reports_section, width=12, lg=8),
            dbc.Col(history_card, width=12, lg=4),
        ], className="mb-4"),
        dashboard_info,
        html.Hr(style={"margin": f"{SPACING['lg']} 0"}),
        instructions,
    ])
