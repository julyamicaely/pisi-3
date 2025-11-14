"""
Página de análise de clusterização (K-Means).
Exibe distribuição de clusters, visualizações e características dos grupos.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
import os
from pathlib import Path
import pandas as pd

# Adicionar paths para imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.utils import (
    make_page_header, make_card, build_metric_grid,
    make_tabs, make_alert, make_empty_state
)
from components.cards import build_pie_chart, build_box_plot
from styles import PALETTE, SPACING

# Registrar página
dash.register_page(
    __name__,
    path="/clusterizacao",
    name="Clusterização",
    title="Clusterização - Dashboard",
    icon="diagram-3-fill",
)

# ================== CAMINHOS DOS ARTEFATOS ==================
BASE_DIR = Path(__file__).parent.parent.parent
CLUSTER_DIR = BASE_DIR / "clusterization"


# ================== FUNÇÕES DE CARREGAMENTO ==================
def load_cluster_artifacts():
    """Carrega artefatos da clusterização."""
    artifacts = {
        "summary": None,
        "report": None,
        "images": [],
        "error": None,
    }
    
    try:
        # Carregar sumário dos clusters
        summary_path = CLUSTER_DIR / "cluster_2_summary.csv"
        if summary_path.exists():
            artifacts["summary"] = pd.read_csv(summary_path)
        
        # Carregar relatório
        report_path = CLUSTER_DIR / "cluster_report.txt"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                artifacts["report"] = f.read()
        
        # Listar imagens disponíveis
        image_files = [
            "cluster_distribution.png",
            "cluster_comparison.png",
            "cluster_boxplots.png",
            "cluster_pca.png",
            "Cotovelo x Silhouete.png",
        ]
        
        for img in image_files:
            img_path = CLUSTER_DIR / img
            if img_path.exists():
                artifacts["images"].append(img)
    
    except Exception as e:
        artifacts["error"] = str(e)
    
    return artifacts


# ================== LAYOUT DA PÁGINA ==================
def layout():
    """Cria layout da página de Clusterização."""
    
    # Carregar artefatos
    artifacts = load_cluster_artifacts()
    
    # Header da página
    page_header = make_page_header(
        "Análise de Clusterização",
        "Segmentação de pacientes usando K-Means (aprendizado não supervisionado)",
        icon="diagram-3-fill"
    )
    
    # Alertas
    alerts = []
    if artifacts["error"]:
        alerts.append(make_alert(
            f"⚠️ Erro ao carregar artefatos: {artifacts['error']}",
            color="warning"
        ))
    
    if artifacts["summary"] is None:
        alerts.append(make_alert(
            "ℹ️ Execute o pipeline de clusterização primeiro: python clusterization/n2_clusters.py",
            color="info"
        ))
    
    # Métricas dos clusters (se houver sumário)
    if artifacts["summary"] is not None:
        # Calcular métricas agregadas
        n_clusters = len(artifacts["summary"])
        total_samples = artifacts["summary"].get("count", artifacts["summary"].iloc[:, 0]).sum() if "count" in artifacts["summary"].columns else "N/A"
        
        metrics_section = html.Div([
            html.H4("Visão Geral", className="mb-3"),
            build_metric_grid([
                {"label": "Número de Clusters", "value": n_clusters, "format_fn": str},
                {"label": "Total de Amostras", "value": total_samples, "format_fn": lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x)},
                {"label": "Algoritmo", "value": "K-Means", "format_fn": str},
                {"label": "Método de Inicialização", "value": "k-means++", "format_fn": str},
            ], cols=4),
        ])
        
        # Tabela de sumário dos clusters
        summary_table = make_card(
            "Características dos Clusters",
            dbc.Table.from_dataframe(
                artifacts["summary"],
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size="sm",
            ),
            icon="table"
        )
    else:
        metrics_section = make_empty_state("Execute a clusterização para ver métricas")
        summary_table = html.Div()
    
    # Visualizações dos clusters
    if artifacts["images"]:
        viz_tabs = []
        
        for img in artifacts["images"]:
            viz_tabs.append({
                "label": img.replace("_", " ").replace(".png", "").title(),
                "content": html.Img(
                    src=f"/assets/../clusterization/{img}",
                    style={"width": "100%", "maxHeight": "600px", "objectFit": "contain"}
                ),
            })
        
        visualizations = make_card(
            "Visualizações",
            make_tabs(viz_tabs),
            icon="bar-chart-line-fill"
        )
    else:
        visualizations = make_empty_state("Nenhuma visualização disponível")
    
    # Relatório textual
    if artifacts["report"]:
        report_card = make_card(
            "Relatório de Clusterização",
            html.Pre(
                artifacts["report"],
                style={
                    "backgroundColor": "#F5F7FA",
                    "padding": SPACING["md"],
                    "borderRadius": "4px",
                    "fontSize": "13px",
                    "fontFamily": "'Fira Code', monospace",
                    "overflow": "auto",
                    "maxHeight": "400px",
                }
            ),
            icon="file-text"
        )
    else:
        report_card = make_empty_state("Nenhum relatório disponível")
    
    # Interpretação dos clusters
    interpretation = make_card(
        "Interpretação dos Resultados",
        html.Div([
            html.H6("Como interpretar:", className="mb-3"),
            html.Ul([
                html.Li([
                    html.Strong("Distribuição de Clusters: "),
                    "Mostra quantos pacientes pertencem a cada grupo."
                ]),
                html.Li([
                    html.Strong("Comparação de Features: "),
                    "Identifica características que diferenciam os clusters."
                ]),
                html.Li([
                    html.Strong("PCA: "),
                    "Visualização 2D da separação entre clusters."
                ]),
                html.Li([
                    html.Strong("Cotovelo e Silhouette: "),
                    "Ajudam a determinar o número ideal de clusters."
                ]),
            ]),
            html.Hr(),
            html.H6("Aplicações práticas:", className="mb-2 mt-3"),
            html.Ul([
                html.Li("Identificar perfis de risco cardiovascular"),
                html.Li("Personalizar estratégias de prevenção por grupo"),
                html.Li("Descobrir padrões ocultos nos dados"),
            ]),
        ]),
        icon="lightbulb"
    )
    
    # Instruções
    instructions = make_card(
        "Como executar a clusterização",
        html.Ol([
            html.Li("Navegue até clusterization/"),
            html.Li("Execute: python n_clusters_find.py (para encontrar k ótimo)"),
            html.Li("Execute: python n2_clusters.py (para gerar clusters e visualizações)"),
            html.Li("Atualize esta página para ver os resultados"),
        ]),
        icon="play-circle"
    )
    
    # Layout completo
    return html.Div([
        page_header,
        *alerts,
        metrics_section,
        html.Hr(style={"margin": f"{SPACING['lg']} 0"}),
        summary_table,
        html.Hr(style={"margin": f"{SPACING['lg']} 0"}),
        visualizations,
        html.Hr(style={"margin": f"{SPACING['lg']} 0"}),
        dbc.Row([
            dbc.Col(report_card, width=12, lg=7),
            dbc.Col(interpretation, width=12, lg=5),
        ]),
        html.Hr(style={"margin": f"{SPACING['lg']} 0"}),
        instructions,
    ])
