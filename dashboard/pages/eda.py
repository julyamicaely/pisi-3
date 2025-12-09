"""
Dashboard Interativo - Initial EDA
Análise Exploratória de Dados Inicial
"""

import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import chi2_contingency
from pathlib import Path
import sys

# Importar estilos
sys.path.append(str(Path(__file__).parent.parent))
from styles import PALETTE, SHADOWS, GRADIENTS

# Registrar página
dash.register_page(
    __name__,
    path="/eda",
    name="EDA inicial",
    icon="bar-chart-line"
)

# ====== Template (evita bug de template em algumas versões) ======
pio.templates.default = "plotly_white"

# ====== Carregamento e preparação dos dados ======
try:
    df = pd.read_parquet('../EDA/cardio_data.parquet')
except FileNotFoundError:
    try:
        df = pd.read_parquet('EDA/cardio_data.parquet')
    except FileNotFoundError:
        print("Erro: cardio_data.parquet não encontrado.")
        df = pd.DataFrame()

if not df.empty:
    # Gênero legível
    if 'gender' in df.columns:
        df['gender_label'] = df['gender'].map({1: 'Feminino', 2: 'Masculino'}).astype('category')
    else:
        df['gender_label'] = 'Desconhecido'

    # Variáveis binárias (0/1)
    binary_cols = ['smoke', 'alco', 'active', 'cardio']
    for c in binary_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').clip(0, 1)

    # Rótulos de colesterol e glicose
    lvl_map = {1: 'Normal', 2: 'Acima do normal', 3: 'Muito acima do normal'}
    df['chol_label'] = pd.to_numeric(df.get('cholesterol'), errors='coerce').map(lvl_map)
    df['gluc_label']  = pd.to_numeric(df.get('gluc'),        errors='coerce').map(lvl_map)

# ====== Função Cramér's V ======
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k-1, r-1))))

# ====== Componentes de UI Auxiliares ======
def make_metric_card(title, value, icon, color_key="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.H6(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                    html.H3(value, className=f"text-{color_key} mb-0", style={"fontWeight": "700"}),
                ]),
                html.Div([
                    html.I(className=f"bi {icon}", style={"fontSize": "2rem", "color": PALETTE.get(color_key, "#000"), "opacity": "0.2"})
                ])
            ], className="d-flex justify-content-between align-items-center")
        ])
    ], className="shadow-sm border-0 h-100")

def check_sanity(df):
    results = []
    def _num(s): return pd.to_numeric(df[s], errors='coerce') if s in df.columns else pd.Series(dtype='float64')
    def _count(cond): return int(pd.Series(cond).sum())

    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        ap_hi = _num('ap_hi'); ap_lo = _num('ap_lo')
        results.append({"check": "Pressão Sistólica < Diastólica", "count": _count(ap_hi < ap_lo)})
        results.append({"check": "Pressão Sistólica fora [80, 250]", "count": _count((ap_hi < 80) | (ap_hi > 250))})
        results.append({"check": "Pressão Diastólica fora [40, 150]", "count": _count((ap_lo < 40) | (ap_lo > 150))})
    
    if 'height' in df.columns:
        h = _num('height')
        results.append({"check": "Altura fora [120, 220] cm", "count": _count((h < 120) | (h > 220))})
    
    if 'weight' in df.columns:
        w = _num('weight')
        results.append({"check": "Peso fora [30, 200] kg", "count": _count((w < 30) | (w > 200))})
        
    return results

# ====== Layout ======
def layout():
    if df.empty:
        return dbc.Container(html.H3("Erro ao carregar dados.", className="text-danger text-center mt-5"))

    # Métricas
    n_rows = f"{df.shape[0]:,}".replace(",", ".")
    n_cols = df.shape[1]
    n_dups = f"{df.duplicated().sum():,}".replace(",", ".")
    n_missing = f"{df.isnull().sum().sum():,}".replace(",", ".")
    
    # Sanity Checks
    sanity_results = check_sanity(df)
    sanity_list_items = []
    for item in sanity_results:
        color = "success" if item['count'] == 0 else "warning"
        icon = "bi-check-circle-fill" if item['count'] == 0 else "bi-exclamation-triangle-fill"
        sanity_list_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Span([html.I(className=f"bi {icon} me-2 text-{color}"), item['check']]),
                    dbc.Badge(f"{item['count']} violações", color=color, pill=True, className="ms-auto")
                ], className="d-flex justify-content-between align-items-center")
            ], className="border-0 bg-light mb-1 rounded")
        )

    # Tabela
    table_df = df.head(100).copy()
    if 'id' in table_df.columns: table_df = table_df.drop(columns=['id'])
    
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([html.I(className="bi bi-bar-chart-line me-2"), "Análise Exploratória Inicial"], 
                       className="fw-bold text-primary mb-1"),
                html.P("Visão geral da qualidade e distribuição dos dados cardiovasculares.", className="text-muted")
            ])
        ], className="mb-4 mt-3"),

        # Métricas Principais
        dbc.Row([
            dbc.Col(make_metric_card("Total de Registros", n_rows, "bi-table", "primary"), md=3, className="mb-3"),
            dbc.Col(make_metric_card("Total de Colunas", n_cols, "bi-layout-three-columns", "info"), md=3, className="mb-3"),
            dbc.Col(make_metric_card("Linhas Duplicadas", n_dups, "bi-files", "warn" if int(n_dups.replace(".","")) > 0 else "success"), md=3, className="mb-3"),
            dbc.Col(make_metric_card("Valores Ausentes", n_missing, "bi-question-square", "danger" if int(n_missing.replace(".","")) > 0 else "success"), md=3, className="mb-3"),
        ], className="mb-2"),

        # Qualidade e Amostra
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Checagens de Sanidade (Regras de Negócio)", className="bg-white fw-bold border-bottom-0 pt-3"),
                    dbc.CardBody([
                        dbc.ListGroup(sanity_list_items, flush=True)
                    ])
                ], className="shadow-sm border-0 h-100")
            ], lg=5, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Amostra de Dados (Top 100)", className="bg-white fw-bold border-bottom-0 pt-3"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            data=table_df.to_dict('records'),
                            columns=[{"name": i, "id": i} for i in table_df.columns],
                            page_size=6,
                            style_table={'overflowX': 'auto'},
                            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'border': 'none'},
                            style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px', 'border': 'none', 'borderBottom': '1px solid #eee'},
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fcfcfc'}]
                        )
                    ], style={"overflow": "hidden"})
                ], className="shadow-sm border-0 h-100")
            ], lg=7, className="mb-4"),
        ]),

        # Gráficos - Linha 1
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Distribuição de Variáveis Binárias", className="card-title fw-bold mb-3"),
                        dcc.RadioItems(
                            id='metric-radio-stacked',
                            options=[
                                {'label': ' Fumante', 'value': 'smoke'},
                                {'label': ' Álcool', 'value': 'alco'},
                                {'label': ' Ativo', 'value': 'active'},
                                {'label': ' Cardiopatia', 'value': 'cardio'},
                            ],
                            value='cardio',
                            inline=True,
                            className="mb-2 small",
                            inputStyle={"marginRight": "4px", "marginLeft": "10px"}
                        ),
                        dcc.Graph(id='graph-stacked-binary', style={'height': '320px'})
                    ])
                ], className="shadow-sm border-0 h-100")
            ], md=6, className="mb-4"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Análise por Faixa Etária", className="card-title fw-bold mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.RadioItems(
                                id='metric-radio-age',
                                options=[{'label': ' Fumante', 'value': 'smoke'}, {'label': ' Cardiopatia', 'value': 'cardio'}],
                                value='cardio', inline=True, className="small"
                            ), width=7),
                            dbc.Col(dcc.Dropdown(
                                id='status-dd-age',
                                options=[{'label': 'Positivos', 'value': 1}, {'label': 'Negativos', 'value': 0}],
                                value=1, clearable=False, className="small"
                            ), width=5)
                        ], className="mb-2 align-items-center"),
                        dcc.Graph(id='graph-by-age', style={'height': '320px'})
                    ])
                ], className="shadow-sm border-0 h-100")
            ], md=6, className="mb-4"),
        ]),

        # Gráficos - Linha 2
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Distribuições Categóricas", className="card-title fw-bold mb-3"),
                        dcc.RadioItems(
                            id='cat-radio',
                            options=[{'label': ' PA', 'value': 'bp'}, {'label': ' Colesterol', 'value': 'chol'}, {'label': ' Glicose', 'value': 'gluc'}],
                            value='bp', inline=True, className="mb-2 small",
                            inputStyle={"marginRight": "4px", "marginLeft": "10px"}
                        ),
                        dcc.Graph(id='graph-categorical-counts', style={'height': '320px'})
                    ])
                ], className="shadow-sm border-0 h-100")
            ], md=6, className="mb-4"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Categorias por Idade", className="card-title fw-bold mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.RadioItems(
                                id='cat-var-age',
                                options=[{'label': ' PA', 'value': 'bp'}, {'label': ' Colesterol', 'value': 'chol'}],
                                value='chol', inline=True, className="small"
                            ), width=6),
                            dbc.Col(dcc.Dropdown(id='cat-value-age', placeholder='Selecione...', className="small"), width=6)
                        ], className="mb-2 align-items-center"),
                        dcc.Graph(id='graph-cat-by-age', style={'height': '320px'})
                    ])
                ], className="shadow-sm border-0 h-100")
            ], md=6, className="mb-4"),
        ]),

        # Correlações
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Correlações (Cramér's V)", className="card-title fw-bold mb-3"),
                        dcc.Checklist(
                            id='cramers-vars',
                            options=[{'label': f' {c}', 'value': c} for c in ['gender_label','smoke','alco','active','cardio','chol_label','gluc_label']],
                            value=['gender_label','smoke','alco','active','cardio','chol_label','gluc_label'],
                            inline=True, className="mb-3 small"
                        ),
                        dcc.Graph(id='cramers-heatmap', style={'height': '400px'})
                    ])
                ], className="shadow-sm border-0 mb-4")
            ])
        ]),

    ], fluid=True, className="py-3")


# ====== Cores e ordem comuns ======
COLOR_MAP = {'Masculino': '#93c5fd', 'Feminino': '#d8b4fe'}
GENDER_ORDER = ['Feminino', 'Masculino']

# ====== Callbacks ======

@callback(Output('graph-stacked-binary', 'figure'), Input('metric-radio-stacked', 'value'))
def update_graph_stacked(metric):
    if df.empty: return px.bar(title="Sem dados")
    tmp = df[['gender_label', metric]].dropna().astype({metric: int})
    g = tmp.groupby(['gender_label', metric]).size().reset_index(name='count')
    
    # Completar categorias faltantes
    full_idx = pd.MultiIndex.from_product([GENDER_ORDER, [0, 1]], names=['gender_label', metric])
    g = g.set_index(['gender_label', metric]).reindex(full_idx, fill_value=0).reset_index()
    
    g['status'] = g[metric].map({1: 'Positivo', 0: 'Negativo'})
    totals = g.groupby('status')['count'].transform('sum')
    g['perc'] = g['count'] / totals
    
    fig = px.bar(g, x='status', y='count', color='gender_label', text='count',
                 color_discrete_map=COLOR_MAP, category_orders={'gender_label': GENDER_ORDER},
                 labels={'status': '', 'count': 'Qtd'}, custom_data=['gender_label', 'perc'])
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None))
    fig.update_traces(textposition='auto', hovertemplate='%{customdata[0]}: %{y} (%{customdata[1]:.1%})')
    return fig

@callback(Output('graph-by-age', 'figure'), [Input('metric-radio-age', 'value'), Input('status-dd-age', 'value')])
def update_graph_by_age(metric, status_val):
    if df.empty: return px.line(title="Sem dados")
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    
    tmp = df[['age_years', 'gender_label', metric]].dropna()
    tmp['age_bin'] = pd.cut(tmp['age_years'], bins=age_bins, labels=age_labels)
    tmp['is_target'] = (tmp[metric] == int(status_val)).astype(int)
    
    agg = tmp.groupby(['age_bin', 'gender_label'], observed=False)['is_target'].mean().reset_index()
    
    fig = px.line(agg, x='age_bin', y='is_target', color='gender_label', markers=True,
                  color_discrete_map=COLOR_MAP, category_orders={'gender_label': GENDER_ORDER},
                  labels={'age_bin': 'Idade', 'is_target': 'Proporção'})
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                      yaxis=dict(tickformat=".0%"))
    return fig

@callback(Output('graph-categorical-counts', 'figure'), Input('cat-radio', 'value'))
def update_cat_counts(var):
    if df.empty: return px.bar(title="Sem dados")
    col_map = {'chol': 'chol_label', 'gluc': 'gluc_label', 'bp': 'bp_category_encoded'}
    col = col_map.get(var)
    
    if col not in df.columns: return px.bar(title="Coluna não encontrada")
    
    tmp = df[['gender_label', col]].dropna()
    # Simplificar labels se for BP
    if var == 'bp':
        tmp[col] = tmp[col].astype(str).replace({'1': 'Normal', '2': 'Elevada', '3': 'Hipert. 1', '4': 'Hipert. 2', '5': 'Crise'})
        
    g = tmp.groupby([col, 'gender_label']).size().reset_index(name='count')
    
    fig = px.bar(g, x=col, y='count', color='gender_label', text='count',
                 color_discrete_map=COLOR_MAP, category_orders={'gender_label': GENDER_ORDER},
                 labels={col: '', 'count': 'Qtd'})
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None))
    return fig

@callback([Output('cat-value-age', 'options'), Output('cat-value-age', 'value')], Input('cat-var-age', 'value'))
def update_cat_options(var):
    if df.empty: return [], None
    col = 'chol_label' if var == 'chol' else 'bp_category_encoded'
    vals = sorted(df[col].dropna().unique().astype(str))
    return [{'label': v, 'value': v} for v in vals], vals[0] if vals else None

@callback(Output('graph-cat-by-age', 'figure'), [Input('cat-var-age', 'value'), Input('cat-value-age', 'value')])
def update_cat_age(var, val):
    if df.empty or not val: return px.line(title="Sem dados")
    col = 'chol_label' if var == 'chol' else 'bp_category_encoded'
    
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    
    tmp = df[['age_years', 'gender_label', col]].dropna()
    tmp['age_bin'] = pd.cut(tmp['age_years'], bins=age_bins, labels=age_labels)
    tmp['is_target'] = (tmp[col].astype(str) == str(val)).astype(int)
    
    agg = tmp.groupby(['age_bin', 'gender_label'], observed=False)['is_target'].mean().reset_index()
    
    fig = px.line(agg, x='age_bin', y='is_target', color='gender_label', markers=True,
                  color_discrete_map=COLOR_MAP, category_orders={'gender_label': GENDER_ORDER},
                  labels={'age_bin': 'Idade', 'is_target': 'Proporção'})
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                      yaxis=dict(tickformat=".0%"))
    return fig

@callback(Output('cramers-heatmap', 'figure'), Input('cramers-vars', 'value'))
def update_heatmap(cols):
    if df.empty or len(cols) < 2: return px.imshow([[0]], title="Selecione variáveis")
    
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i <= j:
                v = cramers_v(df[c1], df[c2])
                mat.loc[c1, c2] = v
                mat.loc[c2, c1] = v
                
    fig = px.imshow(mat.astype(float), text_auto='.2f', color_continuous_scale='Blues', zmin=0, zmax=1)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig
