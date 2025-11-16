"""
Página de análise de clusterização (K-Means).
Exibe distribuição, características e perfis dos 6 clusters.
"""

import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler

# Definir tema padrão para os gráficos
pio.templates.default = "plotly_white"

# Adicionar paths para imports
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from components.utils import make_page_header, make_card, make_tabs, build_metric_grid
    print("Componentes customizados de 'utils' carregados com sucesso.")
    
except ImportError:
    print("Aviso: Componentes customizados não encontrados. Usando componentes padrão.")
    
    # --- Funções padrão para fallback ---
    def make_page_header(title, subtitle, icon=""):
        return dbc.Container([
            html.H1(title, className="display-4"),
            html.P(subtitle, className="lead"),
        ], fluid=True, className="py-3 mb-3 bg-light")
    
    def make_card(title, content, icon=""):
        return dbc.Card([
            dbc.CardHeader(f"{icon} {title}" if icon else title, className="fw-bold"),
            dbc.CardBody(content)
        ], className="mb-3")
    
    def make_tabs(tabs_list):
        tabs = []
        for i, tab in enumerate(tabs_list):
            tabs.append(dbc.Tab(tab["content"], label=tab["label"], tab_id=f"tab-{i}"))
        return dbc.Tabs(tabs, id="tabs")

    def build_metric_grid(metrics, cols=4):
        if not metrics:
            return html.Div("Nenhuma métrica para exibir.")
        
        col_size = 12 // cols if 12 % cols == 0 else 4 
        
        cards = []
        for metric in metrics:
            label = metric.get("label", "N/A")
            value = metric.get("value", "N/A")
            format_fn = metric.get("format_fn", str)
            try:
                display_value = format_fn(value)
            except Exception:
                display_value = str(value)
                
            card = dbc.Card(
                dbc.CardBody([
                    html.P(label, className="card-text text-muted", style={"fontSize": "0.9rem", "marginBottom": "0.25rem"}),
                    html.H4(display_value, className="card-title"),
                ]),
                className="text-center shadow-sm",
                style={"height": "100%"}
            )
            cards.append(dbc.Col(card, width=12, lg=col_size, className="mb-3"))
            
        return dbc.Row(cards, className="g-3")

# --- FUNÇÃO-AUXILIAR ---
def create_styled_table(df):
    style_max = {"backgroundColor": "#f8d7da", "color": "#721c24", "fontWeight": "bold"}
    style_min = {"backgroundColor": "#d4edda", "color": "#155724", "fontWeight": "bold"}

    numeric_cols = df.select_dtypes(include='number').columns
    if TRADUCOES['cluster'] in numeric_cols:
        numeric_cols = numeric_cols.drop(TRADUCOES['cluster'])
    
    try:
        col_max = df[numeric_cols].max()
        col_min = df[numeric_cols].min()
    except Exception as e:
        print(f"Aviso ao calcular min/max para estilo: {e}")
        col_max = pd.Series()
        col_min = pd.Series()

    header = html.Thead(html.Tr([html.Th(col) for col in df.columns]))

    body_rows = []
    for index, row in df.iterrows():
        cells = []
        for col_name in df.columns:
            value = row[col_name]
            style = {} 
            
            if col_name in numeric_cols:
                try:
                    if np.isclose(value, col_max[col_name]):
                        style = style_max
                    elif np.isclose(value, col_min[col_name]):
                        style = style_min
                except KeyError:
                    pass
                         
            if col_name == TRADUCOES['cluster']:
                display_value = f"{int(value)}" 
            elif isinstance(value, float):
                display_value = f"{value:.2f}" 
            else:
                display_value = value 
            
            cells.append(html.Td(display_value, style=style))
        
        body_rows.append(html.Tr(cells))
        
    body = html.Tbody(body_rows)

    return dbc.Table(
        [header, body],
        striped=True, bordered=True, hover=True, responsive=True, size="sm"
    )

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
DATA_FILE = CLUSTER_DIR / "cardio_data_processed_with_clusters.parquet"

DASHBOARD_DIR = Path(__file__).parent.parent
GRAPHICS_DIR = DASHBOARD_DIR / "assets"

EVALUATION_IMAGES = [
    "elbow_plot_v2.png",
    "silhouette_summary.png",
    "silhouette_k06.png",
    "silhouette_k07.png",
    "silhouette_k08.png",
    "davies_bouldin_summary.png",
]

BOXPLOT_COLS = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight']

ALL_ATTRIBUTES = [
    "age_years", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

PERSONA_INTERPRETATIONS = {
    "Cluster 1 (Risco Alto)": "Hipertensão Severa (83.4% Risco): Pressão arterial disparada (150/93) com BMI moderado. Recomenda-se atenção médica imediata.",
    "Cluster 5 (Risco Médio-Alto)": "Obesidade Severa (65.7% Risco): Definido pelo BMI extremo (37.0) e os piores indicadores de atividade física.",
    "Cluster 2 (Risco Médio)": "Risco pela Idade (49.3% Risco): O grupo mais velho (59 anos)...",
    "Cluster 4 (Risco Médio-Baixo)": "Risco Comportamental (36.6% Risco): 'Os Fumantes' e 'Consumidores de Álcool'.",
    "Cluster 0 (Risco Médio-Baixo)": "Risco Moderado (34.6%): Os mais Jovens, poucos bebem ou fumam.",
    "Cluster 3 (Risco Baixo)": "Grupo Saudável (19.2% Risco): Menores valores de BMI e Pressão arterial enquando possuem os maiores indicadores de atividade física.",
}

TRADUCOES = {
    'cluster': 'Cluster',
    'age_years': 'Idade (anos)',
    'bmi': 'IMC (Índice de Massa Corporal)',
    'ap_hi': 'Pressão Sistólica (Alta)',
    'ap_lo': 'Pressão Diastólica (Baixa)',
    'height': 'Altura (cm)',
    'weight': 'Peso (kg)',
    'gender': 'Género',
    'cholesterol': 'Colesterol',
    'gluc': 'Glicose',
    'smoke': 'Fumante (%)',
    'alco': 'Álcool (%)',
    'active': 'Ativo (%)',
    'Taxa de Risco (%)': 'Taxa de Risco (%)',
    'Atributo': 'Atributo',
    'Percentual': 'Percentual (%)',
    'count': 'Contagem'
}

GENDER_OPTIONS = [
    {'label': 'Todos', 'value': 0},
    {'label': 'Feminino', 'value': 1},
    {'label': 'Masculino', 'value': 2},
]


# --- Carregar dados globalmente para o callback ---
try:
    df_global = pd.read_parquet(DATA_FILE)
except Exception as e:
    df_global = None
    print(f"Erro global ao carregar dados: {e}")
    
# ================== FUNÇÕES DE CARREGAMENTO ==================
def load_data_and_artifacts():
    artifacts = {
        "df": None,
        "profile_numeric": None,
        "profile_lifestyle": None,
        "validation": None,
        "eval_images": [],
        "error": None,
    }
    
    if df_global is None:
        artifacts["error"] = f"Arquivo principal não encontrado: {DATA_FILE}."
    else:
        try:
            artifacts["df"] = df_global
            df = df_global 
            
            # Gerar tabelas de profiling (sempre com TODOS os dados)
            numeric_cols = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight']
            artifacts["profile_numeric"] = df.groupby('cluster')[numeric_cols].mean().reset_index().rename(columns=TRADUCOES)

            lifestyle_cols = ['smoke', 'alco', 'active']
            artifacts["profile_lifestyle"] = df.groupby('cluster')[lifestyle_cols].mean().mul(100).reset_index().rename(columns=TRADUCOES)
            
            validation = df.groupby('cluster')['cardio'].mean().mul(100).sort_values(ascending=False)
            artifacts["validation"] = validation.reset_index(name="Taxa de Risco (%)").rename(columns=TRADUCOES)
            
        except Exception as e:
            artifacts["error"] = f"Erro ao processar dados: {e}"

    # Carregar imagens de avaliação
    try:
        for img_name in EVALUATION_IMAGES:
            img_path = GRAPHICS_DIR / img_name
            
            if img_path.exists():
                artifacts["eval_images"].append({
                    "name": img_name.replace(".png", "").replace("_", " ").title(),
                    "src": f"/assets/{img_name}"
                })
            else:
                print(f"Aviso: Imagem de avaliação não encontrada em {img_path}")
    
    except Exception as e:
        artifacts["error"] = f"Erro ao carregar imagens: {e}"
    
    return artifacts


# ================== LAYOUT ==================
def layout():

    artifacts = load_data_and_artifacts()
    
    page_header = make_page_header(
        "Análise de Clusterização (K=6)",
        "Segmentação de pacientes com base em perfis de risco (K-Means)",
        icon="diagram-3-fill"
    )
    
    if artifacts["df"] is None:
        error_msg = artifacts["error"] or "Dados não encontrados."
        return html.Div([
            page_header,
            dbc.Alert(
                f"⚠️ Erro ao carregar dados: {error_msg}.",
                color="danger",
                className="m-4"
            )
        ])
    
    # ================== Seção de Métricas ==================
    metrics_section = html.Div([
        build_metric_grid([
            {"label": "Número de Clusters (K)", "value": 6},
            {"label": "Total de Pacientes", "value": len(artifacts["df"]), "format_fn": lambda x: f"{x:,.0f}"},
            {"label": "Cluster de Maior Risco", "value": "Cluster 1"},
            {"label": "Risco (Cluster 1)", "value": f"{artifacts['validation'].iloc[0, 1]:.1f}%"},
        ], cols=4),
    ])
    
    # ================== Seção de Profiling (Tabelas) ==================
    profile_numeric_df = artifacts["profile_numeric"]
    profile_lifestyle_df = artifacts["profile_lifestyle"]
    validation_df = artifacts["validation"].sort_values(by=TRADUCOES['cluster'])

    profiling_tables = make_card(
        "Perfis dos Clusters (Tabelas)",
        make_tabs([
            {
                "label": "Perfil Numérico",
                "content": create_styled_table(profile_numeric_df),
            },
            {
                "label": "Estilo de Vida",
                "content": create_styled_table(profile_lifestyle_df),
            },
            {
                "label": "Validação",
                "content": create_styled_table(validation_df),
            },
        ]),
        icon="table"
    )
    
    # ================== Seção: Gráficos Dinâmicos ==================
    
    # --- Conteúdo do Boxplot (Dinâmico) ---
    boxplot_tab_content = dbc.CardBody([
        html.P("Selecione uma característica para visualizar:", className="mb-2"),
        dcc.Dropdown(
            id='cluster-boxplot-dropdown',
            options=[{'label': TRADUCOES.get(col, col), 'value': col} for col in BOXPLOT_COLS],
            value=BOXPLOT_COLS[0], 
            clearable=False,
            className="mb-3"
        ),
        dcc.Graph(id='cluster-boxplot-graph') # Placeholder
    ])
    
    # --- Conteúdo dos gráficos dinâmicos ---
    dynamic_graphs_content = html.Div([
        # Filtro de Género
        dbc.Card(dbc.CardBody([
            html.H6("Filtrar por Género:", className="card-title"),
            dbc.RadioItems(
                id="gender-filter",
                options=GENDER_OPTIONS,
                value=0, # Default = Todos
                inline=True,
                label_checked_style={"fontWeight": "bold"},
            ),
        ]), className="mb-3"),
        
        # Abas com placeholders
        make_tabs([
            {
                "label": "Distribuição Normalizada (%)",
                "content": dcc.Graph(id='cluster-dist-norm-graph'), 
            },
            {
                "label": "Heatmap",
                "content": dcc.Graph(id='cluster-heatmap-graph'),
            },
            {
                "label": "Gráfico de Radar",
                "content": dcc.Graph(id='cluster-radar-graph'),
            },
            {
                "label": "Características (Box Plot)",
                "content": boxplot_tab_content, 
            },
        ])
    ])

    dynamic_graphs = make_card(
        "Análise Visual dos Clusters",
        dynamic_graphs_content,
        icon="bar-chart-line-fill"
    )

    # ================== Interpretação ==================
    interpretation_list = []
    for title, description in PERSONA_INTERPRETATIONS.items():
        interpretation_list.append(html.Li([
            html.Strong(f"{title}: "),
            description
        ]))

    interpretation_card = make_card(
        "Interpretação das 6 'Personas'",
        html.Ul(interpretation_list, className="mb-0"),
        icon="lightbulb"
    )
    
    # ================== Avaliação (Estilizado) ==================
    INACTIVE_TAB_STYLE = {
        "backgroundColor": "#f8f9fa", "color": "#555",
        "border": "1px solid #ddd", "border-bottom": "1px solid #ddd",
        "padding": "10px 15px", "border-radius": "4px 4px 0 0",
    }
    ACTIVE_TAB_STYLE = {
        "backgroundColor": "#0d6efd", "color": "white",
        "border": "1px solid #0d6efd", "padding": "10px 15px",
        "border-radius": "4px 4px 0 0", "border-bottom": "none",
    }
    TAB_CONTENT_CARD_STYLE = {"border-top": "none", "border-top-left-radius": "0"}

    if artifacts["eval_images"]:
        tabs_list = []
        for img in artifacts["eval_images"]:
            tab_content = dbc.Card(
                dbc.CardBody(
                    html.Img(src=img["src"], 
                             style={"width": "100%", "maxHeight": "600px", "objectFit": "contain"})
                ),
                style=TAB_CONTENT_CARD_STYLE
            )
            
            tabs_list.append(
                dbc.Tab(
                    tab_content, 
                    label=img["name"],
                    label_style=INACTIVE_TAB_STYLE,
                    active_label_style=ACTIVE_TAB_STYLE
                )
            )
        
        tabs_component = dbc.Tabs(
            tabs_list, 
            className="mt-2",
            style={"border-bottom": "1px solid #0d6efd"}
        )
        
        evaluation_card = make_card(
            "Justificativa da Escolha de K=6",
            tabs_component,
            icon="check-circle-fill"
        )
    else:
        evaluation_card = dbc.Alert("Gráficos de avaliação não encontrados. Verifique a pasta 'dashboard/assets/'.", color="warning")


    # ================== Layout Final ==================
    return html.Div([
        page_header,
        dbc.Alert("Análise de K=6 carregada com sucesso.", color="success"),
        metrics_section,
        html.Hr(),
        evaluation_card,
        html.Hr(),
        profiling_tables,
        html.Hr(),
        interpretation_card,
        html.Hr(),
        dynamic_graphs,
        html.Hr(),
        
    ], className="mb-5")


# ================== CALLBACKS ==================

def filter_dataframe(df, gender_value):
    """Filtra o DataFrame global com base no valor do género."""
    if df is None:
        return pd.DataFrame() 
    if gender_value == 0: # 0 = Todos
        return df
    return df[df['gender'] == gender_value]

@callback(
    Output('cluster-boxplot-graph', 'figure'),
    [Input('cluster-boxplot-dropdown', 'value'),
     Input('gender-filter', 'value')]
)
def update_boxplot(selected_characteristic, selected_gender):
    if df_global is None:
        return px.bar(title="Dados não encontrados. Impossível gerar gráfico.")

    if not selected_characteristic:
        selected_characteristic = BOXPLOT_COLS[0]
        
    translated_label = TRADUCOES.get(selected_characteristic, selected_characteristic.replace('_', ' ').title())
    
    df_filtered = filter_dataframe(df_global, selected_gender)
    
    if df_filtered.empty or df_filtered['cluster'].nunique() == 0:
         return px.bar(title=f"Sem dados para o filtro selecionado.")

    fig = px.box(
        df_filtered, 
        x="cluster",
        y=selected_characteristic,
        color="cluster",
        title=f"Comparação de: {translated_label}",
        points=False,
        labels=TRADUCOES 
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Cluster",
        yaxis_title=translated_label
    )
    
    return fig

@callback(
    Output('cluster-dist-norm-graph', 'figure'),
    Input('gender-filter', 'value')
)
def update_dist_norm_graph(selected_gender):
    if df_global is None:
        return px.bar(title="Dados não encontrados.")
        
    df_filtered = filter_dataframe(df_global, selected_gender)
    
    if df_filtered.empty or df_filtered['cluster'].nunique() == 0:
         return px.bar(title=f"Sem dados para o filtro selecionado.")
         
    attrs = ALL_ATTRIBUTES.copy()
    attrs.remove('gender') 
        
    attr_data = df_filtered.groupby("cluster")[attrs].mean()
    
    all_clusters = pd.Index(range(6), name='cluster')
    attr_data = attr_data.reindex(all_clusters).fillna(0) 
    
    attr_normalized = attr_data.div(attr_data.sum(axis=0), axis=1) * 100
    attr_normalized = attr_normalized.reset_index()
    attr_long = attr_normalized.melt(
        id_vars="cluster", var_name="Atributo", value_name="Percentual"
    )
    
    attr_long['Atributo'] = attr_long['Atributo'].map(TRADUCOES)
    attr_long['Atributo'] = attr_long['Atributo'].str.replace(" (%)", "", regex=False)
    
    translated_attrs = [TRADUCOES.get(attr, attr).replace(" (%)", "") for attr in attrs]
    category_order = { "Atributo": translated_attrs[::-1] }

    fig_bar_multi = px.bar(
        attr_long,
        x="Percentual", y="Atributo", color="cluster", orientation="h",
        barmode="group", title="Distribuição Normalizada (%) dos Atributos por Cluster",
        labels=TRADUCOES, 
        category_orders=category_order,
    )
    fig_bar_multi.update_traces(
        texttemplate="%{x:.1f}%", textposition="inside", 
        insidetextanchor="middle", textfont_size=12
    )
    fig_bar_multi.update_layout(
        legend_title_text="Cluster", xaxis_title="Percentual (%)", yaxis_title="Atributo",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        bargap=0.15
    )
    return fig_bar_multi

# --- MODIFICADO: Callback para o Heatmap ---
@callback(
    Output('cluster-heatmap-graph', 'figure'),
    Input('gender-filter', 'value')
)
def update_heatmap(selected_gender):
    if df_global is None:
        return px.bar(title="Dados não encontrados.")

    df_filtered = filter_dataframe(df_global, selected_gender)
    
    if df_filtered.empty or df_filtered['cluster'].nunique() == 0:
         return px.bar(title=f"Sem dados para o filtro selecionado.")
         
    attrs = ALL_ATTRIBUTES.copy()
    if selected_gender != 0: 
        attrs.remove('gender') 
    
    profile_data = df_filtered.groupby('cluster')[attrs].mean()
    
    # --- MODIFICAÇÃO: A LÓGICA .reindex() FOI REMOVIDA DESTE CALLBACK ---
    # O Heatmap agora só mostrará os clusters que existem no filtro.
    
    scaler_heatmap = MinMaxScaler()
    # Adicionar verificação para evitar erro se profile_data tiver < 2 linhas
    if profile_data.shape[0] < 2:
        return px.bar(title=f"Não há dados suficientes (clusters) para o Heatmap neste filtro.")
        
    profile_heatmap_data = scaler_heatmap.fit_transform(profile_data)
    profile_heatmap_df = pd.DataFrame(profile_heatmap_data, columns=attrs, index=profile_data.index) # O índice agora é dinâmico
    
    df_heatmap = profile_heatmap_df.rename(columns=TRADUCOES).rename(columns=lambda x: x.replace(" (%)", ""))
    
    fig_heatmap = px.imshow(
        df_heatmap,
        text_auto=".2f",
        aspect="auto",
        title="Heatmap Normalizado (Min-Max) por Atributo",
        labels=dict(color="Nível (0-1)", y="Cluster", x="Atributo"),
        color_continuous_scale='RdYlGn_r'
    )
    fig_heatmap.update_xaxes(side="top", tickangle=45)
    
    return fig_heatmap

# --- Callback para o Gráfico de Radar (Mantém a lógica .reindex) ---
@callback(
    Output('cluster-radar-graph', 'figure'),
    Input('gender-filter', 'value')
)
def update_radar(selected_gender):
    if df_global is None:
        return px.bar(title="Dados não encontrados.")
        
    df_filtered = filter_dataframe(df_global, selected_gender)
    
    if df_filtered.empty or df_filtered['cluster'].nunique() == 0:
         return px.bar(title=f"Sem dados para o filtro selecionado.")
         
    attrs = ALL_ATTRIBUTES.copy()
    if selected_gender != 0: 
        attrs.remove('gender') 
    
    scaler_radar = MinMaxScaler()
    radar_data = df_filtered[attrs].copy()
    
    if radar_data.shape[0] < 2:
        return px.bar(title="Não há dados suficientes para este filtro.")
        
    radar_data_scaled = scaler_radar.fit_transform(radar_data)
    radar_df = pd.DataFrame(radar_data_scaled, columns=attrs)
    radar_df['cluster'] = df_filtered['cluster'].values
    
    radar_grouped = radar_df.groupby('cluster').mean()
    
    # A lógica .reindex() é MANTIDA aqui para o Radar funcionar corretamente
    all_clusters = pd.Index(range(6), name='cluster')
    radar_grouped = radar_grouped.reindex(all_clusters).fillna(0).reset_index() 
    
    radar_long = radar_grouped.melt(id_vars='cluster', var_name='Atributo', value_name='Valor')
    
    radar_long['Atributo'] = radar_long['Atributo'].map(TRADUCOES).str.replace(" (%)", "")
    
    fig_radar = px.line_polar(
        radar_long,
        r='Valor',
        theta='Atributo',
        color='cluster',
        line_close=True,
        title="Gráfico de Radar: Perfil Médio dos Clusters (Normalizado Min-Max)",
        labels={'Valor': 'Nível (0-1)', 'cluster': 'Cluster'}
    )
    fig_radar.update_traces(fill='toself', opacity=0.7)
    
    return fig_radar