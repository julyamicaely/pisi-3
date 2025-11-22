"""
Página de análise de clusterização (K-Means).
Exibe distribuição, características e perfis dos clusters.
"""

import dash
from dash import html, dcc, Input, Output, callback, State
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
    from styles import PALETTE
    print("✅ Componentes customizados carregados com sucesso.")
    
except ImportError:
    print("⚠️ Aviso: Componentes customizados não encontrados. Usando componentes padrão.")
    
    # Palette padrão para fallback
    PALETTE = {
        'gradient_start': '#667eea',
        'gradient_end': '#764ba2',
        'accent': '#f093fb',
        'dark': '#2d3748',
        'light': '#f7fafc'
    }
    
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
    if 'Cluster' in numeric_cols:
        numeric_cols = numeric_cols.drop('Cluster')
    
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
                         
            if col_name == 'Cluster':
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

EVALUATION_IMAGES_K6 = [
    "elbow_plot_v2.png",
    "silhouette_summary.png",
    "silhouette_k06.png",
    "silhouette_k07.png",
    "silhouette_k08.png",
    "davies_bouldin_summary.png",
]

EVALUATION_IMAGES_K16 = [
    "elbow_plot_v2.png",
    "silhouette_summary.png",
    "silhouette_k16.png",
    "davies_bouldin_summary.png",
]

BOXPLOT_COLS = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight']

ALL_ATTRIBUTES = [
    "age_years", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
]

PERSONA_INTERPRETATIONS_K6 = {
    "Cluster 1 (Risco Alto)": "Hipertensão Severa (83.4% Risco): Pressão arterial disparada (150/93) com BMI moderado. Recomenda-se atenção médica imediata.",
    "Cluster 5 (Risco Médio-Alto)": "Obesidade Severa (65.7% Risco): Definido pelo BMI extremo (37.0) e os piores indicadores de atividade física.",
    "Cluster 2 (Risco Médio)": "Risco pela Idade (49.3% Risco): O grupo mais velho (59 anos)...",
    "Cluster 4 (Risco Médio-Baixo)": "Risco Comportamental (36.6% Risco): 'Os Fumantes' e 'Consumidores de Álcool'.",
    "Cluster 0 (Risco Médio-Baixo)": "Risco Moderado (34.6%): Os mais Jovens, poucos bebem ou fumam.",
    "Cluster 3 (Risco Baixo)": "Grupo Saudável (19.2% Risco): Menores valores de BMI e Pressão arterial enquando possuem os maiores indicadores de atividade física.",
}

PERSONA_INTERPRETATIONS_K16 = {
    "Cluster 6 (Risco Muito Alto - 86.0%)": "Hipertensão Crítica: Pressão arterial extremamente elevada (164/101) com IMC moderado (30.0). Grupo com maior risco cardiovascular, necessitando intervenção médica urgente.",
    "Cluster 1 (Risco Muito Alto - 81.1%)": "Obesidade Hipertensa: IMC muito alto (34.9) combinado com hipertensão (144/90). Perfil de alto risco metabólico.",
    "Cluster 5 (Risco Alto - 80.6%)": "Hipertensão Jovem com Maus Hábitos: Pacientes relativamente jovens (45 anos) com hipertensão (142/91) e maiores taxas de tabagismo (13.3%) e álcool (8.1%).",
    "Cluster 12 (Risco Alto - 79.9%)": "Hipertensão com Baixos Fatores Comportamentais: Pressão alta (142/89) mas com muito baixos índices de fumo (1.6%) e álcool (2.1%). Risco provavelmente relacionado a fatores não-comportamentais.",
    "Cluster 8 (Risco Alto - 78.1%)": "Hipertensos com Estilo de Vida de Risco: Pressão alta (142/89) combinada com altas taxas de tabagismo (23.0%) e consumo de álcool (11.9%).",
    "Cluster 11 (Risco Médio-Alto - 69.6%)": "Pré-Hipertensão: Pressão moderadamente elevada (126/80) com IMC normal. Grupo que pode se beneficiar de intervenções preventivas.",
    "Cluster 4 (Risco Médio-Alto - 69.5%)": "Obesidade Mórbida: IMC extremamente alto (45.0) - o maior entre todos os clusters. Altura média baixa (158cm) com peso muito elevado (113kg).",
    "Cluster 2 (Risco Médio - 53.0%)": "Sobrepeso com Pressão Normal: IMC alto (33.7) mas pressão arterial dentro dos limites (122/78). Risco moderado principalmente pela obesidade.",
    "Cluster 9 (Risco Médio-Baixo - 45.0%)": "Idosos com Estilo de Vida Moderado: Grupo mais velho (60 anos) com pressão normal. Taxas moderadas de tabagismo (16.8%) e álcool (7.4%).",
    "Cluster 0 (Risco Médio-Baixo - 44.5%)": "Meia-Idade com Perfil Moderado: Idade média (55 anos) com IMC levemente elevado (26.8) e pressão normal. Perfil intermediário.",
    "Cluster 7 (Risco Baixo - 40.6%)": "Saúde Comportamental Exemplar: Maior cluster (9092 pacientes) com baixíssimos índices de fumo (1.3%) e álcool (1.5%). Pressão e IMC normais.",
    "Cluster 13 (Risco Baixo - 34.3%)": "Jovens com Sobrepeso: Idade jovem (46 anos) com IMC alto (33.5) mas pressão normal. Potencial para intervenção precoce.",
    "Cluster 10 (Risco Baixo - 33.7%)": "Tabagistas Ativos: Altas taxas de tabagismo (23.5%) e álcool (10.3%) mas com parâmetros clínicos normais. Grupo que se beneficiaria de cessação do tabaco.",
    "Cluster 14 (Risco Muito Baixo - 25.3%)": "Jovens Saudáveis: Segundo maior cluster (6268 pacientes). Idade jovem (46 anos) com IMC normal, pressão normal e excelentes hábitos comportamentais.",
    "Cluster 15 (Risco Muito Baixo - 20.9%)": "Jovens com Hábitos de Risco mas Parâmetros Normais: Grupo mais jovem (43 anos) com altas taxas de tabagismo (24.5%) e álcool (10.8%), mas com todos os parâmetros clínicos normais.",
    "Cluster 3 (Risco Mínimo - 17.2%)": "Perfil de Saúde Ideal: Pressão arterial mais baixa entre todos (105/66), IMC normal e hábitos comportamentais saudáveis. Menor risco cardiovascular."
}

TRADUCOES = {
    'clusterk6': 'Cluster K=6',
    'clusterk16': 'Cluster K=16',
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
    'cardio': 'Doença Cardiovascular (%)',
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

K_OPTIONS = [
    {'label': 'K=6 Clusters', 'value': 6},
    {'label': 'K=16 Clusters', 'value': 16},
]

# --- Carregar dados globalmente para o callback ---
try:
    df_global = pd.read_parquet(DATA_FILE)
    print(f"✅ Dados carregados com sucesso! Colunas disponíveis: {list(df_global.columns)}")
except Exception as e:
    df_global = None
    print(f"Erro global ao carregar dados: {e}")
    
# ================== FUNÇÕES DE CARREGAMENTO ==================
def load_data_and_artifacts(k_value):
    artifacts = {
        "df": None,
        "profile_numeric": None,
        "profile_lifestyle": None,
        "validation": None,
        "eval_images": [],
        "persona_interpretations": {},
        "error": None,
    }
    
    if df_global is None:
        artifacts["error"] = f"Arquivo principal não encontrado: {DATA_FILE}."
        return artifacts
    
    try:
        artifacts["df"] = df_global
        df = df_global 
        
        # Determinar a coluna de cluster baseada no K selecionado
        cluster_col = f'clusterk{k_value}'
        
        if cluster_col not in df.columns:
            artifacts["error"] = f"Coluna de cluster '{cluster_col}' não encontrada no dataset."
            return artifacts
        
        # Gerar tabelas de profiling
        numeric_cols = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight']
        profile_numeric = df.groupby(cluster_col)[numeric_cols].mean().reset_index()
        profile_numeric = profile_numeric.rename(columns={cluster_col: 'Cluster', **TRADUCOES})
        artifacts["profile_numeric"] = profile_numeric

        # REMOVIDO 'cardio' do lifestyle_cols - agora só tem smoke, alco, active
        lifestyle_cols = ['smoke', 'alco', 'active']  # Removido 'cardio'
        profile_lifestyle = df.groupby(cluster_col)[lifestyle_cols].mean().mul(100).reset_index()
        profile_lifestyle = profile_lifestyle.rename(columns={cluster_col: 'Cluster', **TRADUCOES})
        artifacts["profile_lifestyle"] = profile_lifestyle
        
        validation = df.groupby(cluster_col)['cardio'].mean().mul(100).sort_values(ascending=False)
        validation_df = validation.reset_index(name="Taxa de Risco (%)")
        validation_df = validation_df.rename(columns={cluster_col: 'Cluster'})
        artifacts["validation"] = validation_df
        
        # Carregar imagens de avaliação baseadas no K
        eval_images = EVALUATION_IMAGES_K6 if k_value == 6 else EVALUATION_IMAGES_K16
        for img_name in eval_images:
            img_path = GRAPHICS_DIR / img_name
            if img_path.exists():
                artifacts["eval_images"].append({
                    "name": img_name.replace(".png", "").replace("_", " ").title(),
                    "src": f"/assets/{img_name}"
                })
            else:
                print(f"Aviso: Imagem de avaliação não encontrada em {img_path}")
        
        # Carregar interpretações baseadas no K
        artifacts["persona_interpretations"] = PERSONA_INTERPRETATIONS_K6 if k_value == 6 else PERSONA_INTERPRETATIONS_K16
            
    except Exception as e:
        artifacts["error"] = f"Erro ao processar dados: {e}"
    
    return artifacts


# ================== LAYOUT ==================
def layout():
    # Não carregar dados no layout inicial - isso será feito pelos callbacks
    return html.Div([
        # ================== Seletor de K ==================
        dbc.Card([
            dbc.CardHeader("Configuração da Análise", className="fw-bold"),
            dbc.CardBody([
                html.Div([
                    html.Label("Selecione o número de clusters (K):", className="fw-bold me-3"),
                    dbc.RadioItems(
                        id="k-selector",
                        options=K_OPTIONS,
                        value=6,
                        inline=True,
                        label_checked_style={"fontWeight": "bold"},
                    ),
                ], className="d-flex align-items-center")
            ])
        ], className="mb-4"),
        
        html.Hr(),
        
        # ================== Seção de Métricas (placeholder) ==================
        html.Div(id="metrics-section"),
        
        html.Hr(),
        
        # ================== Seção de Avaliação (placeholder) ==================
        html.Div(id="evaluation-section"),
        
        html.Hr(),
        
        # ================== Seção de Profiling (placeholder) ==================
        html.Div(id="profiling-section"),
        
        html.Hr(),
        
        # ================== Seção de Gráficos Dinâmicos ==================
        # Conteúdo fixo para gráficos dinâmicos (Solução 4)
        html.Div([
            # Filtro de Género
            dbc.Card(dbc.CardBody([
                html.H6("Filtrar por Género:", className="card-title"),
                dbc.RadioItems(
                    id="gender-filter",
                    options=GENDER_OPTIONS,
                    value=0,
                    inline=True,
                    label_checked_style={"fontWeight": "bold"},
                ),
            ]), className="mb-3"),
            
            # Abas com conteúdo pré-carregado (Solução 4)
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id='cluster-dist-norm-graph'), 
                    label="Distribuição Normalizada (%)",
                    tab_id="tab-0"
                ),
                dbc.Tab(
                    dbc.CardBody([
                        html.P("Selecione uma característica para visualizar:", className="mb-2"),
                        dcc.Dropdown(
                            id='cluster-boxplot-dropdown',
                            options=[{'label': TRADUCOES.get(col, col), 'value': col} for col in BOXPLOT_COLS],
                            value=BOXPLOT_COLS[0], 
                            clearable=False,
                            className="mb-3"
                        ),
                        dcc.Graph(id='cluster-boxplot-graph')
                    ]),
                    label="Características (Box Plot)", 
                    tab_id="tab-1"
                ),
                dbc.Tab(
                    dcc.Graph(id='cluster-heatmap-graph'),
                    label="Heatmap",
                    tab_id="tab-2"
                ),
            ], id="tabs", active_tab="tab-0")
        ], id="dynamic-graphs-content"),
        
        html.Hr(),
        
        # ================== Seção de Interpretação (placeholder) ==================
        html.Div(id="interpretation-section"),
        
        html.Hr(),
    ], className="mb-5")


# ================== FUNÇÕES AUXILIARES PARA GRÁFICOS ==================

def filter_dataframe(df, gender_value, cluster_col):
    """Filtra o DataFrame global com base no valor do género."""
    if df is None:
        return pd.DataFrame() 
    if gender_value == 0: # 0 = Todos
        return df
    return df[df['gender'] == gender_value]

def create_dist_norm_graph(selected_gender, k_value):
    """Cria gráfico de distribuição normalizada"""
    if df_global is None:
        return px.bar(title="Dados não encontrados.")
    
    cluster_col = f'clusterk{k_value}'
    df_filtered = filter_dataframe(df_global, selected_gender, cluster_col)
    
    if df_filtered.empty or df_filtered[cluster_col].nunique() == 0:
        return px.bar(title=f"Sem dados para o filtro selecionado.")
        
    attrs = ALL_ATTRIBUTES.copy()
    attrs.remove('gender') 
        
    attr_data = df_filtered.groupby(cluster_col)[attrs].mean()
    
    n_clusters = 6 if k_value == 6 else 16
    all_clusters = pd.Index(range(n_clusters), name=cluster_col)
    attr_data = attr_data.reindex(all_clusters).fillna(0) 
    
    attr_normalized = attr_data.div(attr_data.sum(axis=0), axis=1) * 100
    attr_normalized = attr_normalized.reset_index()
    attr_long = attr_normalized.melt(
        id_vars=cluster_col, var_name="Atributo", value_name="Percentual"
    )
    
    attr_long['Atributo'] = attr_long['Atributo'].map(TRADUCOES)
    attr_long['Atributo'] = attr_long['Atributo'].str.replace(" (%)", "", regex=False)
    
    translated_attrs = [TRADUCOES.get(attr, attr).replace(" (%)", "") for attr in attrs]
    category_order = { "Atributo": translated_attrs[::-1] }

    fig_bar_multi = px.bar(
        attr_long,
        x="Percentual", y="Atributo", color=cluster_col, orientation="h",
        barmode="group", title=f"Distribuição Normalizada (%) dos Atributos por Cluster (K={k_value})",
        labels=TRADUCOES, 
        category_orders=category_order,
    )
    fig_bar_multi.update_traces(
        texttemplate="%{x:.1f}%", textposition="inside", 
        insidetextanchor="middle", textfont_size=12
    )
    fig_bar_multi.update_layout(
        legend_title_text=f"Cluster K={k_value}", 
        xaxis_title="Percentual (%)", 
        yaxis_title="Atributo",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        bargap=0.15
    )
    return fig_bar_multi

def create_heatmap(selected_gender, k_value):
    """Cria gráfico de heatmap"""
    if df_global is None:
        return px.bar(title="Dados não encontrados.")

    cluster_col = f'clusterk{k_value}'
    df_filtered = filter_dataframe(df_global, selected_gender, cluster_col)
    
    if df_filtered.empty or df_filtered[cluster_col].nunique() == 0:
        return px.bar(title=f"Sem dados para o filtro selecionado.")
        
    attrs = ALL_ATTRIBUTES.copy()
    if selected_gender != 0: 
        attrs.remove('gender') 
    
    profile_data = df_filtered.groupby(cluster_col)[attrs].mean()
    
    scaler_heatmap = MinMaxScaler()
    if profile_data.shape[0] < 2:
        return px.bar(title=f"Não há dados suficientes para o Heatmap neste filtro.")
        
    profile_heatmap_data = scaler_heatmap.fit_transform(profile_data)
    
    # Transpor a matriz para ter atributos no eixo Y e clusters no eixo X
    profile_heatmap_data_transposed = profile_heatmap_data.T
    
    # Criar DataFrame com atributos como índice e clusters como colunas
    profile_heatmap_df = pd.DataFrame(
        profile_heatmap_data_transposed, 
        index=attrs,
        columns=profile_data.index
    )
    
    # Traduzir os nomes dos atributos
    profile_heatmap_df.index = profile_heatmap_df.index.map(TRADUCOES)
    profile_heatmap_df.index = profile_heatmap_df.index.str.replace(" (%)", "")
    
    fig_heatmap = px.imshow(
        profile_heatmap_df,
        text_auto=".2f",
        aspect="auto",
        title=f"Heatmap Normalizado (Min-Max) por Atributo (K={k_value})",
        labels=dict(x=f"Cluster K={k_value}", y="Atributo", color="Nível (0-1)"),
        color_continuous_scale='RdYlGn_r'
    )
    
    fig_heatmap.update_layout(
        yaxis=dict(tickangle=0),
        xaxis=dict(side="top")
    )
    
    return fig_heatmap

def create_boxplot(selected_characteristic, selected_gender, k_value):
    """Cria gráfico de boxplot"""
    if df_global is None:
        return px.bar(title="Dados não encontrados.")

    if not selected_characteristic:
        selected_characteristic = BOXPLOT_COLS[0]
        
    translated_label = TRADUCOES.get(selected_characteristic, selected_characteristic.replace('_', ' ').title())
    
    cluster_col = f'clusterk{k_value}'
    df_filtered = filter_dataframe(df_global, selected_gender, cluster_col)
    
    if df_filtered.empty or df_filtered[cluster_col].nunique() == 0:
        return px.bar(title=f"Sem dados para o filtro selecionado.")

    # Usar uma cor única (azul) para todas as caixas
    fig = px.box(
        df_filtered, 
        x=cluster_col,
        y=selected_characteristic,
        color_discrete_sequence=['#1f77b4'],
        title=f"Comparação de: {translated_label} (K={k_value})",
        points=False,
        labels=TRADUCOES 
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title=f"Cluster K={k_value}",
        yaxis_title=translated_label
    )
    
    return fig

# ================== CALLBACKS ==================

# Callback para atualizar todas as visualizações simultaneamente (Solução 5)
@callback(
    [Output('cluster-dist-norm-graph', 'figure'),
     Output('cluster-heatmap-graph', 'figure'),
     Output('cluster-boxplot-graph', 'figure')],
    [Input('k-selector', 'value'),
     Input('gender-filter', 'value')],
    [State('cluster-boxplot-dropdown', 'value')]
)
def update_all_visualizations(k_value, gender_value, selected_char):
    """Atualiza todas as visualizações quando K ou género mudam"""
    dist_fig = create_dist_norm_graph(gender_value, k_value)
    heatmap_fig = create_heatmap(gender_value, k_value)
    boxplot_fig = create_boxplot(selected_char, gender_value, k_value)
    
    return dist_fig, heatmap_fig, boxplot_fig

# Callback para atualizar o título da seção de gráficos dinâmicos
@callback(
    Output('dynamic-graphs-content', 'children'),
    Input('k-selector', 'value')
)
def update_dynamic_graphs_title(k_value):
    dynamic_graphs_content = html.Div([
        # Filtro de Género
        dbc.Card(dbc.CardBody([
            html.H6("Filtrar por Género:", className="card-title"),
            dbc.RadioItems(
                id="gender-filter",
                options=GENDER_OPTIONS,
                value=0,
                inline=True,
                label_checked_style={"fontWeight": "bold"},
            ),
        ]), className="mb-3"),
        
        # Abas com conteúdo pré-carregado (Solução 4)
        dbc.Tabs([
            dbc.Tab(
                dcc.Graph(id='cluster-dist-norm-graph'), 
                label="Distribuição Normalizada (%)",
                tab_id="tab-0"
            ),
            dbc.Tab(
                dbc.CardBody([
                    html.P("Selecione uma característica para visualizar:", className="mb-2"),
                    dcc.Dropdown(
                        id='cluster-boxplot-dropdown',
                        options=[{'label': TRADUCOES.get(col, col), 'value': col} for col in BOXPLOT_COLS],
                        value=BOXPLOT_COLS[0], 
                        clearable=False,
                        className="mb-3"
                    ),
                    dcc.Graph(id='cluster-boxplot-graph')
                ]),
                label="Características (Box Plot)", 
                tab_id="tab-1"
            ),
            dbc.Tab(
                dcc.Graph(id='cluster-heatmap-graph'),
                label="Heatmap",
                tab_id="tab-2"
            ),
        ], id="tabs", active_tab="tab-0")
    ])

    return make_card(
        f"Análise Visual dos Clusters K={k_value}",
        dynamic_graphs_content,
        icon="bar-chart-line-fill"
    )

# Callback principal para atualizar as seções baseado no K selecionado
@callback(
    [Output('metrics-section', 'children'),
     Output('evaluation-section', 'children'),
     Output('profiling-section', 'children'),
     Output('interpretation-section', 'children')],
    Input('k-selector', 'value')
)
def update_static_sections(k_value):
    artifacts = load_data_and_artifacts(k_value)
    
    if artifacts["error"]:
        error_alert = dbc.Alert(f"Erro: {artifacts['error']}", color="danger")
        return error_alert, error_alert, error_alert, error_alert
    
    df = artifacts["df"]
    cluster_col = f'clusterk{k_value}'
    
    # ================== Métricas ==================
    validation_sorted = artifacts["validation"].sort_values(by="Taxa de Risco (%)", ascending=False)
    
    maior_risco_cluster = int(validation_sorted.iloc[0]['Cluster'])
    maior_risco_valor = validation_sorted.iloc[0]["Taxa de Risco (%)"]
    
    menor_risco_cluster = int(validation_sorted.iloc[-1]['Cluster'])
    menor_risco_valor = validation_sorted.iloc[-1]["Taxa de Risco (%)"]
    
    idade_media = df['age_years'].mean()
    pacientes_risco_alto = (df['cardio'] == 1).sum()
    percentual_risco = (pacientes_risco_alto / len(df)) * 100
    
    metrics_content = html.Div([
        html.Div([
            html.Div([
                html.H1(f"🔬 Análise de Clusterização (K={k_value})", 
                       className="display-4 fw-bold text-white mb-3"),
                html.P("Segmentação de pacientes com base em perfis de risco cardiovascular", 
                      className="lead text-white-50 mb-4"),
                
                # MÉTRICAS PRINCIPAIS
                dbc.Row([
                    # Métrica 1: Total de Clusters
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-diagram-3-fill", 
                                      style={"fontSize": "28px", "color": "white", "marginBottom": "8px"}),
                                html.P("Clusters (K)", className="text-white-50 mb-1", 
                                      style={"fontSize": "13px"}),
                                html.H2(f"{k_value}", 
                                       className="text-white mb-0 fw-bold"),
                            ], className="text-center p-3", 
                               style={"backgroundColor": "rgba(255,255,255,0.15)", 
                                      "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"})
                        ])
                    ], md=2),
                    
                    # Métrica 2: Total de Pacientes
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-people-fill", 
                                      style={"fontSize": "28px", "color": "white", "marginBottom": "8px"}),
                                html.P("Total de Pacientes", className="text-white-50 mb-1", 
                                      style={"fontSize": "13px"}),
                                html.H2(f"{len(df):,}", 
                                       className="text-white mb-0 fw-bold"),
                            ], className="text-center p-3", 
                               style={"backgroundColor": "rgba(255,255,255,0.15)", 
                                      "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"})
                        ])
                    ], md=2),
                    
                    # Métrica 3: Idade Média
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-calendar-heart", 
                                      style={"fontSize": "28px", "color": "white", "marginBottom": "8px"}),
                                html.P("Idade Média", className="text-white-50 mb-1", 
                                      style={"fontSize": "13px"}),
                                html.H2(f"{idade_media:.0f}", 
                                       className="text-white mb-0 fw-bold"),
                            ], className="text-center p-3", 
                               style={"backgroundColor": "rgba(255,255,255,0.15)", 
                                      "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"})
                        ])
                    ], md=2),
                    
                    # Métrica 4: Maior Risco
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-exclamation-triangle-fill", 
                                      style={"fontSize": "28px", "color": "#ff6b6b", "marginBottom": "8px"}),
                                html.P(f"Maior Risco (Cluster {maior_risco_cluster})", className="text-white-50 mb-1", 
                                      style={"fontSize": "13px"}),
                                html.H2(f"{maior_risco_valor:.1f}%", 
                                       className="text-white mb-0 fw-bold"),
                            ], className="text-center p-3", 
                               style={"backgroundColor": "rgba(255,107,107,0.2)", 
                                      "borderRadius": "12px", "border": "2px solid rgba(255,107,107,0.4)"})
                        ])
                    ], md=2),
                    
                    # Métrica 5: Menor Risco
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-shield-check", 
                                      style={"fontSize": "28px", "color": "#51cf66", "marginBottom": "8px"}),
                                html.P(f"Menor Risco (Cluster {menor_risco_cluster})", className="text-white-50 mb-1", 
                                      style={"fontSize": "13px"}),
                                html.H2(f"{menor_risco_valor:.1f}%", 
                                       className="text-white mb-0 fw-bold"),
                            ], className="text-center p-3", 
                               style={"backgroundColor": "rgba(81,207,102,0.2)", 
                                      "borderRadius": "12px", "border": "2px solid rgba(81,207,102,0.4)"})
                        ])
                    ], md=2),
                    
                    # Métrica 6: % com Doença
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-heart-pulse-fill", 
                                      style={"fontSize": "28px", "color": "white", "marginBottom": "8px"}),
                                html.P("Com Doença Cardio.", className="text-white-50 mb-1", 
                                      style={"fontSize": "13px"}),
                                html.H2(f"{percentual_risco:.1f}%", 
                                       className="text-white mb-0 fw-bold"),
                            ], className="text-center p-3", 
                               style={"backgroundColor": "rgba(255,255,255,0.15)", 
                                      "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"})
                        ])
                    ], md=2),
                ], className="mt-4")
            ], className="container py-5")
        ], style={
            "background": f"linear-gradient(135deg, {PALETTE['gradient_start']} 0%, {PALETTE['gradient_end']} 100%)",
            "marginBottom": "40px",
            "borderRadius": "0 0 30px 30px",
            "boxShadow": "0 10px 40px rgba(0,0,0,0.2)"
        })
    ])
    
    # ================== Avaliação - Sistema de Abas para Ambos K ==================
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
        
        # Para K=16, ativar a primeira aba por padrão; para K=6, primeira aba também fica ativa
        tabs_component = dbc.Tabs(
            tabs_list, 
            className="mt-2",
            style={"border-bottom": "1px solid #0d6efd"}
        )
        
        evaluation_content = make_card(
            f"Justificativa da Escolha de K={k_value}",
            tabs_component,
            icon="check-circle-fill"
        )
    else:
        evaluation_content = dbc.Alert("Gráficos de avaliação não encontrados. Verifique a pasta 'dashboard/assets/'.", color="warning")
    
    # ================== Profiling Tables - Sistema de Abas para Ambos K ==================
    profile_numeric_df = artifacts["profile_numeric"]
    profile_lifestyle_df = artifacts["profile_lifestyle"]
    validation_df = artifacts["validation"].sort_values(by='Cluster')

    # Sistema de abas para ambos os valores de K (6 e 16)
    profiling_content = make_card(
        f"Perfis dos Clusters K={k_value} (Tabelas)",
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
    
    # ================== Interpretação ==================
    interpretation_list = []
    for title, description in artifacts["persona_interpretations"].items():
        interpretation_list.append(html.Li([
            html.Strong(f"{title}: "),
            description
        ]))

    interpretation_content = make_card(
        f"Interpretação dos {k_value} Clusters",
        html.Ul(interpretation_list, className="mb-0"),
        icon="lightbulb"
    ) if artifacts["persona_interpretations"] else html.Div()

    return (metrics_content, evaluation_content, profiling_content, interpretation_content)