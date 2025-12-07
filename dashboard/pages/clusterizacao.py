import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import warnings
import functools
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ================== CONFIGURAÇÃO CENTRALIZADA ==================

class Config:
    """Configurações centralizadas da aplicação"""
    
    # Clusters válidos
    VALID_CLUSTERS = [6, 16]
    
    # Cache settings
    CACHE_TTL = 300  # 5 minutos
    CACHE_ENABLED = True
    
    # Sistema de cores
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'danger': '#d62728',
        'info': '#9467bd',
        'warning': '#8c564b',
        'accent': '#e377c2',
        'background': '#7f7f7f',
        'muted': '#bcbd22'
    }
    
    # Paleta gradient
    GRADIENT_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
    ]
    
    # Boxplot colors
    BOXPLOT_COLOR = '#1f77b4'
    
    # Heatmap colors
    HEATMAP_COLOR_SCALE = 'RdYlGn_r'
    
    # Mensagens de erro padronizadas
    MESSAGES = {
        'data_not_found': "❌ Dados não encontrados. Verifique se o arquivo está disponível.",
        'invalid_k': "❌ Valor de K inválido. Use apenas K=6 ou K=16.",
        'cluster_not_found': "❌ Cluster não encontrado no dataset.",
        'insufficient_data': "❌ Dados insuficientes para gerar o gráfico.",
        'loading': "⏳ Carregando dados...",
        'error_processing': "❌ Erro ao processar dados: {error}"
    }

# Importar estilos do projeto principal
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from styles import PALETTE, create_plotly_template
    print("✅ Paleta de cores carregada com sucesso.")
except ImportError:
    print("⚠️ Paleta de cores não encontrada. Usando paleta padrão.")
    PALETTE = {
        'gradient_start': '#667eea',
        'gradient_end': '#764ba2', 
        'accent': '#f093fb',
        'dark': '#2d3748',
        'light': '#f7fafc',
        'primary': Config.COLORS['primary'],
        'secondary': Config.COLORS['secondary'],
        'success': '#28a745',
        'warn': '#ffc107',
        'chart_2': '#17a2b8'
    }

# Definir tema padrão para os gráficos
pio.templates.default = "plotly_white"

# ================== SISTEMA DE CACHE ==================

class DataCache:
    """Sistema de cache inteligente para melhorar performance"""
    
    def __init__(self, ttl: int = 300):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = ttl
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Verifica se o cache expirou"""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera item do cache se não expirou"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if not self._is_expired(timestamp):
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Armazena item no cache"""
        self._cache[key] = (value, datetime.now())
    
    def clear(self) -> None:
        """Limpa todo o cache"""
        self._cache.clear()
    
    def clear_expired(self) -> None:
        """Remove apenas itens expirados"""
        expired_keys = []
        for key, (value, timestamp) in self._cache.items():
            if self._is_expired(timestamp):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]

# Instância global do cache
cache = DataCache(Config.CACHE_TTL)

# ================== SISTEMA DE VALIDAÇÃO ==================

class DataValidator:
    """Validação robusta de parâmetros de entrada"""
    
    @staticmethod
    def validate_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Valida e retorna DataFrame válido"""
        if df is None:
            raise ValueError(Config.MESSAGES['data_not_found'])
        if df.empty:
            raise ValueError(f"❌ DataFrame está vazio.")
        return df
    
    @staticmethod
    def validate_cluster_column(df: pd.DataFrame, cluster_col: str) -> str:
        """Valida coluna de cluster"""
        if cluster_col not in df.columns:
            raise ValueError(f"❌ Coluna '{cluster_col}' não encontrada. Colunas disponíveis: {list(df.columns)}")
        return cluster_col
    
    @staticmethod
    def validate_k_value(k_value: int) -> int:
        """Valida valor de K"""
        if k_value not in Config.VALID_CLUSTERS:
            raise ValueError(f"❌ K={k_value} não é suportado. Use apenas K=6 ou K=16.")
        return k_value
    
    @staticmethod
    def validate_cluster_exists(df: pd.DataFrame, cluster_col: str, cluster_id: int) -> int:
        """Valida se cluster existe"""
        if cluster_id not in df[cluster_col].unique():
            raise ValueError(f"❌ Cluster {cluster_id} não encontrado no dataset.")
        return cluster_id
    
    @staticmethod
    def validate_characteristic(col: str, valid_cols: List[str]) -> str:
        """Valida característica selecionada"""
        if col not in valid_cols:
            raise ValueError(f"❌ Característica '{col}' não é válida.")
        return col
    
    @staticmethod
    def safe_numeric_value(value: Any, default: float = 0.0) -> float:
        """Conversão segura para número"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

# ================== FUNÇÕES AUXILIARES OTIMIZADAS ==================

def safe_filter_dataframe(df: pd.DataFrame, gender_value: int) -> pd.DataFrame:
    """Filtra DataFrame com validação robusta"""
    try:
        DataValidator.validate_dataframe(df)
        
        if gender_value == 0:  # Todos os gêneros
            return df
        elif gender_value in [1, 2]:  # Feminino/Masculino
            return df[df['gender'] == gender_value]
        else:
            warnings.warn(f"Gênero inválido: {gender_value}. Usando todos os dados.")
            return df
    except Exception as e:
        print(f"Erro ao filtrar DataFrame: {e}")
        return pd.DataFrame()

def cached_cluster_statistics(df: pd.DataFrame, cluster_col: str, k_value: int, gender_filter: int) -> pd.DataFrame:
    """Calcula estatísticas de cluster com cache"""
    try:
        # Gerar chave única baseada nos parâmetros
        cache_key = f"cluster_stats_{k_value}_{gender_filter}_{id(df)}"
        
        # Verificar cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Calcular estatísticas
        df_filtered = safe_filter_dataframe(df, gender_filter)
        
        stats = []
        for cluster_id in range(k_value):
            cluster_data = df_filtered[df_filtered[cluster_col] == cluster_id]
            if len(cluster_data) > 0:
                stats.append({
                    'cluster': cluster_id,
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(df_filtered)) * 100,
                    'age_mean': cluster_data['age_years'].mean() if 'age_years' in cluster_data.columns else 0,
                    'bmi_mean': cluster_data['bmi'].mean() if 'bmi' in cluster_data.columns else 0,
                    'cardio_rate': (cluster_data['cardio'] == 1).sum() / len(cluster_data) * 100 if 'cardio' in cluster_data.columns else 0
                })
        
        result = pd.DataFrame(stats)
        
        # Armazenar em cache
        if Config.CACHE_ENABLED and not result.empty:
            cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        print(f"Erro ao calcular estatísticas cacheadas: {e}")
        return pd.DataFrame()

def get_cached_profiling_data(df: pd.DataFrame, k_value: int) -> Dict[str, pd.DataFrame]:
    """Gera profiling data com cache"""
    try:
        # Gerar chave única
        cache_key = f"profiling_{k_value}_{id(df)}"
        
        # Verificar cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        cluster_col = f'clusterk{k_value}'
        
        # Calcular profiling
        numeric_cols = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight']
        
        profile_numeric = df.groupby(cluster_col)[numeric_cols].mean().reset_index()
        profile_numeric = profile_numeric.rename(columns={cluster_col: 'Cluster', **TRADUCOES})
        
        lifestyle_cols = ['smoke', 'alco', 'active']
        profile_lifestyle = df.groupby(cluster_col)[lifestyle_cols].mean().mul(100).reset_index()
        profile_lifestyle = profile_lifestyle.rename(columns={cluster_col: 'Cluster', **TRADUCOES})
        
        validation = df.groupby(cluster_col)['cardio'].mean().mul(100).sort_values(ascending=False)
        validation_df = validation.reset_index(name="Taxa de Risco (%)")
        validation_df = validation_df.rename(columns={cluster_col: 'Cluster'})
        
        result = {
            'profile_numeric': profile_numeric,
            'profile_lifestyle': profile_lifestyle,
            'validation': validation_df
        }
        
        # Armazenar em cache
        if Config.CACHE_ENABLED:
            cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        print(f"Erro ao gerar profiling cacheado: {e}")
        return {
            'profile_numeric': pd.DataFrame(),
            'profile_lifestyle': pd.DataFrame(), 
            'validation': pd.DataFrame()
        }

# ================== COMPONENTES CUSTOMIZADOS ==================

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

def create_styled_table_improved(df: pd.DataFrame, max_rows: int = 50) -> dbc.Table:
    """Cria tabela estilizada com validação robusta"""
    
    # Validação de entrada
    if df is None or df.empty:
        return dbc.Alert("Dados não disponíveis para exibir.", color="warning")
    
    # Limitar número de linhas para performance
    if len(df) > max_rows:
        df_display = df.head(max_rows).copy()
        footer_note = f"Mostrando as primeiras {max_rows} linhas de {len(df)} total."
    else:
        df_display = df.copy()
        footer_note = ""
    
    # Estilos para destacar valores extremos
    style_max = {"backgroundColor": "#f8d7da", "color": "#721c24", "fontWeight": "bold"}
    style_min = {"backgroundColor": "#d4edda", "color": "#155724", "fontWeight": "bold"}
    
    # Identificar colunas numéricas
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns.tolist()
    if 'Cluster' in numeric_cols:
        numeric_cols.remove('Cluster')
    
    # Calcular extremos com validação
    try:
        col_max = df_display[numeric_cols].max() if numeric_cols else pd.Series()
        col_min = df_display[numeric_cols].min() if numeric_cols else pd.Series()
    except Exception as e:
        print(f"Aviso ao calcular min/max para estilo: {e}")
        col_max = pd.Series()
        col_min = pd.Series()
    
    # Criar cabeçalho
    header = html.Thead(html.Tr([html.Th(col) for col in df_display.columns]))
    
    # Criar corpo da tabela
    body_rows = []
    for index, row in df_display.iterrows():
        cells = []
        for col_name in df_display.columns:
            value = row[col_name]
            style = {}
            
            # Aplicar estilo para valores extremos
            if col_name in numeric_cols and col_name in col_max.index:
                try:
                    if np.isclose(value, col_max[col_name], rtol=1e-5):
                        style = style_max
                    elif np.isclose(value, col_min[col_name], rtol=1e-5):
                        style = style_min
                except (KeyError, TypeError):
                    pass
                         
            # Formatação de valores
            if col_name == 'Cluster':
                display_value = f"{int(value)}" if pd.notna(value) else "N/A"
            elif isinstance(value, (int, float)) and pd.notna(value):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value) if pd.notna(value) else "N/A"
            
            cells.append(html.Td(display_value, style=style))
        
        body_rows.append(html.Tr(cells))
    
    body = html.Tbody(body_rows)
    
    # Criar tabela
    table = dbc.Table(
        [header, body],
        striped=True, bordered=True, hover=True, responsive=True, size="sm"
    )
    
    # Adicionar nota se dados foram truncados
    if footer_note:
        return html.Div([
            table,
            html.Small(footer_note, className="text-muted mt-2 d-block text-center")
        ])
    
    return table

# ================== CONSTANTES E CONFIGURAÇÕES ==================

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

# Configurações de imagens de avaliação
EVALUATION_IMAGES = {
    6: [
        "elbow_plot_v2.png",
        "silhouette_summary.png", 
        "silhouette_k06.png",
        "silhouette_k07.png",
        "silhouette_k08.png",
        "davies_bouldin_summary.png",
    ],
    16: [
        "elbow_plot_v2.png",
        "silhouette_summary.png",
        "silhouette_k16.png", 
        "davies_bouldin_summary.png",
    ]
}

# Colunas para diferentes análises
BOXPLOT_COLS = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight']
EDA_NUMERIC_COLS = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight', 'cholesterol', 'gluc']
EDA_BINARY_COLS = ['smoke', 'alco', 'active', 'cardio']

ALL_ATTRIBUTES = [
    "age_years", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
]

# Interpretações de personas por K
PERSONA_INTERPRETATIONS = {
    6: {
        "Cluster 1 (Risco Alto)": "Hipertensão Severa (83.4% Risco): Pressão arterial disparada (150/93) com BMI moderado. Recomenda-se atenção médica imediata.",
        "Cluster 5 (Risco Médio-Alto)": "Obesidade Severa (65.7% Risco): Definido pelo BMI extremo (37.0) e os piores indicadores de atividade física.",
        "Cluster 2 (Risco Médio)": "Risco pela Idade (49.3% Risco): O grupo mais velho (59 anos)...",
        "Cluster 4 (Risco Médio-Baixo)": "Risco Comportamental (36.6% Risco): 'Os Fumantes' e 'Consumidores de Álcool'.",
        "Cluster 0 (Risco Médio-Baixo)": "Risco Moderado (34.6%): Os mais Jovens, poucos bebem ou fumam.",
        "Cluster 3 (Risco Baixo)": "Grupo Saudável (19.2% Risco): Menores valores de BMI e Pressão arterial enquando possuem os maiores indicadores de atividade física.",
    },
    16: {
        "Cluster 6 (Risco Muito Alto - 86.0%)": "Hipertensão Crítica: Pressão arterial extremamente elevada (164/101) com IMC moderado (30.0). Grupo com maior risco cardiovascular, necessitando intervenção médica urgente.",
        "Cluster 1 (Risco Muito Alto - 81.1%)": "Obesidade Hipertensa: IMC muito alto (34.9) combinado com hipertensão (144/90). Perfil de alto risco metabólico.",
        "Cluster 5 (Risco Alto - 80.6%)": "Hipertensão Jovem com Maus Hábitos: Pacientes relativamente jovens (45 anos) com hipertensão (142/91) e maiores taxas de tabagismo (13.3%) e álcool (8.1%).",
        "Cluster 12 (Risco Alto - 79.9%)": "Hipertensão com Baixos Fatores Comportamentais: Pressão alta (142/89) mas com muito baixos índices de fumo (1.6%) e álcool (2.1%). Risco provavelmente relacionado a fatores não-comportamentais.",
        "Cluster 8 (Risco Alto - 78.1%)": "Hipertensos com Estilo de Vida de Risco: Pressão alta (142/89) combinada com altas taxas de tabagismo (23.0%) e consumo de álcool (11.9%).",
        "Cluster 11 (Risco Médio-Alto - 69.6%)": "Pré-Hipertensão: Pressão moderadamente elevada (126/80) com IMC normal. Grupo que pode se beneficiar de intervenções preventivas.",
        "Cluster 4 (Risco Médio-Alto - 69.5%)": "Obesidade Mórbida: IMC extremamente alto (45.0) - o maior entre todos os clusters. Altura média baixa (158cm) com peso muito elevado (113kg).",
        "Cluster 2 (Risco Médio - 53.0%)": "Sobrepeso com Pressão Normal: IMC alto (33.7) mas pressão arterial dentro dos limites (122/78). Risco moderado principalmente pela obesidadade.",
        "Cluster 9 (Risco Médio-Baixo - 45.0%)": "Idosos com Estilo de Vida Moderado: Grupo mais velho (60 anos) com pressão normal. Taxas moderadas de tabagismo (16.8%) e álcool (7.4%).",
        "Cluster 0 (Risco Médio-Baixo - 44.5%)": "Meia-Idade com Perfil Moderado: Idade média (55 anos) com IMC levemente elevado (26.8) e pressão normal. Perfil intermediário.",
        "Cluster 7 (Risco Baixo - 40.6%)": "Saúde Comportamental Exemplar: Maior cluster (9092 pacientes) com baixíssimos índices de fumo (1.3%) e álcool (1.5%). Pressão e IMC normais.",
        "Cluster 13 (Risco Baixo - 34.3%)": "Jovens com Sobrepeso: Idade jovem (46 anos) com IMC alto (33.5) mas pressão normal. Potencial para intervenção precoce.",
        "Cluster 10 (Risco Baixo - 33.7%)": "Tabagistas Ativos: Altas taxas de tabagismo (23.5%) e álcool (10.3%) mas com parâmetros clínicos normais. Grupo que se beneficiaria de cessação do tabaco.",
        "Cluster 14 (Risco Muito Baixo - 25.3%)": "Jovens Saudáveis: Segundo maior cluster (6268 pacientes). Idade jovem (46 anos) com IMC normal, pressão normal e excelentes hábitos comportamentais.",
        "Cluster 15 (Risco Muito Baixo - 20.9%)": "Jovens com Hábitos de Risco mas Parâmetros Normais: Grupo mais jovem (43 anos) com altas taxas de tabagismo (24.5%) e álcool (10.8%), mas com todos os parâmetros clínicos normais.",
        "Cluster 3 (Risco Mínimo - 17.2%)": "Perfil de Saúde Ideal: Pressão arterial mais baixa entre todos (105/66), IMC normal e hábitos comportamentais saudáveis. Menor risco cardiovascular."
    }
}

# Traduções
TRADUCOES = {
    'clusterk6': 'Cluster K=6',
    'clusterk16': 'Cluster K=16',
    'age_years': 'Idade (anos)',
    'bmi': 'IMC (Índice de Massa Corporal)',
    'ap_hi': 'Pressão Sistólica (Alta)',
    'ap_lo': 'Pressão Diastólica (Baixa)',
    'height': 'Altura (cm)',
    'weight': 'Peso (kg)',
    'gender': 'Gênero',
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

# Opções de filtros
GENDER_OPTIONS = [
    {'label': 'Todos', 'value': 0},
    {'label': 'Feminino', 'value': 2},
    {'label': 'Masculino', 'value': 1},
]

K_OPTIONS = [
    {'label': 'K=6 Clusters', 'value': 6},
    {'label': 'K=16 Clusters', 'value': 16},
]

# ================== CARREGAMENTO DE DADOS OTIMIZADO ==================

class DataLoader:
    """Carregador de dados com cache e validação"""
    
    def __init__(self, data_file: Path):
        self.data_file = data_file
        self._df_global = None
        self._load_timestamp = None
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """Carrega dados com cache"""
        # Verificar cache primeiro
        if Config.CACHE_ENABLED:
            cached_data = cache.get('global_data')
            if cached_data is not None:
                return cached_data
        
        # Carregar dados
        try:
            if not self.data_file.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {self.data_file}")
            
            df = pd.read_parquet(self.data_file)
            self._df_global = df
            self._load_timestamp = datetime.now()
            
            # Armazenar em cache
            if Config.CACHE_ENABLED:
                cache.set('global_data', df)
            
            print(f"✅ Dados carregados com sucesso! Colunas disponíveis: {list(df.columns)}")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return None
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Retorna dados globais (do cache ou carrega se necessário)"""
        if self._df_global is not None:
            return self._df_global
        return self.load_data()

# Instância global do carregador
data_loader = DataLoader(DATA_FILE)

# ================== FUNÇÕES DE PROCESSAMENTO OTIMIZADAS ==================

def load_data_and_artifacts_optimized(k_value: int) -> Dict[str, Any]:
    """Carrega dados e artefatos com otimizações"""
    
    artifacts = {
        "df": None,
        "profile_numeric": None,
        "profile_lifestyle": None,
        "validation": None,
        "eval_images": [],
        "persona_interpretations": {},
        "error": None,
    }
    
    # Carregar dados
    df = data_loader.get_data()
    if df is None:
        artifacts["error"] = f"Arquivo principal não encontrado: {DATA_FILE}."
        return artifacts
    
    try:
        # Validações
        cluster_col = f'clusterk{DataValidator.validate_k_value(k_value)}'
        DataValidator.validate_cluster_column(df, cluster_col)
        
        artifacts["df"] = df
        
        # Gerar profiling com cache
        profiles_data = get_cached_profiling_data(df, k_value)
        artifacts.update(profiles_data)
        
        # Carregar imagens de avaliação
        eval_images = EVALUATION_IMAGES.get(k_value, [])
        for img_name in eval_images:
            img_path = GRAPHICS_DIR / img_name
            if img_path.exists():
                artifacts["eval_images"].append({
                    "name": img_name.replace(".png", "").replace("_", " ").title(),
                    "src": f"/assets/{img_name}"
                })
            else:
                print(f"Aviso: Imagem de avaliação não encontrada em {img_path}")
        
        # Carregar interpretações
        artifacts["persona_interpretations"] = PERSONA_INTERPRETATIONS.get(k_value, {})
            
    except Exception as e:
        artifacts["error"] = f"Erro ao processar dados: {str(e)}"
    
    return artifacts

# ================== FUNÇÕES DE CRIAÇÃO DE GRÁFICOS OTIMIZADAS ==================

def create_dist_norm_graph_optimized(df: pd.DataFrame, selected_gender: int, k_value: int) -> go.Figure:
    """Cria gráfico de distribuição normalizada otimizado"""
    
    try:
        # Validações
        cluster_col = f'clusterk{DataValidator.validate_k_value(k_value)}'
        DataValidator.validate_cluster_column(df, cluster_col)
        
        # Filtrar dados
        df_filtered = safe_filter_dataframe(df, selected_gender)
        
        if df_filtered.empty:
            return go.Figure().add_annotation(
                text=Config.MESSAGES['insufficient_data'], 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Processar atributos
        attrs = ALL_ATTRIBUTES.copy()
        attrs.remove('gender') 
            
        attr_data = df_filtered.groupby(cluster_col)[attrs].mean()
        
        # Verificar dados suficientes
        if attr_data.empty or attr_data.shape[0] == 0:
            return go.Figure().add_annotation(
                text="❌ Sem dados para os filtros selecionados.", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Normalizar dados
        n_clusters = 6 if k_value == 6 else 16
        all_clusters = pd.Index(range(n_clusters), name=cluster_col)
        attr_data = attr_data.reindex(all_clusters, fill_value=0) 
        
        attr_normalized = attr_data.div(attr_data.sum(axis=0), axis=1) * 100
        attr_normalized = attr_normalized.reset_index()
        
        # Transformar para formato longo
        attr_long = attr_normalized.melt(
            id_vars=cluster_col, var_name="Atributo", value_name="Percentual"
        )
        
        # Traduções
        attr_long['Atributo'] = attr_long['Atributo'].map(TRADUCOES)
        attr_long['Atributo'] = attr_long['Atributo'].str.replace(" (%)", "", regex=False)
        
        # Criar gráfico
        fig = px.bar(
            attr_long,
            x="Percentual", y="Atributo", color=cluster_col, orientation="h",
            barmode="group", 
            title=f"Distribuição Normalizada (%) dos Atributos por Cluster (K={k_value})",
            labels=TRADUCOES,
            color_discrete_sequence=Config.GRADIENT_COLORS[:n_clusters]
        )
        
        fig.update_traces(
            texttemplate="%{x:.1f}%", textposition="inside", 
            insidetextanchor="middle", textfont_size=12
        )
        
        fig.update_layout(
            legend_title_text=f"Cluster K={k_value}", 
            xaxis_title="Percentual (%)", 
            yaxis_title="Atributo",
            xaxis=dict(range=[0, 100], ticksuffix="%"),
            bargap=0.15,
            height=600
        )
        
        return fig
        
    except Exception as e:
        print(f"Erro ao criar gráfico de distribuição: {e}")
        return go.Figure().add_annotation(
            text=f"❌ Erro ao criar gráfico: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

def create_heatmap_optimized(df: pd.DataFrame, selected_gender: int, k_value: int) -> go.Figure:
    """Cria heatmap otimizado"""
    
    try:
        # Validações
        cluster_col = f'clusterk{DataValidator.validate_k_value(k_value)}'
        DataValidator.validate_cluster_column(df, cluster_col)
        
        # Filtrar dados
        df_filtered = safe_filter_dataframe(df, selected_gender)
        
        if df_filtered.empty:
            return go.Figure().add_annotation(
                text=Config.MESSAGES['insufficient_data'], 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Processar atributos
        attrs = ALL_ATTRIBUTES.copy()
        if selected_gender != 0: 
            attrs.remove('gender') 
        
        profile_data = df_filtered.groupby(cluster_col)[attrs].mean()
        
        # Verificar dados suficientes para heatmap
        if profile_data.shape[0] < 2 or profile_data.shape[1] < 2:
            return go.Figure().add_annotation(
                text="❌ Dados insuficientes para gerar heatmap.", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Normalizar dados
        scaler_heatmap = MinMaxScaler()
        profile_heatmap_data = scaler_heatmap.fit_transform(profile_data)
        
        # Transpor matriz
        profile_heatmap_data_transposed = profile_heatmap_data.T
        
        # Criar DataFrame para heatmap
        profile_heatmap_df = pd.DataFrame(
            profile_heatmap_data_transposed, 
            index=attrs,
            columns=profile_data.index
        )
        
        # Traduzir atributos
        profile_heatmap_df.index = profile_heatmap_df.index.map(TRADUCOES)
        profile_heatmap_df.index = profile_heatmap_df.index.str.replace(" (%)", "")
        
        # Criar heatmap
        fig = px.imshow(
            profile_heatmap_df,
            text_auto=".2f",
            aspect="auto",
            title=f"Heatmap Normalizado (Min-Max) por Atributo (K={k_value})",
            labels=dict(x=f"Cluster K={k_value}", y="Atributo", color="Nível (0-1)"),
            color_continuous_scale=Config.HEATMAP_COLOR_SCALE
        )
        
        fig.update_layout(
            yaxis=dict(tickangle=0),
            xaxis=dict(side="top"),
            height=max(400, len(attrs) * 50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Erro ao criar heatmap: {e}")
        return go.Figure().add_annotation(
            text=f"❌ Erro ao criar heatmap: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

def create_cluster_comparison_visualization(df: pd.DataFrame, gender_filter: int, k_value: int, metric: str) -> go.Figure:
    """Cria comparação inter-cluster para Análise Visual dos Clusters"""
    
    try:
        cluster_col = f'clusterk{k_value}'
        
        # Calcular estatísticas por cluster
        cluster_stats = []
        
        for cluster_id in range(k_value):
            cluster_mask = df[cluster_col] == cluster_id
            
            if gender_filter == 1:  # Masculino
                cluster_mask = cluster_mask & (df['gender'] == 1)
            elif gender_filter == 2:  # Feminino
                cluster_mask = cluster_mask & (df['gender'] == 2)
            
            cluster_data = df[cluster_mask]
            
            if len(cluster_data) > 0:
                if metric == 'mean':
                    values = cluster_data[EDA_NUMERIC_COLS].mean()
                elif metric == 'median':
                    values = cluster_data[EDA_NUMERIC_COLS].median()
                elif metric == 'std':
                    values = cluster_data[EDA_NUMERIC_COLS].std()
                elif metric == 'q25':
                    values = cluster_data[EDA_NUMERIC_COLS].quantile(0.25)
                elif metric == 'q75':
                    values = cluster_data[EDA_NUMERIC_COLS].quantile(0.75)
                else:
                    values = cluster_data[EDA_NUMERIC_COLS].mean()
                
                cluster_stats.append({
                    'cluster': cluster_id,
                    'values': values,
                    'total': len(cluster_data)
                })
        
        if not cluster_stats:
            return go.Figure().add_annotation(
                text="❌ Dados insuficientes para comparação.", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Criar gráfico de barras agrupadas
        fig = go.Figure()
        
        for i, col in enumerate(EDA_NUMERIC_COLS):
            values = [stats['values'][col] if col in stats['values'] else 0 for stats in cluster_stats]
            
            fig.add_trace(go.Bar(
                x=[f'Cluster {stats["cluster"]}' for stats in cluster_stats],
                y=values,
                name=TRADUCOES.get(col, col),
                marker_color=Config.GRADIENT_COLORS[i % len(Config.GRADIENT_COLORS)],
                text=[f'{val:.1f}' for val in values],
                textposition='outside',
                textfont=dict(size=10)
            ))
        
        # Totais removidos da visualização conforme solicitado
        
        fig.update_layout(
            title=f"⚖️ Comparação Inter-Cluster - {metric.upper()} (K={k_value})",
            xaxis_title="Clusters",
            yaxis_title="Valor da Métrica",
            height=600,
            barmode='group',
            font=dict(size=11),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"❌ Erro na comparação: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

# ================== FUNÇÕES EDA OTIMIZADAS ==================

class ClusterAnalyzer:
    """Analisador de clusters com funcionalidades avançadas"""
    
    @staticmethod
    def get_cluster_info_safe(df: pd.DataFrame, cluster_col: str, cluster_id: int, gender_filter: int = 0) -> Optional[Dict[str, Any]]:
        """Obtém informações de cluster com validação robusta"""
        
        try:
            # Validações
            DataValidator.validate_dataframe(df)
            DataValidator.validate_cluster_column(df, cluster_col)
            DataValidator.validate_cluster_exists(df, cluster_col, cluster_id)
            
            # Filtrar dados
            df_filtered = safe_filter_dataframe(df, gender_filter) if gender_filter != 0 else df
            
            # Filtrar cluster
            cluster_data = df_filtered[df_filtered[cluster_col] == cluster_id]
            
            if cluster_data.empty:
                return None
            
            # Calcular estatísticas com validação
            numeric_stats = cluster_data[EDA_NUMERIC_COLS].describe() if EDA_NUMERIC_COLS else pd.DataFrame()
            binary_stats = cluster_data[EDA_BINARY_COLS].mean() * 100 if EDA_BINARY_COLS else pd.Series()
            
            # Estatísticas de idade
            age_stats = {}
            if 'age_years' in cluster_data.columns:
                age_series = cluster_data['age_years']
                age_stats = {
                    'mean': DataValidator.safe_numeric_value(age_series.mean()),
                    'median': DataValidator.safe_numeric_value(age_series.median()),
                    'std': DataValidator.safe_numeric_value(age_series.std()),
                    'min': DataValidator.safe_numeric_value(age_series.min()),
                    'max': DataValidator.safe_numeric_value(age_series.max())
                }
            
            # Informações do cluster
            info = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0,
                'numeric_stats': numeric_stats,
                'binary_stats': binary_stats,
                'age_stats': age_stats,
                'gender_dist': cluster_data['gender'].value_counts() if 'gender' in cluster_data.columns else pd.Series(),
                'cardio_rate': (cluster_data['cardio'] == 1).sum() / len(cluster_data) * 100 if 'cardio' in cluster_data.columns else 0,
                'data': cluster_data
            }
            
            return info
            
        except Exception as e:
            print(f"Erro ao obter informações do cluster: {e}")
            return None
    
    @staticmethod
    def create_correlation_heatmap(cluster_info: Optional[Dict], k_value: int, cluster_id: int) -> go.Figure:
        """Cria heatmap de correlações intra-cluster"""
        
        if cluster_info is None:
            return go.Figure().add_annotation(
                text="❌ Dados do cluster não encontrados.", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        try:
            cluster_data = cluster_info['data']
            
            # Selecionar apenas colunas numéricas
            numeric_cols = EDA_NUMERIC_COLS
            available_cols = [col for col in numeric_cols if col in cluster_data.columns]
            
            if len(available_cols) < 2:
                return go.Figure().add_annotation(
                    text="❌ Dados insuficientes para análise de correlação.", 
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
            
            # Calcular matriz de correlação
            corr_data = cluster_data[available_cols].corr()
            
            # Traduzir labels
            translated_labels = [TRADUCOES.get(col, col) for col in available_cols]
            
            # Criar heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=translated_labels,
                y=translated_labels,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(corr_data.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                colorbar=dict(title="Correlação")
            ))
            
            fig.update_layout(
                title=f"🔥 Matriz de Correlações - Cluster {cluster_id} (K={k_value})",
                height=600,
                xaxis_title="",
                yaxis_title="",
                font=dict(size=11)
            )
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"❌ Erro ao criar heatmap: {str(e)}", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
    
    @staticmethod
    def create_gender_comparison(cluster_info: Optional[Dict], k_value: int, cluster_id: int, variable: str = 'age_years') -> go.Figure:
        """Cria comparação de distribuições por género para uma variável específica"""
        
        if cluster_info is None:
            return go.Figure().add_annotation(
                text="❌ Dados do cluster não encontrados.", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        try:
            cluster_data = cluster_info['data']
            
            # Validar variável
            if variable not in cluster_data.columns:
                return go.Figure().add_annotation(
                    text=f"❌ Variável '{variable}' não encontrada nos dados.", 
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
            
            # Separar por género
            male_data = cluster_data[cluster_data['gender'] == 1][variable].dropna()
            female_data = cluster_data[cluster_data['gender'] == 2][variable].dropna()
            
            # Criar figura com histogramas sobrepostos
            fig = go.Figure()
            
            # Verificar se há dados para mostrar
            if len(male_data) == 0 and len(female_data) == 0:
                return go.Figure().add_annotation(
                    text="❌ Nenhum dado disponível para este cluster e variável.", 
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
            
            # Calcular histogramas para ambos os gêneros usando os mesmos bins para garantir consistência
            if len(male_data) > 0 and len(female_data) > 0:
                # Usar bins comuns para ambos os gêneros
                male_counts, male_x = np.histogram(male_data, bins=30)
                female_counts, female_x = np.histogram(female_data, bins=male_x)  # Usar os mesmos bins do masculino
                # Calcular totais por barra (soma masculino + feminino)
                total_counts = male_counts + female_counts
            elif len(male_data) > 0:
                male_counts, male_x = np.histogram(male_data, bins=30)
                female_counts, female_x = np.array([]), male_x
                total_counts = male_counts
            elif len(female_data) > 0:
                male_counts, male_x = np.array([]), np.array([])
                female_counts, female_x = np.histogram(female_data, bins=30)
                total_counts = female_counts
            else:
                male_counts, male_x = np.array([]), np.array([])
                female_counts, female_x = np.array([]), np.array([])
                total_counts = np.array([])
            
            # Calcular centros e labels de intervalos para os bins
            bin_centers = [(male_x[i] + male_x[i + 1]) / 2 for i in range(len(male_x) - 1)]
            bin_labels = [f'{male_x[i]:.1f} - {male_x[i + 1]:.1f}' for i in range(len(male_x) - 1)]
            
            # Adicionar barras para masculino com hover baseado em intervalos
            if len(male_data) > 0:
                fig.add_trace(go.Bar(
                    x=bin_centers,
                    y=male_counts,
                    name=f'👨 Masculino ({len(male_data)} registros)',
                    opacity=0.7,
                    marker_color='#1f77b4',
                    customdata=bin_labels,
                    hovertemplate='<b>👨 Masculino</b><br>' +
                                  f'{TRADUCOES.get(variable, variable)}: %{{customdata}}<br>' +
                                  'Frequência: %{y}<br>' +
                                  '<extra></extra>'
                ))
            
            # Adicionar barras para feminino com hover baseado em intervalos
            if len(female_data) > 0:
                fig.add_trace(go.Bar(
                    x=bin_centers,
                    y=female_counts,
                    name=f'👩 Feminino ({len(female_data)} registros)', 
                    opacity=0.7,
                    marker_color='#ff7f0e',
                    customdata=bin_labels,
                    hovertemplate='<b>👩 Feminino</b><br>' +
                                  f'{TRADUCOES.get(variable, variable)}: %{{customdata}}<br>' +
                                  'Frequência: %{y}<br>' +
                                  '<extra></extra>'
                ))
            
            # Adicionar linha trace com totais baseado em intervalos
            if len(total_counts) > 0 and len(bin_centers) > 0:
                # Filtrar apenas bins com valores > 0
                filtered_centers = []
                filtered_values = []
                filtered_labels = []
                
                for i in range(len(total_counts)):
                    if total_counts[i] > 0:
                        filtered_centers.append(bin_centers[i])
                        filtered_values.append(total_counts[i])
                        filtered_labels.append(bin_labels[i])
                
                if filtered_centers:
                    fig.add_trace(go.Scatter(
                        x=filtered_centers,
                        y=filtered_values,
                        mode='lines+markers+text',
                        name=f'📊 Total (M+F): {sum(total_counts)} registros',
                        line=dict(color='#2ca02c', width=2, dash='dot'),
                        marker=dict(color='#2ca02c', size=8, symbol='circle'),
                        text=[str(v) for v in filtered_values],
                        textposition='top center',
                        textfont=dict(size=10, color='black', family='Arial'),
                        opacity=0.8,
                        customdata=filtered_labels,
                        hovertemplate='<b>📊 Total (M+F)</b><br>' +
                                      f'{TRADUCOES.get(variable, variable)}: %{{customdata}}<br>' +
                                      'Total de registros: %{y}<br>' +
                                      '<extra></extra>'
                    ))
            
            # Atualizar layout
            fig.update_layout(
                title=f"👫 Comparação por Género - {TRADUCOES.get(variable, variable)} (Cluster {cluster_id})",
                xaxis_title=TRADUCOES.get(variable, variable),
                yaxis_title="Frequência",
                barmode='overlay',
                height=600,
                showlegend=True,
                margin=dict(t=100, b=50, l=50, r=80),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            return fig
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"❌ Erro na comparação por género: {str(e)}", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
    
    @staticmethod
    def create_age_boxplots(cluster_info: Optional[Dict], k_value: int, cluster_id: int, variable: str = 'age_years') -> go.Figure:
        """Cria box plots estratificados por faixa etária para uma variável específica"""
        
        if cluster_info is None:
            return go.Figure().add_annotation(
                text="❌ Dados do cluster não encontrados.", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        try:
            cluster_data = cluster_info['data']
            
            # Verificar se a variável existe
            if variable not in cluster_data.columns:
                return go.Figure().add_annotation(
                    text=f"❌ Variável '{variable}' não encontrada nos dados.", 
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
            
            # Definir faixas etárias
            cluster_data_copy = cluster_data.copy()
            cluster_data_copy['age_group'] = pd.cut(cluster_data_copy['age_years'], 
                                                 bins=[0, 30, 45, 60, 100], 
                                                 labels=['18-30', '31-45', '46-60', '60+'])
            
            # Criar figura simples com box plots por faixa etária
            fig = go.Figure()
            
            colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']
            
            for i, age_group in enumerate(cluster_data_copy['age_group'].cat.categories):
                data = cluster_data_copy[cluster_data_copy['age_group'] == age_group][variable].dropna()
                if len(data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=data,
                            name=age_group,
                            marker_color=colors[i % len(colors)],
                            showlegend=False
                        )
                    )
            
            fig.update_layout(
                title=f"👥 Análise por Faixa Etária - {TRADUCOES.get(variable, variable)} (Cluster {cluster_id}, K={k_value})",
                xaxis_title="Faixa Etária",
                yaxis_title=TRADUCOES.get(variable, variable),
                height=500,
                font=dict(size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Adicionar grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"❌ Erro na análise por faixa etária: {str(e)}", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
    
    @staticmethod
    def create_radar_chart(cluster_info: Optional[Dict], k_value: int, cluster_id: int, 
                          df: pd.DataFrame, cluster_col: str) -> go.Figure:
        """Gria gráfico radar comparando cluster vs população geral"""
        
        if cluster_info is None:
            return go.Figure().add_annotation(
                text="❌ Dados do cluster não encontrados.", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        try:
            cluster_data = cluster_info['data']
            
            # Calcular estatísticas do cluster
            cluster_stats = cluster_data[EDA_NUMERIC_COLS].mean()
            
            # Calcular estatísticas da população geral (para o mesmo K)
            overall_stats = df[EDA_NUMERIC_COLS].mean()
            
            # Normalizar valores para escala 0-1
            cluster_normalized = (cluster_stats - overall_stats.min()) / (overall_stats.max() - overall_stats.min())
            overall_normalized = (overall_stats - overall_stats.min()) / (overall_stats.max() - overall_stats.min())
            
            # Traduzir labels
            labels = [TRADUCOES.get(col, col) for col in EDA_NUMERIC_COLS]
            
            fig = go.Figure()
            
            # Adicionar cluster
            fig.add_trace(go.Scatterpolar(
                r=cluster_normalized.tolist(),
                theta=labels,
                fill='toself',
                name=f'Cluster {cluster_id}',
                line_color='#1f77b4'
            ))
            
            # Adicionar população geral
            fig.add_trace(go.Scatterpolar(
                r=overall_normalized.tolist(),
                theta=labels,
                fill='toself',
                name='População Geral',
                line_color='#ff7f0e'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title=f"🎯 Perfil Radar - Cluster {cluster_id} vs População Geral (K={k_value})",
                height=600,
                font=dict(size=11)
            )
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"❌ Erro ao criar radar: {str(e)}", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )

# ================== COMPONENTES DE INTERFACE ==================

def create_cluster_metrics_component(cluster_info: Dict, cluster_id: int) -> html.Div:
    """Cria componente de métricas do cluster"""
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5(f"📋 Resumo do Cluster {cluster_id}", className="mb-0 fw-bold")
        ], className="bg-light"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-people-fill", style={"fontSize": "28px", "color": "#007bff"}),
                        ], className="mb-2"),
                        html.P("👥 Tamanho do Cluster", className="mb-1 fw-bold text-muted small"),
                        html.H3(f"{cluster_info['size']:,}", className="text-primary mb-0")
                    ], className="text-center p-3 bg-gradient bg-light rounded-3", 
                           style={"borderLeft": "4px solid #007bff"})
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-pie-chart-fill", style={"fontSize": "28px", "color": "#28a745"}),
                        ], className="mb-2"),
                        html.P("📊 % do Total", className="mb-1 fw-bold text-muted small"),
                        html.H3(f"{cluster_info['percentage']:.1f}%", className="text-success mb-0")
                    ], className="text-center p-3 bg-gradient bg-light rounded-3",
                           style={"borderLeft": "4px solid #28a745"})
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-heart-pulse-fill", style={"fontSize": "28px", "color": "#dc3545"}),
                        ], className="mb-2"),
                        html.P("❤️ Taxa de Doença", className="mb-1 fw-bold text-muted small"),
                        html.H3(f"{cluster_info['cardio_rate']:.1f}%", className="text-danger mb-0")
                    ], className="text-center p-3 bg-gradient bg-light rounded-3",
                           style={"borderLeft": "4px solid #dc3545"})
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-calendar3", style={"fontSize": "28px", "color": "#fd7e14"}),
                        ], className="mb-2"),
                        html.P("🎂 Idade Média", className="mb-1 fw-bold text-muted small"),
                        html.H3(f"{cluster_info['age_stats']['mean']:.1f} anos", className="text-primary mb-0")
                    ], className="text-center p-3 bg-gradient bg-light rounded-3",
                           style={"borderLeft": "4px solid #fd7e14"})
                ], md=3),
            ], className="g-3"),
            html.Hr(),
            html.Div([
                html.Small("💡 Use os controles acima para explorar diferentes atributos e métricas dos dados.", 
                          className="text-muted")
            ], className="text-center")
        ])
    ], className="shadow-lg border-0")

def create_loading_component(message: str = "Carregando...") -> html.Div:
    """Cria componente de loading"""
    return html.Div([
        dbc.Spinner(size="lg", color="primary"),
        html.P(message, className="mt-3 text-center")
    ], className="text-center p-5")

def create_error_component(error_message: str) -> dbc.Alert:
    """Cria componente de erro"""
    return dbc.Alert([
        html.H4("❌ Erro", className="alert-heading"),
        html.P(error_message),
        html.Hr(),
        html.P("Verifique os dados e tente novamente.", className="mb-0")
    ], color="danger", className="mb-4")

# ================== LAYOUT PADRONIZADO ==================

def layout():
    """Layout principal padronizado seguindo o estilo do Random Forest Dashboard"""
    
    return dbc.Container([
        # Trigger para carregamento automático
        dcc.Store(id='init-trigger', storage_type='memory'),
        
        # ========== HERO SECTION ==========
        html.Div([
            html.Div([
                html.H1("📊 Clusterização K-Means", 
                       className="display-4 fw-bold text-white mb-3"),
                html.P("Análise de Clusterização para Identificação de Padrões de Risco Cardiovascular", 
                      className="lead text-white-50 mb-4"),
                
                # MÉTRICAS GLOBAIS (dinâmicas)
                html.Div(id="hero-metrics-container", className="mt-4")
            ], className="container py-5")
        ], style={
            "background": f"linear-gradient(135deg, {PALETTE['gradient_start']} 0%, {PALETTE['gradient_end']} 100%)",
            "marginBottom": "40px",
            "borderRadius": "0 0 30px 30px",
            "boxShadow": "0 10px 40px rgba(0,0,0,0.2)"
        }),
        
        # ========== CONFIGURAÇÃO ==========
        dbc.Card([
            dbc.CardBody([
                html.H3([
                    html.I(className="bi bi-gear-fill me-3", style={"color": PALETTE['accent']}),
                    "Configuração da Análise"
                ], className="mb-4 fw-bold text-center"),
                
                html.Div([
                    html.Label("Selecione o número de clusters (K):", className="fw-bold me-3"),
                    dbc.RadioItems(
                        id="k-selector",
                        options=K_OPTIONS,
                        value=6,
                        inline=True,
                        label_checked_style={"fontWeight": "bold"},
                    ),
                ], className="d-flex align-items-center justify-content-center mb-3"),
                
                # Indicador de status dos dados
                html.Div(id="data-status-indicator", className="mt-3")
            ])
        ], style={"boxShadow": "0 6px 20px rgba(0,0,0,0.15)", "border": "none", "borderRadius": "15px"}, className="mb-5"),
        
        # ========== TABS DE NAVEGAÇÃO ==========
        dbc.Tabs([
            # ========== TAB 1: AVALIAÇÃO ==========
            dbc.Tab(
                label="🔍 Avaliação do Modelo",
                tab_id="tab-evaluation",
                children=[
                    html.Div([
                        make_section_header("search", "Avaliação do Modelo", 
                                            "Métricas de qualidade da clusterização"),
                        html.Div(id="evaluation-section")
                    ], className="p-4")
                ]
            ),
            
            # ========== TAB 2: VISUALIZAÇÕES ==========
            dbc.Tab(
                label="📊 Visualizações",
                tab_id="tab-visualizations",
                children=[
                    html.Div([
                        make_section_header("bar-chart-line-fill", "Análise Visual dos Clusters", 
                                            "Explore distribuições, comparações e padrões dos clusters"),
                        
                        # Filtro de Género
                        dbc.Card(dbc.CardBody([
                            dbc.Label("👥 Filtro por Género", className="fw-bold text-primary"),
                            dbc.RadioItems(
                                id="gender-filter",
                                options=GENDER_OPTIONS,
                                value=0,
                                inline=True,
                                label_checked_style={"fontWeight": "bold"},
                            ),
                        ]), className="mb-3"),
                        
                        # Loading para gráficos
                        html.Div(id="graphs-loading", style={"display": "none"}, children=[
                            create_loading_component("Carregando gráficos...")
                        ]),
                        
                        # Abas com gráficos
                        dbc.Tabs([
                            dbc.Tab(
                                dcc.Graph(id='cluster-dist-norm-graph'), 
                                label="📈 Distribuição Normalizada (%)",
                                tab_id="tab-0"
                            ),
                            dbc.Tab(
                                dbc.CardBody([
                                    dbc.Label("📊 Selecione uma métrica para comparar os clusters", className="fw-bold text-primary mb-2"),
                                    dcc.Dropdown(
                                        id='eda-metric-dropdown',
                                        options=[
                                            {'label': '📊 Média', 'value': 'mean'},
                                            {'label': '📐 Mediana', 'value': 'median'},
                                            {'label': '📏 Desvio Padrão', 'value': 'std'},
                                            {'label': '📉 Percentil 25', 'value': 'q25'},
                                            {'label': '📈 Percentil 75', 'value': 'q75'}
                                        ],
                                        value='mean',
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                    dcc.Graph(id='cluster-comparison-graph')
                                ]),
                                label="⚖️ Comparação Inter-Cluster", 
                                tab_id="tab-1"
                            ),
                            dbc.Tab(
                                dcc.Graph(id='cluster-heatmap-graph'),
                                label="🔥 Heatmap",
                                tab_id="tab-2"
                            ),
                        ], id="tabs", active_tab="tab-0")
                    ], className="p-4")
                ]
            ),
            
            # ========== TAB 3: EDA ==========
            dbc.Tab(
                label="🔬 EDA",
                tab_id="tab-eda",
                children=[
                    html.Div([
                        make_section_header("search", "Análise Exploratória de Dados (EDA)", 
                                            "Análise detalhada por cluster específico"),
                        
                        # 🎯 SELETOR DE CLUSTER
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5("🎯 Seleção de Cluster", className="mb-0 fw-bold text-dark")
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Escolha o Cluster para Análise:", className="fw-bold text-primary mb-3"),
                                        dcc.Dropdown(
                                            id='eda-cluster-selector',
                                            options=[],  # Será preenchido pelo callback
                                            value=0,
                                            clearable=False,
                                            className="mb-3",
                                            style={
                                                'borderRadius': '8px', 
                                                'fontSize': '16px',
                                                'zIndex': 9999,
                                                'position': 'relative'
                                            },
                                            placeholder="Selecione um cluster..."
                                        ),
                                        html.Div([
                                            html.I(className="bi bi-info-circle-fill", style={"color": "#17a2b8", "fontSize": "16px"}),
                                            html.Small(" Selecione um cluster para visualizar análises.", 
                                                     className="text-muted ms-2")
                                        ], className="d-flex align-items-center")
                                    ], md=6)
                                ])
                            ])
                        ], className="mb-4 shadow-sm", style={
                            "height": "200px", 
                            "overflow": "visible",
                            "position": "relative",
                            "zIndex": 9999
                        }),
                        
                        # 📋 RESUMO DO CLUSTER
                        html.Div(id="eda-cluster-metrics", className="mb-4"),
                        
                        # Loading para EDA
                        html.Div(id="eda-loading", style={"display": "none"}, children=[
                            create_loading_component("Carregando análise EDA...")
                        ]),
                        
                        # Abas com análises avançadas
                        dbc.Tabs([
                            dbc.Tab(
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("👥 Filtro por Género", className="fw-bold text-primary"),
                                            dbc.RadioItems(
                                                id='eda-correlation-gender-filter',
                                                options=[
                                                    {"label": "🌍 Todos", "value": 0},
                                                    {"label": "👨 Masculino", "value": 1}, 
                                                    {"label": "👩 Feminino", "value": 2}
                                                ],
                                                value=0,
                                                inline=True,
                                                label_checked_style={"fontWeight": "bold"},
                                            )
                                        ], md=4)
                                    ], className="mb-3"),
                                    dcc.Graph(id='eda-correlation-heatmap')
                                ]),
                                label="🔥 Matriz de Correlações",
                                tab_id="eda-tab-0"
                            ),
                            dbc.Tab(
                                html.Div([
                                    # Seletor de Variável para Comparação por Gênero
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("📊 Variável para Comparação por Género", className="fw-bold text-primary"),
                                            dcc.Dropdown(
                                                id='eda-distribution-variable',
                                                options=[
                                                    {'label': f'📈 {TRADUCOES.get(col, col)}', 'value': col} 
                                                    for col in ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight', 'cholesterol', 'gluc']
                                                ],
                                                value='age_years',
                                                clearable=False,
                                                className="mb-3",
                                                style={'borderRadius': '8px'}
                                            )
                                        ], md=4)
                                    ], className="mb-3"),
                                    dcc.Graph(id='eda-gender-comparison-graph')
                                ]),
                                label="👫 Comparação por Género",
                                tab_id="eda-tab-1"
                            ),
                            dbc.Tab(
                                html.Div([
                                    # Seletor de Variável e Filtro de Género para Análise por Faixa Etária
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("📊 Variável para Análise por Faixa Etária", className="fw-bold text-primary"),
                                            dcc.Dropdown(
                                                id='eda-age-variable',
                                                options=[
                                                    {'label': f'📈 {TRADUCOES.get(col, col)}', 'value': col} 
                                                    for col in ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight', 'cholesterol', 'gluc']
                                                ],
                                                value='age_years',
                                                clearable=False,
                                                className="mb-3",
                                                style={'borderRadius': '8px'}
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("👥 Filtro por Género", className="fw-bold text-primary"),
                                            dbc.RadioItems(
                                                id='eda-age-gender-filter',
                                                options=[
                                                    {"label": "🌍 Todos", "value": 0},
                                                    {"label": "👨 Masculino", "value": 1}, 
                                                    {"label": "👩 Feminino", "value": 2}
                                                ],
                                                value=0,
                                                inline=True,
                                                label_checked_style={"fontWeight": "bold"},
                                            )
                                        ], md=4)
                                    ], className="mb-3"),
                                    dcc.Graph(id='eda-age-boxplots')
                                ]),
                                label="👥 Análise por Faixa Etária",
                                tab_id="eda-tab-2"
                            ),
                            dbc.Tab(
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("👥 Filtro por Género", className="fw-bold text-primary"),
                                            dbc.RadioItems(
                                                id='eda-radar-gender-filter',
                                                options=[
                                                    {"label": "🌍 Todos", "value": 0},
                                                    {"label": "👨 Masculino", "value": 1}, 
                                                    {"label": "👩 Feminino", "value": 2}
                                                ],
                                                value=0,
                                                inline=True,
                                                label_checked_style={"fontWeight": "bold"},
                                            )
                                        ], md=4)
                                    ], className="mb-3"),
                                    dcc.Graph(id='eda-radar-chart')
                                ]),
                                label="🎯 Perfil Radar",
                                tab_id="eda-tab-3"
                            ),
                        ], id="eda-tabs", active_tab="eda-tab-0", className="mt-4")
                    ], className="p-4")
                ]
            ),
            
            # ========== TAB 4: INTERPRETAÇÃO ==========
            dbc.Tab(
                label="💡 Interpretação",
                tab_id="tab-interpretation",
                children=[
                    html.Div([
                        make_section_header("lightbulb", "Interpretação dos Clusters", 
                                            "Personas e características de cada cluster"),
                        html.Div(id="interpretation-section")
                    ], className="p-4")
                ]
            ),
        ], id="main-tabs", active_tab="tab-evaluation", className="mb-4"),
        
        # Footer
        html.Div([
            html.Hr(style={"borderTop": f"2px solid {PALETTE['muted']}"}),
            html.P("Dashboard desenvolvido com Dash & Plotly | Clusterização K-Means", 
                   className="text-center text-muted small py-3")
        ])
        
    ], fluid=True, className="px-4 py-4", style={"backgroundColor": "#fafbfc"})

# ================== CALLBACKS ==================

@callback(
    Output('init-trigger', 'data'),
    [Input('k-selector', 'value')]
)
def initialize_app(k_value):
    """Trigger inicial para carregar automaticamente todas as seções"""
    return k_value

@callback(
    Output('data-status-indicator', 'children'),
    [Input('k-selector', 'value'),
     Input('init-trigger', 'data')]
)
def update_data_status(k_value, init_trigger):
    """Atualiza indicador de status dos dados"""
    try:
        artifacts = load_data_and_artifacts_optimized(k_value)
        if artifacts["error"]:
            return create_error_component(artifacts["error"])
        else:
            return None
    except Exception as e:
        return create_error_component(f"Erro ao verificar status: {str(e)}")

@callback(
    Output('hero-metrics-container', 'children'),
    [Input('k-selector', 'value'),
     Input('init-trigger', 'data')]
)
def update_hero_metrics(k_value, init_trigger):
    """Atualiza métricas do hero section quando K muda"""
    try:
        artifacts = load_data_and_artifacts_optimized(k_value)
        validation_df = artifacts.get('validation', pd.DataFrame())
        df = artifacts.get('df', pd.DataFrame())
        
        total_clusters = len(validation_df) if not validation_df.empty else 0
        max_risk = validation_df["Taxa de Risco (%)"].max() if not validation_df.empty else 0
        min_risk = validation_df["Taxa de Risco (%)"].min() if not validation_df.empty else 0
        total_patients = len(df) if not df.empty else 0
    except:
        total_clusters = 0
        max_risk = 0
        min_risk = 0
        total_patients = 0
    
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-diagram-3-fill", 
                          style={"fontSize": "24px", "color": "white", "marginBottom": "8px"}),
                    html.P("Total de Clusters", className="text-white-50 mb-1", 
                          style={"fontSize": "14px"}),
                    html.H2(f"{total_clusters}", 
                           className="text-white mb-0 fw-bold"),
                ], className="text-center p-3", 
                   style={"backgroundColor": "rgba(255,255,255,0.15)", 
                          "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"})
            ])
        ], md=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-exclamation-triangle-fill", 
                          style={"fontSize": "24px", "color": "white", "marginBottom": "8px"}),
                    html.P("Maior Risco", className="text-white-50 mb-1", 
                          style={"fontSize": "14px"}),
                    html.H2(f"{max_risk:.1f}%", 
                           className="text-white mb-0 fw-bold"),
                ], className="text-center p-3", 
                   style={"backgroundColor": "rgba(255,255,255,0.15)", 
                          "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"})
            ])
        ], md=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-check-circle-fill", 
                          style={"fontSize": "24px", "color": "white", "marginBottom": "8px"}),
                    html.P("Menor Risco", className="text-white-50 mb-1", 
                          style={"fontSize": "14px"}),
                    html.H2(f"{min_risk:.1f}%", 
                           className="text-white mb-0 fw-bold"),
                ], className="text-center p-3", 
                   style={"backgroundColor": "rgba(255,255,255,0.15)", 
                          "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"})
            ])
        ], md=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-people-fill", 
                          style={"fontSize": "24px", "color": "white", "marginBottom": "8px"}),
                    html.P("Total Pacientes", className="text-white-50 mb-1", 
                          style={"fontSize": "14px"}),
                    html.H2(f"{total_patients:,}", 
                           className="text-white mb-0 fw-bold"),
                ], className="text-center p-3", 
                   style={"backgroundColor": "rgba(255,255,255,0.15)", 
                          "borderRadius": "12px", "border": "2px solid rgba(255,255,255,0.2)"})
            ])
        ], md=3),
    ])

@callback(
    [Output('cluster-dist-norm-graph', 'figure'),
     Output('cluster-comparison-graph', 'figure'),
     Output('cluster-heatmap-graph', 'figure'),
     Output('graphs-loading', 'style')],
    [Input('k-selector', 'value'),
     Input('gender-filter', 'value'),
     Input('eda-metric-dropdown', 'value'),
     Input('init-trigger', 'data')]
)
def update_all_visualizations(k_value, gender_value, selected_metric, init_trigger):
    """Atualiza todas as visualizações com loading"""
    
    # Mostrar loading
    loading_style = {"display": "block"}
    
    try:
        # Validar entrada
        k_value = DataValidator.validate_k_value(k_value)
        
        # Carregar dados
        artifacts = load_data_and_artifacts_optimized(k_value)
        if artifacts["error"]:
            error_fig = go.Figure().add_annotation(
                text=artifacts["error"], 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return error_fig, error_fig, error_fig, {"display": "none"}
        
        df = artifacts["df"]
        
        # Validar métrica selecionada
        if selected_metric is None:
            selected_metric = 'mean'
        
        # Criar gráficos
        dist_fig = create_dist_norm_graph_optimized(df, gender_value, k_value)
        heatmap_fig = create_heatmap_optimized(df, gender_value, k_value)
        comparison_fig = create_cluster_comparison_visualization(df, gender_value, k_value, selected_metric)
        
        return dist_fig, comparison_fig, heatmap_fig, {"display": "none"}
        
    except Exception as e:
        error_fig = go.Figure().add_annotation(
            text=f"❌ Erro ao atualizar gráficos: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return error_fig, error_fig, error_fig, {"display": "none"}

@callback(
    [Output('eda-cluster-selector', 'options'),
     Output('eda-cluster-selector', 'value'),
     Output('eda-cluster-metrics', 'children'),
     Output('eda-correlation-heatmap', 'figure'),
     Output('eda-gender-comparison-graph', 'figure'),
     Output('eda-age-boxplots', 'figure'),
     Output('eda-radar-chart', 'figure'),
     Output('eda-loading', 'style')],
    [Input('k-selector', 'value'),
     Input('eda-cluster-selector', 'value'),
     Input('eda-correlation-gender-filter', 'value'),
     Input('eda-age-gender-filter', 'value'),
     Input('eda-radar-gender-filter', 'value'),
     Input('eda-distribution-variable', 'value'),
     Input('eda-age-variable', 'value'),
     Input('init-trigger', 'data')]
)
def update_eda_section(k_value, selected_cluster, eda_correlation_gender, eda_age_gender, eda_radar_gender, eda_distribution_variable, eda_age_variable, init_trigger):
    """Atualiza seção de EDA"""
    
    # Mostrar loading
    loading_style = {"display": "block"}
    
    try:
        # Validar K
        k_value = DataValidator.validate_k_value(k_value)
        
        # Validar variável para distribuição
        valid_attributes = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight', 'cholesterol', 'gluc']
        if eda_distribution_variable not in valid_attributes:
            eda_distribution_variable = 'age_years'
        
        # Validar variável para idade
        if eda_age_variable not in valid_attributes:
            eda_age_variable = 'age_years'
        
        # Carregar dados
        artifacts = load_data_and_artifacts_optimized(k_value)
        if artifacts["error"]:
            error_fig = go.Figure().add_annotation(
                text=artifacts["error"], 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return [], 0, create_error_component(artifacts["error"]), error_fig, error_fig, error_fig, error_fig, {"display": "none"}
        
        df = artifacts["df"]
        cluster_col = f'clusterk{k_value}'
        
        # Gerar opções para cluster
        n_clusters = 6 if k_value == 6 else 16
        cluster_options = [{'label': f'Cluster {i}', 'value': i} for i in range(n_clusters)]
        
        # Validar cluster selecionado
        if selected_cluster is None or selected_cluster >= n_clusters:
            selected_cluster = 0
        
        # Obter informações do cluster para cada análise com seu próprio filtro de gênero
        cluster_info_correlation = ClusterAnalyzer.get_cluster_info_safe(
            df, cluster_col, selected_cluster, eda_correlation_gender
        )
        cluster_info_age = ClusterAnalyzer.get_cluster_info_safe(
            df, cluster_col, selected_cluster, eda_age_gender
        )
        cluster_info_radar = ClusterAnalyzer.get_cluster_info_safe(
            df, cluster_col, selected_cluster, eda_radar_gender
        )
        # Para métricas e comparação por gênero, usar todos os dados
        cluster_info = ClusterAnalyzer.get_cluster_info_safe(
            df, cluster_col, selected_cluster, 0
        )
        
        # Criar métricas
        metrics_content = html.Div()
        if cluster_info:
            metrics_content = create_cluster_metrics_component(cluster_info, selected_cluster)
        
        # Criar gráficos EDA avançados com tratamento de erro e filtros de gênero individuais
        try:
            gender_comp_fig = ClusterAnalyzer.create_gender_comparison(cluster_info, k_value, selected_cluster, eda_distribution_variable)
        except Exception:
            gender_comp_fig = go.Figure().add_annotation(text="❌ Erro ao criar comparação por género", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        try:
            correlation_fig = ClusterAnalyzer.create_correlation_heatmap(cluster_info_correlation, k_value, selected_cluster)
        except Exception:
            correlation_fig = go.Figure().add_annotation(text="❌ Erro ao criar correlação", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        try:
            radar_fig = ClusterAnalyzer.create_radar_chart(cluster_info_radar, k_value, selected_cluster, df, cluster_col)
        except Exception:
            radar_fig = go.Figure().add_annotation(text="❌ Erro ao criar radar", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        try:
            age_boxplots_fig = ClusterAnalyzer.create_age_boxplots(cluster_info_age, k_value, selected_cluster, eda_age_variable)
        except Exception:
            age_boxplots_fig = go.Figure().add_annotation(text="❌ Erro nos box plots por idade", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        return (cluster_options, selected_cluster, metrics_content,
                correlation_fig, gender_comp_fig, age_boxplots_fig, radar_fig,
                {"display": "none"})
        
    except Exception as e:
        error_msg = f"Erro na seção EDA: {str(e)}"
        error_fig = go.Figure().add_annotation(
            text=error_msg, 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return [], 0, create_error_component(error_msg), error_fig, error_fig, error_fig, error_fig, {"display": "none"}

@callback(
    Output('evaluation-section', 'children'),
    [Input('k-selector', 'value'),
     Input('init-trigger', 'data')]
)
def update_evaluation_section(k_value, init_trigger):
    """Atualiza seção de avaliação"""
    
    try:
        artifacts = load_data_and_artifacts_optimized(k_value)
        if artifacts["error"]:
            return create_error_component(artifacts["error"])
        
        eval_images = artifacts["eval_images"]
        
        if eval_images:
            # Criar sistema de abas para mostrar uma figura por vez
            tabs_children = []
            for i, img in enumerate(eval_images):
                tabs_children.append(
                    dbc.Tab(
                        html.Div([
                            html.H6(img["name"], className="card-title mb-3 text-center fw-bold"),
                            html.Img(
                                src=img["src"], 
                                style={"width": "100%", "height": "auto", "maxHeight": "600px"},
                                className="img-fluid border rounded"
                            )
                        ], className="p-3"),
                        label=f"📊 {img['name']}",
                        tab_id=f"eval-tab-{i}"
                    )
                )
            
            return dbc.Card([
                dbc.CardBody([
                    dbc.Tabs(
                        tabs_children,
                        id="evaluation-tabs",
                        active_tab="eval-tab-0" if eval_images else None
                    )
                ])
            ], style={"boxShadow": "0 4px 15px rgba(0,0,0,0.1)", "border": "none"})
        
        return create_error_component("Imagens de avaliação não disponíveis.")
        
    except Exception as e:
        return create_error_component(f"Erro ao carregar avaliação: {str(e)}")

@callback(
    Output('interpretation-section', 'children'),
    [Input('k-selector', 'value'),
     Input('init-trigger', 'data')]
)
def update_interpretation_section(k_value, init_trigger):
    """Atualiza seção de interpretação"""
    
    try:
        artifacts = load_data_and_artifacts_optimized(k_value)
        if artifacts["error"]:
            return create_error_component(artifacts["error"])
        
        interpretations = artifacts["persona_interpretations"]
        
        if interpretations:
            interpretation_cards = []
            for cluster_name, description in interpretations.items():
                interpretation_cards.append(
                    dbc.Card([
                        dbc.CardHeader(cluster_name, className="fw-bold"),
                        dbc.CardBody([
                            html.P(description)
                        ])
                    ], className="mb-3", style={"boxShadow": "0 2px 8px rgba(0,0,0,0.1)", "border": "none"})
                )
            
            return html.Div(interpretation_cards)
        
        return create_error_component("Interpretações não disponíveis.")
        
    except Exception as e:
        return create_error_component(f"Erro ao gerar interpretação: {str(e)}")

# ================== LIMPEZA AUTOMÁTICA ==================

def cleanup_cache():
    """Limpa cache expirado periodicamente"""
    cache.clear_expired()

# Executar limpeza de cache periodicamente
import threading

def periodic_cache_cleanup():
    """Executa limpeza de cache a cada 10 minutos"""
    while True:
        time.sleep(600)  # 10 minutos
        cleanup_cache()
        print("🧹 Cache limpo automaticamente")

# Iniciar thread de limpeza em background
cleanup_thread = threading.Thread(target=periodic_cache_cleanup, daemon=True)
cleanup_thread.start()