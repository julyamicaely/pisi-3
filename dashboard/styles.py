"""
Estilos centralizados e templates Plotly para governança visual do dashboard.
Define paleta, tipografia, espaçamentos e configurações Plotly padrão.
"""

import plotly.io as pio
import plotly.graph_objects as go

# ================== PALETA DE CORES ==================
PALETTE = {
    "primary": "#1E88E5",      # Azul vibrante
    "secondary": "#7B1FA2",    # Roxo profundo
    "accent": "#00ACC1",       # Ciano elétrico
    "warn": "#F4511E",         # Laranja forte (alertas)
    "success": "#43A047",      # Verde vibrante (sucesso)
    "info": "#00897B",         # Verde-azulado
    "light": "#F5F7FA",        # Cinza claro (background)
    "dark": "#263238",         # Azul escuro (texto)
    "muted": "#78909C",        # Cinza azulado
    "gradient_start": "#667eea", # Gradiente roxo-azul
    "gradient_end": "#764ba2",   # Gradiente roxo escuro
    "chart_1": "#FF6B6B",      # Vermelho coral
    "chart_2": "#4ECDC4",      # Turquesa
    "chart_3": "#FFE66D",      # Amarelo vibrante
    "chart_4": "#A8E6CF",      # Verde menta
}

# ================== TIPOGRAFIA ==================
FONTS = {
    "base": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif",
    "heading": "Inter, sans-serif",
    "monospace": "'Fira Code', 'Courier New', monospace",
}

# ================== TAMANHOS DE TEXTO ==================
SIZES = {
    "h1": {"fontSize": "32px", "fontWeight": "700", "lineHeight": "1.2"},
    "h2": {"fontSize": "28px", "fontWeight": "700", "lineHeight": "1.3"},
    "h3": {"fontSize": "24px", "fontWeight": "600", "lineHeight": "1.4"},
    "h4": {"fontSize": "20px", "fontWeight": "600", "lineHeight": "1.4"},
    "h5": {"fontSize": "18px", "fontWeight": "600", "lineHeight": "1.5"},
    "body": {"fontSize": "16px", "fontWeight": "400", "lineHeight": "1.6"},
    "small": {"fontSize": "14px", "fontWeight": "400", "lineHeight": "1.5"},
    "caption": {"fontSize": "12px", "fontWeight": "400", "lineHeight": "1.4"},
}

# ================== ESPAÇAMENTOS ==================
SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "48px",
}

# ================== ESTILOS DE COMPONENTES ==================
CARD_STYLE = {
    "backgroundColor": "white",
    "borderRadius": "8px",
    "padding": SPACING["lg"],
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
    "marginBottom": SPACING["md"],
}

METRIC_CARD_STYLE = {
    **CARD_STYLE,
    "textAlign": "center",
    "padding": SPACING["md"],
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "260px",
    "padding": SPACING["lg"],
    "backgroundColor": PALETTE["primary"],
    "color": "white",
    "overflowY": "auto",
}

CONTENT_STYLE = {
    "marginLeft": "260px",
    "padding": SPACING["lg"],
    "backgroundColor": PALETTE["light"],
    "minHeight": "100vh",
}

HEADER_STYLE = {
    "backgroundColor": "white",
    "padding": SPACING["md"],
    "marginBottom": SPACING["lg"],
    "borderRadius": "8px",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
}

# ================== TEMPLATE PLOTLY ==================
def create_plotly_template():
    """Cria e registra template Plotly customizado para governança visual."""
    
    template = go.layout.Template()
    
    # Layout padrão
    template.layout = go.Layout(
        font=dict(
            family=FONTS["base"],
            size=14,
            color=PALETTE["dark"],
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family=FONTS["base"],
            bordercolor=PALETTE["primary"],
        ),
        title=dict(
            font=dict(size=20, color=PALETTE["dark"], family=FONTS["heading"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            showline=True,
            linecolor=PALETTE["muted"],
            linewidth=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            showline=True,
            linecolor=PALETTE["muted"],
            linewidth=1,
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=PALETTE["muted"],
            borderwidth=1,
        ),
        colorway=[
            PALETTE["primary"],
            PALETTE["accent"],
            PALETTE["info"],
            PALETTE["success"],
            PALETTE["warn"],
            PALETTE["secondary"],
        ],
    )
    
    # Registrar template
    pio.templates["team_template"] = template
    pio.templates.default = "team_template"

# Inicializar template ao importar
create_plotly_template()

# ================== UTILITÁRIOS DE ESTILO ==================
def get_metric_color(value, threshold_good=0.85, threshold_warn=0.70):
    """Retorna cor baseada no valor da métrica."""
    if value >= threshold_good:
        return PALETTE["success"]
    elif value >= threshold_warn:
        return PALETTE["warn"]
    else:
        return PALETTE["warn"]

def apply_card_style(**kwargs):
    """Retorna estilo de card com overrides opcionais."""
    style = CARD_STYLE.copy()
    style.update(kwargs)
    return style

def apply_metric_style(**kwargs):
    """Retorna estilo de card de métrica com overrides opcionais."""
    style = METRIC_CARD_STYLE.copy()
    style.update(kwargs)
    return style
