"""
Componentes utilitários reutilizáveis para construção de layouts.
Funções puras para criar cards, métricas, grids e layouts padrão.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from styles import (
    PALETTE, SPACING, SIZES, CARD_STYLE, METRIC_CARD_STYLE,
    get_metric_color, apply_card_style
)


def make_card(title, body, icon=None, **style_overrides):
    """
    Cria um card padrão com título e corpo.
    
    Args:
        title (str): Título do card
        body (component): Componente Dash para o corpo
        icon (str): Nome do ícone Bootstrap (opcional)
        **style_overrides: Estilos customizados adicionais
    
    Returns:
        dbc.Card: Card estilizado
    """
    header_content = [html.H5(title, className="mb-0")]
    if icon:
        header_content.insert(0, html.I(className=f"bi bi-{icon} me-2"))
    
    return dbc.Card([
        dbc.CardHeader(header_content, style={"fontWeight": "600"}),
        dbc.CardBody(body),
    ], style=apply_card_style(**style_overrides))


def make_metric_card(label, value, format_fn=None, threshold_good=0.85, threshold_warn=0.70):
    """
    Cria um card de métrica com valor destacado e cor dinâmica.
    
    Args:
        label (str): Nome da métrica
        value (float): Valor da métrica
        format_fn (callable): Função para formatar valor (default: percentual)
        threshold_good (float): Limite para cor verde
        threshold_warn (float): Limite para cor amarela
    
    Returns:
        dbc.Card: Card de métrica
    """
    if format_fn is None:
        format_fn = lambda x: f"{x*100:.1f}%" if isinstance(x, float) else str(x)
    
    color = get_metric_color(value, threshold_good, threshold_warn) if isinstance(value, (int, float)) else PALETTE["primary"]
    
    return dbc.Card([
        dbc.CardBody([
            html.P(label, className="text-muted mb-1", style=SIZES["small"]),
            html.H3(
                format_fn(value),
                className="mb-0",
                style={"color": color, **SIZES["h2"]}
            ),
        ], style={"textAlign": "center", "padding": SPACING["md"]})
    ], style=METRIC_CARD_STYLE)


def build_metric_grid(metrics, cols=4):
    """
    Constrói grid responsivo de cards de métricas.
    
    Args:
        metrics (list): Lista de dicts com {label, value, format_fn (opcional)}
        cols (int): Número de colunas no grid
    
    Returns:
        dbc.Row: Grid de métricas
    """
    metric_cards = [
        dbc.Col(
            make_metric_card(
                m["label"],
                m["value"],
                m.get("format_fn"),
                m.get("threshold_good", 0.85),
                m.get("threshold_warn", 0.70)
            ),
            width=12 // cols,
            lg=12 // cols,
            md=6,
            sm=12,
        )
        for m in metrics
    ]
    
    return dbc.Row(metric_cards, className="mb-4")


def make_section_header(title, subtitle=None):
    """
    Cria cabeçalho de seção com título e subtítulo opcional.
    
    Args:
        title (str): Título da seção
        subtitle (str): Subtítulo ou descrição
    
    Returns:
        html.Div: Header estilizado
    """
    elements = [html.H3(title, style={**SIZES["h3"], "color": PALETTE["dark"]})]
    
    if subtitle:
        elements.append(
            html.P(subtitle, className="text-muted", style=SIZES["body"])
        )
    
    return html.Div(elements, className="mb-4")


def make_tabs(tabs_config):
    """
    Cria componente de abas com conteúdo.
    
    Args:
        tabs_config (list): Lista de dicts com {label, content, id (opcional)}
    
    Returns:
        dbc.Tabs: Componente de abas
    """
    tabs = [
        dbc.Tab(
            tab["content"],
            label=tab["label"],
            tab_id=tab.get("id", f"tab-{i}"),
        )
        for i, tab in enumerate(tabs_config)
    ]
    
    return dbc.Tabs(tabs, className="mb-3")


def make_loading_wrapper(children, loading_id):
    """
    Envolve componente com spinner de loading.
    
    Args:
        children: Componente a ser envolvido
        loading_id (str): ID único para o loading
    
    Returns:
        dcc.Loading: Wrapper com loading
    """
    return dcc.Loading(
        id=loading_id,
        type="default",
        color=PALETTE["accent"],
        children=children,
    )


def make_alert(message, color="info", dismissable=True):
    """
    Cria alert estilizado.
    
    Args:
        message (str): Mensagem do alert
        color (str): Cor Bootstrap (info, success, warning, danger)
        dismissable (bool): Se pode ser fechado
    
    Returns:
        dbc.Alert: Alert component
    """
    return dbc.Alert(
        message,
        color=color,
        dismissable=dismissable,
        className="mb-3",
    )


def make_empty_state(message="Nenhum dado disponível", icon="info-circle"):
    """
    Cria estado vazio quando não há dados.
    
    Args:
        message (str): Mensagem a exibir
        icon (str): Ícone Bootstrap
    
    Returns:
        html.Div: Estado vazio estilizado
    """
    return html.Div([
        html.I(className=f"bi bi-{icon}", style={"fontSize": "48px", "color": PALETTE["muted"]}),
        html.P(message, className="text-muted mt-3", style=SIZES["body"]),
    ], style={
        "textAlign": "center",
        "padding": SPACING["xxl"],
        "backgroundColor": PALETTE["light"],
        "borderRadius": "8px",
    })


def make_page_header(title, description=None, icon=None):
    """
    Cria header padrão de página com título, descrição e ícone.
    
    Args:
        title (str): Título da página
        description (str): Descrição da página
        icon (str): Ícone Bootstrap
    
    Returns:
        html.Div: Header da página
    """
    header_elements = []
    
    if icon:
        header_elements.append(
            html.I(className=f"bi bi-{icon}", style={"fontSize": "32px", "color": PALETTE["accent"], "marginRight": SPACING["md"]})
        )
    
    title_container = [html.H1(title, style=SIZES["h1"], className="mb-2")]
    
    if description:
        title_container.append(
            html.P(description, className="text-muted", style=SIZES["body"])
        )
    
    header_elements.append(html.Div(title_container))
    
    return html.Div(
        header_elements,
        style={
            "display": "flex",
            "alignItems": "center",
            "backgroundColor": "white",
            "padding": SPACING["lg"],
            "borderRadius": "8px",
            "marginBottom": SPACING["lg"],
            "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
        }
    )


def make_info_tooltip(text, tooltip_id):
    """
    Cria ícone de informação com tooltip.
    
    Args:
        text (str): Texto do tooltip
        tooltip_id (str): ID único
    
    Returns:
        html.Span: Ícone com tooltip
    """
    return html.Span([
        html.I(
            className="bi bi-info-circle",
            id=tooltip_id,
            style={"color": PALETTE["info"], "cursor": "pointer", "marginLeft": SPACING["xs"]}
        ),
        dbc.Tooltip(text, target=tooltip_id),
    ])
