"""
Dashboard principal multipage com Dash.
Centraliza navegação, tema e layout padrão para todas as páginas.
"""

import dash
from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from styles import PALETTE, SPACING, SIDEBAR_STYLE, CONTENT_STYLE

# ================== INICIALIZAÇÃO ==================
app = Dash(
    __name__,
    use_pages=True,  # Habilita multipáginas
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,  # Ícones Bootstrap
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Dashboard Cardiovascular - ML Pipeline",
    update_title="Carregando...",
)

# Permitir callbacks duplicados (necessário para hot-reload em desenvolvimento)
app.config.suppress_callback_exceptions = True

server = app.server  # Para deploy em produção

# ================== SIDEBAR ==================
def create_sidebar():
    """Cria barra lateral colapsável com navegação entre páginas."""
    
    # Header da sidebar
    sidebar_header = html.Div([
        html.Div([
            html.H3("ML Dashboard", className="text-white mb-1", 
                   style={"fontWeight": "700", "fontSize": "20px"}, id="sidebar-title"),
            html.P("Pipeline Cardiovascular", className="text-white-50 small mb-3", 
                  id="sidebar-subtitle"),
        ], id="sidebar-header-content"),
    ])
    
    # Links de navegação automáticos baseados no page_registry
    nav_links = []
    
    # Adicionar link para Home/Overview
    nav_links.append(
        dbc.NavLink(
            [
                html.I(className="bi bi-house-door-fill me-2", id="icon-home"),
                html.Span("Home", id="label-home"),
            ],
            href="/",
            active="exact",
            className="text-white mb-2 nav-link-custom",
            id="nav-home",
        )
    )
    
    for page in dash.page_registry.values():
        if page["path"] != "/":  # Home já foi adicionado
            page_id = page["path"].strip("/").replace("-", "_")
            nav_links.append(
                dbc.NavLink(
                    [
                        html.I(className=f"bi bi-{page.get('icon', 'file-bar-graph')} me-2", 
                              id=f"icon-{page_id}"),
                        html.Span(page["name"], id=f"label-{page_id}"),
                    ],
                    href=page["path"],
                    active="exact",
                    className="text-white mb-2 nav-link-custom",
                    id=f"nav-{page_id}",
                )
            )
    
    # Footer da sidebar
    sidebar_footer = html.Div([
        html.Hr(style={"borderColor": "rgba(255,255,255,0.2)"}, id="sidebar-hr"),
        html.Div([
            html.P([
                html.I(className="bi bi-github me-2"),
                html.Span("GitHub", className="sidebar-footer-text"),
            ], className="mb-1 text-white-50 small", id="footer-github"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                html.Span("Docs", className="sidebar-footer-text"),
            ], className="mb-0 text-white-50 small", id="footer-docs"),
        ], id="sidebar-footer-links"),
    ], style={"marginTop": "auto"}, id="sidebar-footer-container")
    
    return html.Div([
        # Botão de colapsar/expandir
        html.Button(
            html.I(className="bi bi-chevron-left", id="collapse-icon"),
            id="sidebar-toggle",
            className="btn btn-link text-white position-absolute",
            style={
                "top": "10px",
                "right": "10px",
                "zIndex": 1000,
                "padding": "5px 10px",
                "fontSize": "20px",
            },
        ),
        
        html.Div([
            sidebar_header,
            html.Nav(nav_links, className="mb-4 flex-grow-1", id="sidebar-nav"),
            sidebar_footer,
        ], style={"display": "flex", "flexDirection": "column", "height": "100%"}),
    ], id="sidebar", style={**SIDEBAR_STYLE, "transition": "all 0.3s ease"})

# ================== LAYOUT PRINCIPAL ==================
app.layout = html.Div([
    # Store para estado da sidebar
    dcc.Store(id="sidebar-collapsed", data=False),
    
    # Sidebar
    create_sidebar(),
    
    # Container de conteúdo (onde as páginas são renderizadas)
    html.Div([
        # Breadcrumb dinâmico
        html.Div(
            id="breadcrumb-container",
            style={
                "backgroundColor": "white",
                "padding": f"{SPACING['sm']} {SPACING['md']}",
                "marginBottom": SPACING["md"],
                "borderRadius": "6px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
            }
        ),
        
        # Conteúdo da página atual
        dash.page_container,
        
    ], id="content", style={**CONTENT_STYLE, "transition": "margin-left 0.3s ease"}),
    
], style={"fontFamily": "'Inter', sans-serif"})


# ================== CALLBACKS ==================

@callback(
    [Output("sidebar", "className"),
     Output("content", "className"),
     Output("collapse-icon", "className"),
     Output("sidebar-collapsed", "data")],
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-collapsed", "data"),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, is_collapsed):
    """Colapsa/expande a sidebar."""
    if n_clicks is None:
        return dash.no_update
    
    new_state = not is_collapsed
    
    if new_state:  # Colapsada
        return "collapsed", "sidebar-collapsed", "bi bi-chevron-right", new_state
    else:  # Expandida
        return "", "", "bi bi-chevron-left", new_state


@callback(
    Output("breadcrumb-container", "children"),
    Input("_pages_location", "pathname")
)
def update_breadcrumb(pathname):
    """Atualiza breadcrumb baseado na página atual."""
    if pathname == "/" or pathname is None:
        current_page = "Home"
        icon = "house-door-fill"
    else:
        # Buscar página no registry
        page_info = None
        for page in dash.page_registry.values():
            if page["path"] == pathname:
                page_info = page
                break
        
        if page_info:
            current_page = page_info.get("name", "Dashboard")
            icon = page_info.get("icon", "file-bar-graph")
        else:
            current_page = "Dashboard"
            icon = "file-bar-graph"
    
    return html.Div([
        html.Div([
            html.I(className=f"bi bi-{icon} me-2", style={"fontSize": "20px", "color": PALETTE["primary"]}),
            html.Span("Home", className="text-muted small me-2"),
            html.I(className="bi bi-chevron-right me-2", style={"fontSize": "12px", "color": PALETTE["muted"]}),
            html.Span(current_page, className="fw-bold", style={"color": PALETTE["dark"]}),
        ], style={"display": "flex", "alignItems": "center"}),
        
        html.Div([
            html.Small(
                f"📍 {pathname}", 
                className="text-muted",
                style={"fontFamily": "monospace"}
            ),
        ]),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "width": "100%"})



# ================== EXECUÇÃO ==================
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
