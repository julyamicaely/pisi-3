"""
Dashboard principal multipage com Dash.
Centraliza navegação, tema e layout padrão para todas as páginas.
"""

import dash
from dash import Dash, html, dcc
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

server = app.server  # Para deploy em produção

# ================== SIDEBAR ==================
def create_sidebar():
    """Cria barra lateral com navegação entre páginas."""
    
    # Header da sidebar
    sidebar_header = html.Div([
        html.H3("ML Dashboard", className="text-white mb-1", style={"fontWeight": "700"}),
        html.P("Pipeline Cardiovascular", className="text-white-50 small mb-4"),
    ])
    
    # Links de navegação automáticos baseados no page_registry
    nav_links = []
    for page in dash.page_registry.values():
        if page["path"] != "/":  # Ignora home se existir
            nav_links.append(
                dbc.NavLink(
                    [
                        html.I(className=f"bi bi-{page.get('icon', 'file-bar-graph')} me-2"),
                        page["name"],
                    ],
                    href=page["path"],
                    active="exact",
                    className="text-white mb-2",
                    style={
                        "borderRadius": "6px",
                        "padding": "12px 16px",
                        "transition": "all 0.2s",
                    },
                )
            )
    
    # Footer da sidebar
    sidebar_footer = html.Div([
        html.Hr(style={"borderColor": "rgba(255,255,255,0.2)"}),
        html.P([
            html.I(className="bi bi-github me-2"),
            html.A("GitHub", href="#", className="text-white-50 small text-decoration-none"),
        ], className="mb-1"),
        html.P([
            html.I(className="bi bi-file-text me-2"),
            html.A("Documentação", href="#", className="text-white-50 small text-decoration-none"),
        ]),
    ], style={"position": "absolute", "bottom": SPACING["lg"], "left": SPACING["lg"], "right": SPACING["lg"]})
    
    return html.Div(
        [sidebar_header, html.Nav(nav_links, className="mb-4"), sidebar_footer],
        style=SIDEBAR_STYLE,
    )


# ================== LAYOUT PRINCIPAL ==================
app.layout = html.Div([
    # Sidebar
    create_sidebar(),
    
    # Container de conteúdo (onde as páginas são renderizadas)
    html.Div([
        # Breadcrumb / Header global (opcional)
        html.Div(
            id="page-header",
            style={
                "backgroundColor": "white",
                "padding": SPACING["sm"],
                "marginBottom": SPACING["md"],
                "borderRadius": "6px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
            }
        ),
        
        # Conteúdo da página atual
        dash.page_container,
        
    ], style=CONTENT_STYLE),
    
], style={"fontFamily": "'Inter', sans-serif"})


# ================== EXECUÇÃO ==================
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
