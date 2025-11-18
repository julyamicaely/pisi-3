# app_xgboost_clean.py
import dash
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import os
import glob
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import base64

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def load_resources():
    base_path = Path(__file__).parent
    
    resources = {
        "scripts": [
            {"name": "model_xgboost_simple.py", "description": "Treinar modelo XGBoost"},
            {"name": "evaluation_xgb.py", "description": "Avaliar modelo"}, 
            {"name": "feature_importance_simple.py", "description": "Analisar importancia de features"}
        ],
        "reports": [],
        "images": []
    }
    
    # Carregar relatórios
    report_files = list(base_path.glob("*report*.txt"))
    report_files.sort(key=os.path.getmtime, reverse=True)
    
    for report in report_files:
        try:
            with open(report, 'r', encoding='utf-8') as f:
                content = f.read()
                resources["reports"].append({
                    "name": report.name,
                    "content": content,
                    "modified": datetime.fromtimestamp(os.path.getmtime(report))
                })
        except Exception as e:
            print(f"Erro ao ler {report}: {e}")
    
    # Carregar imagens
    image_files = list(base_path.glob("*.png"))
    for img in image_files:
        resources["images"].append({
            "name": img.name,
            "path": str(img),
            "modified": datetime.fromtimestamp(os.path.getmtime(img))
        })
    
    return resources

def create_script_card(script_info):
    return dbc.Card([
        dbc.CardHeader(html.Strong(script_info["name"])),
        dbc.CardBody([
            html.P(script_info["description"], className="text-muted small"),
            dbc.Button(
                "Executar",
                id=f"run-{script_info['name'].replace('.', '-')}",
                color="primary",
                size="sm",
                className="w-100"
            ),
            html.Div(id=f"output-{script_info['name'].replace('.', '-')}", className="mt-2")
        ])
    ], className="h-100")

def create_report_card(report):
    content = report['content'] if report['content'].strip() else "Relatório vazio"
    
    return dbc.Card([
        dbc.CardHeader([
            html.Strong(report['name']),
            html.Small(f" ({report['modified'].strftime('%d/%m %H:%M')})", className="text-muted ms-2")
        ]),
        dbc.CardBody([
            html.Pre(content, style={
                "backgroundColor": "#f8f9fa",
                "padding": "10px",
                "borderRadius": "5px",
                "fontSize": "11px",
                "maxHeight": "200px",
                "overflowY": "auto",
                "whiteSpace": "pre-wrap"
            })
        ])
    ])

def encode_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('ascii')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return None

def create_image_card(image_info):
    encoded_image = encode_image(image_info['path'])
    
    if not encoded_image:
        return dbc.Card([dbc.CardBody("Erro ao carregar imagem")])
    
    return dbc.Card([
        dbc.CardHeader(html.Strong(image_info['name'])),
        dbc.CardBody([
            html.Img(src=encoded_image, style={"width": "100%", "height": "auto"}),
            html.Small(f"Gerado: {image_info['modified'].strftime('%d/%m %H:%M')}", className="text-muted")
        ])
    ])

# Layout principal
resources = load_resources()

app.layout = dbc.Container([
    html.H2("XGBoost Dashboard", className="my-4 text-center"),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H4("Scripts", className="mb-3"),
            *[create_script_card(script) for script in resources["scripts"]]
        ]), md=4),
        
        dbc.Col(html.Div([
            html.H4("Relatórios", className="mb-3"),
            *[create_report_card(report) for report in resources["reports"]],
            html.Div("Nenhum relatório disponível", className="text-muted text-center py-4") 
            if not resources["reports"] else None
        ]), md=4),
        
        dbc.Col(html.Div([
            html.H4("Visualizações", className="mb-3"), 
            *[create_image_card(img) for img in resources["images"]],
            html.Div("Nenhuma imagem disponível", className="text-muted text-center py-4")
            if not resources["images"] else None
        ]), md=4),
    ]),
    
    html.Hr(),
    html.Div([
        html.Small("Execute os scripts na ordem: Treinamento → Avaliação → Features", 
                  className="text-muted")
    ], className="text-center mt-4")
    
], fluid=True, style={"padding": "20px"})

# Callbacks
def create_callback(script_name):
    @callback(
        Output(f"output-{script_name.replace('.', '-')}", "children"),
        Input(f"run-{script_name.replace('.', '-')}", "n_clicks"),
        prevent_initial_call=True
    )
    def run_script(n_clicks):
        if not n_clicks:
            return ""
        
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=Path(__file__).parent,
                timeout=120,
                env=env
            )
            
            output_text = ""
            if result.stdout:
                output_text += f"Saída:\n{result.stdout}\n"
            if result.stderr:
                output_text += f"Erros:\n{result.stderr}\n"
            
            if result.returncode == 0:
                return dbc.Alert([
                    html.Strong("Sucesso"),
                    html.Pre(output_text, style={"fontSize": "10px", "maxHeight": "150px", "overflowY": "auto"})
                ], color="success", className="p-2")
            else:
                return dbc.Alert([
                    html.Strong("Erro na execução"),
                    html.Pre(output_text, style={"fontSize": "10px", "maxHeight": "150px", "overflowY": "auto"})
                ], color="danger", className="p-2")
                
        except subprocess.TimeoutExpired:
            return dbc.Alert("Timeout - script demorou muito", color="warning")
        except Exception as e:
            return dbc.Alert(f"Erro: {str(e)}", color="danger")
    
    return run_script

for script in resources["scripts"]:
    create_callback(script["name"])

if __name__ == "__main__":
    print("Dashboard XGBoost iniciando...")
    print("Acesse: http://127.0.0.1:8055")
    app.run(debug=True, host="127.0.0.1", port=8055)