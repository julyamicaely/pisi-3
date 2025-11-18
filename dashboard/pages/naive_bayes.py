"""
Dash Page: Naive Bayes Classifier Results
"""
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
from pathlib import Path
import base64
import re 
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from styles import PALETTE

# Register page
dash.register_page(
    __name__,
    path="/naive-bayes",
    name="Naive Bayes",
    title="Naive Bayes - Dashboard",
    icon="bi-clipboard-data-fill"
)

# Paths to artifacts
BASE_PATH = Path(__file__).parent.parent.parent
CLASSIFICATION_PATH = BASE_PATH / "classification"
REPORTS_PATH = CLASSIFICATION_PATH / "reports"
RESULTS_PATH = CLASSIFICATION_PATH / "results"
MODELS_PATH = CLASSIFICATION_PATH / "models"
SCALERS_PATH = CLASSIFICATION_PATH / "scalers"

# --- Helper Functions ---

def create_metric_card(title, value, is_highlighted=False):
    """Creates a card for displaying a metric."""
    card_color = "primary" if is_highlighted else "light"
    return dbc.Card(
        dbc.CardBody([
            html.H4(title, className="card-title"),
            html.H2(value, className="card-text"),
        ]),
        color=card_color,
        inverse=is_highlighted,
        className="text-center m-2"
    )

def read_report_metrics():
    """Lê métricas do arquivo de relatório."""
    metrics = {"Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A", "F1-Score": "N/A", "report": "Relatório não encontrado."}
    try:
        with open(REPORTS_PATH / "naive_bayes_report.txt", "r") as f:
            content = f.read()
            metrics["report"] = content
            
            # Use regex to find the values
            accuracy_match = re.search(r"Acurácia: ([\d.]+)", content)
            precision_match = re.search(r"Precisão: ([\d.]+)", content)
            recall_match = re.search(r"Recall: ([\d.]+)", content)
            f1_match = re.search(r"F1-Score: ([\d.]+)", content)

            if accuracy_match:
                metrics["Accuracy"] = f"{float(accuracy_match.group(1)) * 100:.2f}%"
            if precision_match:
                metrics["Precision"] = f"{float(precision_match.group(1)) * 100:.2f}%"
            if recall_match:
                metrics["Recall"] = f"{float(recall_match.group(1)) * 100:.2f}%"
            if f1_match:
                metrics["F1-Score"] = f"{float(f1_match.group(1)) * 100:.2f}%"

    except FileNotFoundError:
        print("Arquivo de relatório não encontrado.")
    return metrics

def encode_image(image_path):
    """Encodes an image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return "data:image/png;base64," + base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return None

# Load metrics and images
metrics = read_report_metrics()
confusion_matrix_img = encode_image(RESULTS_PATH / "naive_bayes_confusion_matrix.png")
roc_curve_img = encode_image(RESULTS_PATH / "naive_bayes_roc_curve.png")
pr_curve_img = encode_image(RESULTS_PATH / "naive_bayes_pr_curve.png")

# Define continuous features for distribution plots
continuous_features = ['age_years', 'ap_hi', 'ap_lo', 'bmi']

# --- Page Components ---

# --- Live Prediction Components ---
prediction_card = dbc.Card(
    dbc.CardBody([
        html.H4("🔮 Predição em Tempo Real", className="card-title text-center mb-4"),
        dbc.Row([
            # Column 1: Continuous Features
            dbc.Col([
                dbc.Label("Idade (Anos)", html_for="nb-age-years"),
                dbc.Input(id="nb-age-years", type="number", placeholder="ex.: 50", min=3, max=120),
                html.Br(),
                dbc.Label("Altura (cm)", html_for="nb-height"),
                dbc.Input(id="nb-height", type="number", placeholder="ex.: 168", min=30, max=220),
                html.Br(),
                dbc.Label("Peso (kg)", html_for="nb-weight"),
                dbc.Input(id="nb-weight", type="number", placeholder="ex.: 70.0", min=5, max=400, step=0.1),
                html.Br(),
                dbc.Label("Pressão Sistólica (ap_hi)", html_for="nb-ap-hi"),
                dbc.Input(id="nb-ap-hi", type="number", placeholder="ex.: 120", min=60, max=240),
                html.Br(),
                dbc.Label("Pressão Diastólica (ap_lo)", html_for="nb-ap-lo"),
                dbc.Input(id="nb-ap-lo", type="number", placeholder="ex.: 80", min=40, max=180),
            ], md=6),
            # Column 2: Categorical Features
            dbc.Col([
                dbc.Label("Gênero", html_for="nb-gender"),
                dcc.Dropdown(id="nb-gender", options=[{'label': 'Feminino', 'value': 0}, {'label': 'Masculino', 'value': 1}], placeholder="Selecione..."),
                html.Br(),
                dbc.Label("Colesterol", html_for="nb-cholesterol"),
                dcc.Dropdown(id="nb-cholesterol", options=[
                    {'label': 'Normal', 'value': 1},
                    {'label': 'Acima do Normal', 'value': 2},
                    {'label': 'Muito Acima do Normal', 'value': 3}
                ], placeholder="Selecione..."),
                html.Br(),
                dbc.Label("Glicose", html_for="nb-gluc"),
                dcc.Dropdown(id="nb-gluc", options=[
                    {'label': 'Normal', 'value': 1},
                    {'label': 'Acima do Normal', 'value': 2},
                    {'label': 'Muito Acima do Normal', 'value': 3}
                ], placeholder="Selecione..."),
                html.Br(),
                dbc.Label("Fumante?", html_for="nb-smoke"),
                dcc.Dropdown(id="nb-smoke", options=[{'label': 'Não', 'value': 0}, {'label': 'Sim', 'value': 1}], placeholder="Selecione..."),
                html.Br(),
                dbc.Label("Bebe Álcool?", html_for="nb-alco"),
                dcc.Dropdown(id="nb-alco", options=[{'label': 'Não', 'value': 0}, {'label': 'Sim', 'value': 1}], placeholder="Selecione..."),
                html.Br(),
                dbc.Label("Ativo Fisicamente?", html_for="nb-active"),
                dcc.Dropdown(id="nb-active", options=[{'label': 'Não', 'value': 0}, {'label': 'Sim', 'value': 1}], placeholder="Selecione..."),
            ], md=6),
        ]),
        html.Div(
            dbc.Button("Obter Predição", id="nb-predict-button", color="primary", n_clicks=0, className="mt-4 w-100"),
            className="d-grid gap-2",
        ),
        html.Div(id="nb-prediction-output", className="mt-4 text-center fs-4"),
    ]),
    className="mb-4"
)

# --- Layout ---
layout = dbc.Container([
    # ========== HERO SECTION ==========
    html.Div([
        html.Div([
            html.H1("📊 Classificador Naive Bayes",
                   className="display-4 fw-bold text-white mb-3"),
            html.P("Classificação Probabilística para Risco Cardiovascular",
                  className="lead text-white-50 mb-4"),
            dbc.Row([
                dbc.Col(create_metric_card("Precisão", metrics["Precision"], is_highlighted=True)),
                dbc.Col(create_metric_card("Acurácia", metrics["Accuracy"])),
                dbc.Col(create_metric_card("Recall", metrics["Recall"])),
                dbc.Col(create_metric_card("F1-Score", metrics["F1-Score"])),
            ])
        ], className="container py-5")
    ], style={
        "background": f"linear-gradient(135deg, {PALETTE['info']} 0%, {PALETTE['secondary']} 100%)",
        "marginBottom": "40px",
        "borderRadius": "0 0 30px 30px",
        "boxShadow": "0 10px 40px rgba(0,0,0,0.2)"
    }),

    # Live Prediction
    prediction_card,

    # Visualizations
    html.H3("Visualizações de Desempenho do Modelo", className="my-4 text-center"),
    dbc.Tabs([
        dbc.Tab(label="Matriz de Confusão", children=[
            html.Img(src=confusion_matrix_img, style={'width': '100%', 'max-width': '600px', 'margin': 'auto', 'display': 'block'})
        ]),
        dbc.Tab(label="Curva ROC", children=[
            html.Img(src=roc_curve_img, style={'width': '100%', 'max-width': '600px', 'margin': 'auto', 'display': 'block'})
        ]),
        dbc.Tab(label="Curva Precisão-Recall", children=[
            html.Img(src=pr_curve_img, style={'width': '100%', 'max-width': '600px', 'margin': 'auto', 'display': 'block'})
        ]),
        dbc.Tab(label="Distribuições de Características", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Selecione Característica:", className="fw-bold"),
                    dcc.Dropdown(
                        id='dist-feature-dropdown',
                        options=[{'label': feat.replace("_", " ").title(), 'value': feat} for feat in continuous_features],
                        value=continuous_features[0],
                        clearable=False
                    ),
                ], width=6, className="mx-auto mt-4")
            ]),
            html.Img(id='dist-plot-img', style={'width': '100%', 'max-width': '800px', 'margin': 'auto', 'display': 'block', 'marginTop': '20px'})
        ]),
    ]),

    html.Hr(className="my-5"),

    # Classification Report
    html.H3("Relatório de Classificação", className="my-4"),
    dbc.Card(
        dbc.CardBody([
            dcc.Markdown(f"```\n{metrics['report']}\n```")
        ])
    )
], fluid=True)

# --- Callbacks ---
@callback(
    Output('nb-prediction-output', 'children'),
    Input('nb-predict-button', 'n_clicks'),
    [State('nb-age-years', 'value'),
     State('nb-height', 'value'),
     State('nb-weight', 'value'),
     State('nb-ap-hi', 'value'),
     State('nb-ap-lo', 'value'),
     State('nb-gender', 'value'),
     State('nb-cholesterol', 'value'),
     State('nb-gluc', 'value'),
     State('nb-smoke', 'value'),
     State('nb-alco', 'value'),
     State('nb-active', 'value')]
)
def predict_live(n_clicks, age_years, height, weight, ap_hi, ap_lo, gender, cholesterol, gluc, smoke, alco, active):
    if n_clicks == 0:
        return ""

    # Validate that all fields are filled
    if any(v is None for v in [age_years, height, weight, ap_hi, ap_lo, gender, cholesterol, gluc, smoke, alco, active]):
        return dbc.Alert("Por favor, preencha todos os campos para obter uma predição.", color="warning")

    try:
        # Load the entire pipeline (preprocessor + model)
        pipeline = joblib.load(MODELS_PATH / "naive_bayes_pipeline.joblib")

        # Feature Engineering: Calculate BMI from user inputs
        height_m = height / 100 if height is not None and height > 0 else 0
        bmi = weight / (height_m ** 2) if height_m > 0 and weight is not None else 0

        # Create a DataFrame with the user input in the correct order.
        # The columns must match the features the model was trained on.
        feature_order = ['age_years', 'ap_hi', 'ap_lo', 'bmi', 
                         'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
        input_data = pd.DataFrame([[
            age_years, ap_hi, ap_lo, bmi,
            gender, cholesterol, gluc, smoke, alco, active
        ]], columns=feature_order)

        # Make prediction using the full pipeline
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0]

        # Display result
        if prediction == 1:
            result_text = "Alto Risco de Doença Cardiovascular"
            result_color = "danger"
            prob_value = f"{probability[1]*100:.2f}%"
        else:
            result_text = "Baixo Risco de Doença Cardiovascular"
            result_color = "success"
            prob_value = f"{probability[0]*100:.2f}%"

        return dbc.Alert([
            html.H5(result_text, className="alert-heading"),
            html.P(f"Confiança: {prob_value}")
        ], color=result_color)

    except Exception as e:
        return dbc.Alert(f"Falha na predição: {e}", color="danger")

@callback(
    Output('dist-plot-img', 'src'),
    Input('dist-feature-dropdown', 'value')
)
def update_dist_plot(feature):
    """Updates the feature distribution plot based on dropdown selection."""
    if not feature:
        return ""
    
    # Construct the path to the pre-generated image
    image_path = CLASSIFICATION_PATH / "results" / "distributions" / f"dist_{feature}.png"
    
    # Encode the image to base64 to display it in the browser
    encoded_image = encode_image(image_path)
    
    if encoded_image:
        return encoded_image
    
    # Return a placeholder or empty string if the image is not found
    return ""