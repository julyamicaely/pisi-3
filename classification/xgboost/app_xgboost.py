import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output

# =======================================================
# 1. Carregamento e pré-processamento
# =======================================================
DATA_PATH = "../../EDA/cardio_data.parquet"

def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna()

    # Calcula idade em anos, se necessário
    if "age" in df.columns and "age_years" not in df.columns:
        df["age_years"] = (df["age"] / 365).round(1)

    # Mantém apenas as variáveis mais relevantes
    cols_to_keep = [
        "age_years", "ap_hi", "ap_lo", "cholesterol", "gluc",
        "height", "weight", "gender", "active", "bp_category_encoded", "cardio"
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Converte variáveis categóricas
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


# =======================================================
# 2. Treinamento do modelo XGBoost
# =======================================================
def train_xgboost():
    df = load_data()
    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        enable_categorical=True
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    feature_importances = model.feature_importances_
    feature_names = X.columns

    return model, X_test, y_test, y_pred, cm, report, feature_importances, feature_names


# =======================================================
# 3. Treina o modelo uma vez
# =======================================================
model, X_test, y_test, y_pred, cm, report, feature_importances, feature_names = train_xgboost()


# =======================================================
# 4. Cria o app Dash
# =======================================================
app = Dash(__name__)
app.title = "Análise de Fatores de Risco Cardiovascular — XGBoost"

# DataFrame de importância
importance_df = pd.DataFrame({
    "Variável": feature_names,
    "Importância relativa": feature_importances
}).sort_values(by="Importância relativa", ascending=False)

# Tabela de métricas
metrics_data = {
    "Métrica": ["Acurácia", "Precisão (Classe 0)", "Precisão (Classe 1)",
                "Recall (Classe 0)", "Recall (Classe 1)",
                "F1-Score (Classe 0)", "F1-Score (Classe 1)"],
    "Valor": [
        accuracy_score(y_test, y_pred),
        report["0"]["precision"],
        report["1"]["precision"],
        report["0"]["recall"],
        report["1"]["recall"],
        report["0"]["f1-score"],
        report["1"]["f1-score"]
    ]
}
df_metrics = pd.DataFrame(metrics_data)

# =======================================================
# 5. Layout principal
# =======================================================
app.layout = html.Div([
    html.H1(
        "Análise de Fatores de Risco Cardiovascular com XGBoost",
        style={"textAlign": "center", "marginBottom": "20px"}
    ),

    html.H3("Selecione variáveis para visualizar"),
    dcc.Dropdown(
        id="var-selector",
        options=[{"label": var, "value": var} for var in importance_df["Variável"]],
        value=importance_df["Variável"].tolist(),  # todas selecionadas por padrão
        multi=True,
        style={"width": "70%", "margin": "auto", "marginBottom": "20px"}
    ),

    dcc.Graph(id="importance-graph"),

    html.P(
        "A análise da importância das variáveis no modelo XGBoost indica que "
        "as variáveis **Pressão Sistólica (ap_hi)**, **Colesterol** e **Idade (anos)** "
        "foram as que mais contribuíram para a predição do risco cardiovascular. "
        "Esses resultados demonstram que valores mais elevados nessas variáveis "
        "tendem a aumentar a probabilidade prevista de ocorrência de doenças cardíacas. "
        "O comportamento observado está em concordância com achados clínicos que "
        "associam pressão arterial elevada, colesterol alto e idade avançada "
        "a um risco aumentado de eventos cardiovasculares.",
        style={"textAlign": "justify", "marginBottom": "40px", "fontSize": "15px"}
    ),

    html.H3("Matriz de Confusão"),
    dcc.Graph(id="confusion-matrix"),

    html.H3("Métricas de Desempenho do Modelo"),
    dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_metrics.columns],
        data=df_metrics.round(3).to_dict("records"),
        style_table={"width": "60%", "margin": "auto"},
        style_cell={"textAlign": "center", "padding": "8px"},
        style_header={"backgroundColor": "#001f3f", "color": "white", "fontWeight": "bold"},
        style_data={"backgroundColor": "#f9f9f9"}
    )
])


# =======================================================
# 6. Callbacks para interatividade
# =======================================================
@app.callback(
    Output("importance-graph", "figure"),
    Output("confusion-matrix", "figure"),
    Input("var-selector", "value")
)
def update_graphs(selected_vars):
    filtered_df = importance_df[importance_df["Variável"].isin(selected_vars)]

    fig_importance = px.bar(
        filtered_df,
        x="Importância relativa",
        y="Variável",
        orientation="h",
        color="Importância relativa",
        color_continuous_scale=["#f4a261", "#001f3f"],
        title="Importância das variáveis no modelo XGBoost"
    )
    fig_importance.update_layout(
        yaxis=dict(autorange="reversed"),
        title_x=0.5,
        font=dict(size=14)
    )

    # Matriz de confusão
    cm_labels = ["Sem Doença", "Com Doença"]
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        x=cm_labels,
        y=cm_labels,
        color_continuous_scale="Blues",
        title="Matriz de Confusão — Desempenho do Modelo"
    )
    fig_cm.update_layout(title_x=0.5, font=dict(size=14))

    return fig_importance, fig_cm


# =======================================================
# 7. Executa o app
# =======================================================
if __name__ == "__main__":
    app.run(debug=True, port=8052)
