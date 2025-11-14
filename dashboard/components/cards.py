"""
Componentes de visualização reutilizáveis (gráficos, tabelas, cards).
Funções puras que encapsulam lógica de apresentação de dados.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from styles import PALETTE, SPACING
from components.utils import make_card, make_empty_state
import pandas as pd
import numpy as np


def build_confusion_matrix(y_true, y_pred, labels=None):
    """
    Cria heatmap de matriz de confusão.
    
    Args:
        y_true (array): Valores reais
        y_pred (array): Valores preditos
        labels (list): Labels das classes
    
    Returns:
        dcc.Graph: Gráfico da matriz de confusão
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f"Classe {i}" for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predito",
        yaxis_title="Real",
        height=400,
    )
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def build_feature_importance(feature_names, importance_values, top_n=15):
    """
    Cria gráfico de barras horizontal com importância de features.
    
    Args:
        feature_names (list): Nomes das features
        importance_values (array): Valores de importância
        top_n (int): Número de features a mostrar
    
    Returns:
        dcc.Graph: Gráfico de importância
    """
    # Criar DataFrame e ordenar
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values
    }).sort_values("importance", ascending=True).tail(top_n)
    
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color=PALETTE["accent"],
        text=df["importance"].round(3),
        textposition="outside",
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Features Mais Importantes",
        xaxis_title="Importância",
        yaxis_title="",
        height=max(400, top_n * 25),
        showlegend=False,
    )
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def build_metrics_table(metrics_dict):
    """
    Cria tabela estilizada de métricas.
    
    Args:
        metrics_dict (dict): Dicionário com métricas {nome: valor}
    
    Returns:
        dbc.Table: Tabela de métricas
    """
    if not metrics_dict:
        return make_empty_state("Nenhuma métrica disponível")
    
    rows = [
        html.Tr([
            html.Td(name, style={"fontWeight": "600"}),
            html.Td(f"{value:.4f}" if isinstance(value, float) else str(value), style={"textAlign": "right"}),
        ])
        for name, value in metrics_dict.items()
    ]
    
    return dbc.Table(
        [html.Tbody(rows)],
        bordered=True,
        hover=True,
        responsive=True,
        className="mb-0",
    )


def build_classification_report_card(report_text):
    """
    Cria card com relatório de classificação formatado.
    
    Args:
        report_text (str): Texto do classification_report
    
    Returns:
        dbc.Card: Card com relatório
    """
    return make_card(
        "Relatório de Classificação",
        html.Pre(
            report_text,
            style={
                "backgroundColor": "#F5F7FA",
                "padding": SPACING["md"],
                "borderRadius": "4px",
                "fontSize": "13px",
                "fontFamily": "'Fira Code', monospace",
                "overflow": "auto",
                "maxHeight": "400px",
            }
        ),
        icon="file-text"
    )


def build_scatter_plot(df, x_col, y_col, color_col=None, title=""):
    """
    Cria gráfico de dispersão.
    
    Args:
        df (DataFrame): Dados
        x_col (str): Coluna do eixo X
        y_col (str): Coluna do eixo Y
        color_col (str): Coluna para colorir pontos
        title (str): Título do gráfico
    
    Returns:
        dcc.Graph: Gráfico de dispersão
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        height=500,
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def build_histogram(data, title="Distribuição", bins=30, color=None):
    """
    Cria histograma.
    
    Args:
        data (array): Dados para histograma
        title (str): Título
        bins (int): Número de bins
        color (str): Cor das barras
    
    Returns:
        dcc.Graph: Histograma
    """
    if color is None:
        color = PALETTE["primary"]
    
    fig = go.Figure(go.Histogram(
        x=data,
        nbinsx=bins,
        marker_color=color,
        opacity=0.75,
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Valor",
        yaxis_title="Frequência",
        height=400,
        showlegend=False,
    )
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def build_box_plot(df, y_col, x_col=None, title=""):
    """
    Cria box plot.
    
    Args:
        df (DataFrame): Dados
        y_col (str): Coluna para eixo Y
        x_col (str): Coluna para agrupar (opcional)
        title (str): Título
    
    Returns:
        dcc.Graph: Box plot
    """
    fig = px.box(
        df,
        y=y_col,
        x=x_col,
        title=title,
        height=450,
    )
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def build_pie_chart(labels, values, title=""):
    """
    Cria gráfico de pizza.
    
    Args:
        labels (list): Labels das fatias
        values (list): Valores das fatias
        title (str): Título
    
    Returns:
        dcc.Graph: Gráfico de pizza
    """
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=[PALETTE["primary"], PALETTE["accent"], PALETTE["info"], PALETTE["success"]]),
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def build_line_chart(df, x_col, y_col, title=""):
    """
    Cria gráfico de linha.
    
    Args:
        df (DataFrame): Dados
        x_col (str): Coluna eixo X
        y_col (str): Coluna eixo Y
        title (str): Título
    
    Returns:
        dcc.Graph: Gráfico de linha
    """
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        height=400,
    )
    
    fig.update_traces(line_color=PALETTE["accent"], line_width=3)
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def build_roc_curve(y_true, y_proba, title="ROC Curve", show_auc=True):
    """
    Cria curva ROC (Receiver Operating Characteristic) reutilizável.
    
    Args:
        y_true (array): Labels verdadeiros
        y_proba (array): Probabilidades da classe positiva
        title (str): Título do gráfico
        show_auc (bool): Exibir AUC no título
    
    Returns:
        dcc.Graph: Gráfico da curva ROC
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # Curva ROC
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})' if show_auc else 'ROC',
        line=dict(color=PALETTE["primary"], width=3),
        fill='tozeroy',
        fillcolor='rgba(30, 136, 229, 0.2)',
        hovertemplate='<b>FPR:</b> %{x:.3f}<br><b>TPR:</b> %{y:.3f}<extra></extra>'
    ))
    
    # Linha diagonal (baseline - classificador aleatório)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Baseline (Random)',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=True,
        hovertemplate='Classificador Aleatório<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>AUC: {roc_auc:.3f} - Quanto mais próximo de 1.0, melhor</sub>" if show_auc else title,
            font=dict(size=18)
        ),
        xaxis_title='Taxa de Falsos Positivos (FPR)',
        yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='closest'
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True})


def build_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    """
    Cria curva Precision-Recall reutilizável.
    
    Args:
        y_true (array): Labels verdadeiros
        y_proba (array): Probabilidades da classe positiva
        title (str): Título do gráfico
    
    Returns:
        dcc.Graph: Gráfico da curva Precision-Recall
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR (AP = {avg_precision:.3f})',
        line=dict(color=PALETTE["accent"], width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 172, 193, 0.2)',
        hovertemplate='<b>Recall:</b> %{x:.3f}<br><b>Precision:</b> %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Average Precision: {avg_precision:.3f} - Útil para datasets desbalanceados</sub>",
            font=dict(size=18)
        ),
        xaxis_title='Recall (Sensibilidade)',
        yaxis_title='Precision (Valor Preditivo Positivo)',
        height=450,
        hovermode='closest'
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True})


def build_calibration_curve(y_true, y_proba, n_bins=10, title="Calibration Curve"):
    """
    Cria curva de calibração com Brier Score para avaliar se as probabilidades
    previstas pelo modelo refletem as frequências reais.
    
    Args:
        y_true (array): Labels verdadeiros
        y_proba (array): Probabilidades da classe positiva
        n_bins (int): Número de bins para calibração
        title (str): Título do gráfico
    
    Returns:
        dcc.Graph: Gráfico da curva de calibração
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    
    # Calcular Brier Score (quanto menor, melhor - 0 é perfeito)
    brier_score = brier_score_loss(y_true, y_proba)
    
    # Calcular curva de calibração
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    
    fig = go.Figure()
    
    # Curva de calibração
    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode='lines+markers',
        name=f'Modelo (Brier: {brier_score:.4f})',
        line=dict(color=PALETTE["primary"], width=3),
        marker=dict(size=10, line=dict(width=2, color='white')),
        hovertemplate='<b>Prob. Prevista:</b> %{x:.3f}<br><b>Fração Positivos:</b> %{y:.3f}<extra></extra>'
    ))
    
    # Linha de calibração perfeita
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfeitamente Calibrado',
        line=dict(color='gray', dash='dash', width=2),
        hovertemplate='Calibração Perfeita<extra></extra>'
    ))
    
    # Interpretação do Brier Score
    if brier_score < 0.1:
        interpretation = "✅ Excelente calibração"
        color = "green"
    elif brier_score < 0.2:
        interpretation = "✓ Boa calibração"
        color = "orange"
    else:
        interpretation = "⚠️ Calibração pode ser melhorada"
        color = "red"
    
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Brier Score: {brier_score:.4f} - <span style='color:{color}'>{interpretation}</span></sub>",
            font=dict(size=18)
        ),
        xaxis_title='Probabilidade Prevista (Média por Bin)',
        yaxis_title='Fração de Positivos (Real)',
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='closest',
        annotations=[
            dict(
                text="Se o modelo está bem calibrado, os pontos devem ficar próximos da linha diagonal",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=11, color="gray"),
                xanchor='center'
            )
        ]
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True})
