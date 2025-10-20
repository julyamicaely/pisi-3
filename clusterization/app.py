import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import base64

# === Carregar dados clusterizados ===
df = pd.read_parquet('./clusterization/cardio_2clusters.parquet')

# Calcular BMI
df['bmi_analysis'] = df['weight'] / ((df['height'] / 100) ** 2)

# Função para categorizar risco cardiovascular
def categorize_risk(row):
    risk_factors = 0
    if row['age_years'] > 50: risk_factors += 1
    if row['bmi_analysis'] > 25: risk_factors += 1
    if row['ap_hi'] > 140: risk_factors += 1
    if row['cholesterol'] > 1.5: risk_factors += 1
    if row['smoke'] > 0.5: risk_factors += 1
    if risk_factors >= 3:
        return 'Alto Risco'
    elif risk_factors >= 2:
        return 'Risco Moderado'
    else:
        return 'Baixo Risco'

df['risk_category'] = df.apply(categorize_risk, axis=1)

# Traduções das variáveis
variable_labels = {
    'age_years': 'Idade (anos)',
    'height': 'Altura (cm)',
    'weight': 'Peso (kg)',
    'ap_hi': 'Pressão Sistólica (mmHg)',
    'ap_lo': 'Pressão Diastólica (mmHg)',
    'bmi_analysis': 'IMC',
    'cholesterol': 'Colesterol',
    'gluc': 'Glicose'
}

# === Função para codificar imagem ===
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# === Inicializar app Dash ===
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("🏥 Análise de Clusters - Doença Cardiovascular", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        html.P("Dashboard interativo para análise dos perfis de pacientes clusterizados",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16}),
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),

    # === NOVA SEÇÃO: Imagem do gráfico de análise ===
    html.Div([
        html.H3("Anlisando Número de clusters", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.Img(
            src=encode_image('./clusterization/Cotovelo x Silhouete.png'),
            style={
                'width': '80%',
                'display': 'block',
                'margin-left': 'auto',
                'margin-right': 'auto',
                'border': '2px solid #ddd',
                'borderRadius': '10px'
            }
        ),
        html.P("Gráfico de Cotovelo e Análise de Silhouette para determinação do número ideal de clusters",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 14, 'marginTop': 10})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),

    # === Métricas principais ===
    html.Div([
        html.Div([
            html.H4("📊 Silhouette Score", style={'color': '#2c3e50'}),
            html.H3("0.214", style={'color': '#27ae60', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'padding': '15px', 'flex': 1}),

        html.Div([
            html.H4("👥 Total de Pacientes", style={'color': '#2c3e50'}),
            html.H3(f"{len(df):,}", style={'color': '#2980b9', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'padding': '15px', 'flex': 1}),

        html.Div([
            html.H4("🟠 Cluster 0 (maior risco cardiovascular)", style={'color': '#2c3e50'}),
            html.H3(f"{len(df[df['cluster'] == 0]):,}", style={'color': '#FF7F0E', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'padding': '15px', 'flex': 1}),

        html.Div([
            html.H4("🔵 Cluster 1 (menor risco cardiovascular)", style={'color': '#2c3e50'}),
            html.H3(f"{len(df[df['cluster'] == 1]):,}", style={'color': '#1F77B4', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'padding': '15px', 'flex': 1}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),

    # === Gráfico de barras: tamanho dos clusters ===
    html.Div([
        dcc.Graph(id='cluster-size-plot')
    ], style={'marginBottom': '30px'}),

    # === Estatísticas por cluster (LADO A LADO) ===
    html.Div([
        html.Div([
            html.H4("📊 Estatísticas - Cluster 0 (maior risco cardiovascular)", style={'color': '#FF7F0E', 'marginBottom': '15px'}),
            html.Pre(id='cluster0-stats', style={
                'padding': '15px',
                'backgroundColor': '#fffaf0',
                'height': '300px',
                'overflowY': 'auto',
                'borderRadius': '10px',
                'fontSize': '14px',
                'whiteSpace': 'pre-wrap',
                'lineHeight': '1.5',
                'border': '2px solid #FF7F0E'
            })
        ], style={'flex': 1, 'padding': '10px'}),

        html.Div([
            html.H4("📊 Estatísticas - Cluster 1 (menor risco cardiovascular)", style={'color': '#1F77B4', 'marginBottom': '15px'}),
            html.Pre(id='cluster1-stats', style={
                'padding': '15px',
                'backgroundColor': '#f0f8ff',
                'height': '300px',
                'overflowY': 'auto',
                'borderRadius': '10px',
                'fontSize': '14px',
                'whiteSpace': 'pre-wrap',
                'lineHeight': '1.5',
                'border': '2px solid #1F77B4'
            })
        ], style={'flex': 1, 'padding': '10px'})
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}),

    # === Gráfico PCA (AGORA AQUI) ===
    html.Div([
        dcc.Graph(id='pca-plot')
    ], style={'marginBottom': '30px'}),

    # === Filtros e Dispersão ===
    html.Div([
        html.Div([
            html.Label("🎯 Selecione as Variáveis para Análise:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='x-variable',
                options=[{'label': label, 'value': col} for col, label in variable_labels.items()],
                value='weight',
                style={'marginBottom': '20px'}
            ),
            dcc.Dropdown(
                id='y-variable',
                options=[{'label': label, 'value': col} for col, label in variable_labels.items()],
                value='height',
                style={'marginBottom': '20px'}
            ),
            html.Label("🔍 Filtrar por Cluster:", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='cluster-filter',
                options=[
                    {'label': ' Cluster 0 - maior risco cardiovascular', 'value': 0},
                    {'label': ' Cluster 1 - menor risco cardiovascular', 'value': 1}
                ],
                value=[0, 1]
            ),
        ], style={'flex': 1, 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

        html.Div([
            dcc.Graph(id='scatter-plot', style={'height': '550px'})
        ], style={'flex': 4})
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}),

    # === Boxplots ===
    html.Div([
        html.Div([dcc.Graph(id='distribution-x')], style={'flex': 1}),
        html.Div([dcc.Graph(id='distribution-y')], style={'flex': 1}),
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}),

    # === Heatmap ===
    html.Div([
        dcc.Graph(id='heatmap-plot')
    ], style={'marginBottom': '30px'}),

    # === Fatores de risco ===
    html.Div([
        dcc.Graph(id='risk-factors')
    ], style={'marginBottom': '30px'})

], style={'padding': '20px'})

# === CALLBACK ===
@callback(
    [Output('cluster-size-plot', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('distribution-x', 'figure'),
     Output('distribution-y', 'figure'),
     Output('heatmap-plot', 'figure'),
     Output('risk-factors', 'figure'),
     Output('pca-plot', 'figure'),
     Output('cluster0-stats', 'children'),
     Output('cluster1-stats', 'children')],
    [Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('cluster-filter', 'value')]
)
def update_dashboard(x_var, y_var, selected_clusters):
    filtered_df = df[df['cluster'].isin(selected_clusters)]
    
    # CORES: Laranja maior risco cardiovascular (Cluster 0), Azul menor risco cardiovascular (Cluster 1)
    colors = {0: '#FF7F0E', 1: '#1F77B4'}

    # === Gráfico de barras (só porcentagem) ===
    cluster_counts = filtered_df['cluster'].value_counts().sort_index()
    fig_size = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        text=[f"{v/cluster_counts.sum()*100:.1f}%" for v in cluster_counts.values],
        color=cluster_counts.index.map({0: 'Cluster 0', 1: 'Cluster 1'}),
        color_discrete_map={'Cluster 0': colors[0], 'Cluster 1': colors[1]},
        title='👥 Número de Pacientes por Cluster'
    )
    fig_size.update_traces(textposition='outside')
    fig_size.update_layout(xaxis_title='Cluster', yaxis_title='', showlegend=False)

    # === Dispersão ===
    fig_scatter = px.scatter(
        filtered_df, x=x_var, y=y_var, color='cluster',
        title=f'📈 Dispersão: {variable_labels[x_var]} vs {variable_labels[y_var]}',
        color_discrete_map=colors, opacity=0.6,
        labels={x_var: variable_labels[x_var], y_var: variable_labels[y_var]}  # RÓTULOS NOS EIXOS
    )

    # === Boxplots ===
    fig_dist_x = px.box(filtered_df, x='cluster', y=x_var, color='cluster',
                        title=f'📦 Distribuição de {variable_labels[x_var]} por Cluster',
                        color_discrete_map=colors, points='outliers',
                        labels={x_var: variable_labels[x_var], 'cluster': 'Cluster'})  # RÓTULOS NOS EIXOS
    
    fig_dist_y = px.box(filtered_df, x='cluster', y=y_var, color='cluster',
                        title=f'📦 Distribuição de {variable_labels[y_var]} por Cluster',
                        color_discrete_map=colors, points='outliers',
                        labels={y_var: variable_labels[y_var], 'cluster': 'Cluster'})  # RÓTULOS NOS EIXOS

    # === Heatmap ===
    numeric_vars = list(variable_labels.keys())
    corr = filtered_df[numeric_vars].corr().round(2)
    
    # Criar labels traduzidos para o heatmap
    translated_labels = [variable_labels[var] for var in numeric_vars]
    
    fig_heat = px.imshow(corr, 
                         x=translated_labels,  # EIXO X TRADUZIDO
                         y=translated_labels,  # EIXO Y TRADUZIDO
                         text_auto=True, 
                         color_continuous_scale='RdBu_r',
                         title='🔥 Correlação entre Variáveis',
                         labels={'color': 'Correlação'})

    # === Fatores de risco ===
    risk_factors_data = []
    for c in selected_clusters:
        data = filtered_df[filtered_df['cluster'] == c]
        risk_factors_data.append({
            'Cluster': f'Cluster {c}',
            'Hipertensão': (data['ap_hi'] > 140).mean() * 100,
            'Obesidade (BMI>30)': (data['bmi_analysis'] > 30).mean() * 100,
            'Colesterol Alto': (data['cholesterol'] > 1.5).mean() * 100,
            'Fumantes': (data['smoke'] == 1).mean() * 100,
            'Idade > 50': (data['age_years'] > 50).mean() * 100
        })
    risk_df = pd.DataFrame(risk_factors_data)
    fig_risk = px.bar(risk_df, x='Cluster', y=risk_df.columns[1:], barmode='group',
                      title='⚠️ Fatores de Risco por Cluster', text_auto='.1f',
                      labels={'value': 'Porcentagem (%)', 'variable': 'Fator de Risco'})

    # === PCA ===
    pca_vars = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
    scaled = StandardScaler().fit_transform(filtered_df[pca_vars])
    pca = PCA(n_components=2)
    comp = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(comp, columns=['PC1', 'PC2'])
    pca_df['cluster'] = filtered_df['cluster']
    
    # PCA COM AS MESMAS CORES PADRONIZADAS
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='cluster',
                         title=f'🔍 PCA - {pca.explained_variance_ratio_.sum():.1%} Variância Explicada',
                         color_discrete_map=colors,
                         labels={'PC1': 'Componente Principal 1', 
                                 'PC2': 'Componente Principal 2',
                                 'cluster': 'Cluster'})

    # === Estatísticas por cluster (SEPARADAS) ===
    stats0 = ""
    stats1 = ""
    
    for c in selected_clusters:
        d = filtered_df[filtered_df['cluster'] == c]
        
        # Calcular distribuição de gênero
        women_count = (d['gender'] == 1).sum()
        men_count = (d['gender'] == 2).sum()
        total_count = len(d)
        women_percentage = (women_count / total_count) * 100
        men_percentage = (men_count / total_count) * 100
        
        stats = f"👩 Mulheres: {women_percentage:.1f}%\n"
        stats += f"👨 Homens: {men_percentage:.1f}%\n"
        stats += f"👵 Idade média: {d['age_years'].mean():.1f} anos\n"
        stats += f"📏 Altura média: {d['height'].mean():.1f} cm\n"
        stats += f"⚖️ Peso médio: {d['weight'].mean():.1f} kg\n"
        stats += f"🏷️ BMI médio: {d['bmi_analysis'].mean():.1f}\n"
        stats += f"💓 Pressão: {d['ap_hi'].mean():.1f}/{d['ap_lo'].mean():.1f} mmHg\n"
        stats += f"🩸 Colesterol: {d['cholesterol'].mean():.2f}\n"
        stats += f"🍭 Glicose: {d['gluc'].mean():.2f}\n"
        stats += f"🚬 Fumantes: {d['smoke'].mean()*100:.1f}%\n"
        stats += f"🍺 Consumo de álcool: {d['alco'].mean()*100:.1f}%\n"
        stats += f"🏃 Ativos: {d['active'].mean()*100:.1f}%\n"
        
        if c == 0:
            stats0 = stats
        else:
            stats1 = stats

    return fig_size, fig_scatter, fig_dist_x, fig_dist_y, fig_heat, fig_risk, fig_pca, stats0, stats1


if __name__ == '__main__':
    app.run(debug=True, port=8050)