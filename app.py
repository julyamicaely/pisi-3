# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from io import StringIO

# ====== Template (evita bug de template em algumas versões) ======
pio.templates.default = "plotly_white"

# ====== Carregamento e preparação dos dados ======
df = pd.read_csv('cardio_data.csv')

# Idade em anos (era em dias)
df['age_years'] = df['age'] / 365

# Gênero legível
df['gender_label'] = df['gender'].map({1: 'Feminino', 2: 'Masculino'}).astype('category')

# Variáveis binárias (0/1)
binary_cols = ['smoke', 'alco', 'active', 'cardio']
for c in binary_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').clip(0, 1)

# Rótulos de colesterol e glicose (dataset cardio: 1,2,3)
lvl_map = {1: 'Normal', 2: 'Acima do normal', 3: 'Muito acima do normal'}
df['chol_label'] = pd.to_numeric(df.get('cholesterol'), errors='coerce').map(lvl_map)
df['gluc_label']  = pd.to_numeric(df.get('gluc'),        errors='coerce').map(lvl_map)

# ====== Resumos para exibição ======
shape_text = f"Shape (linhas x colunas): {df.shape[0]} x {df.shape[1]}"
buf = StringIO(); df.info(buf=buf); info_text = buf.getvalue()

# ====== Qualidade dos dados: cálculos (sem gráfico) ======
# 1) Valores ausentes por coluna (texto estilo df.isnull().sum())
_missing_series = df.isnull().sum()
missing_text = _missing_series.to_string() + f"\ndtype: {_missing_series.dtype}"

# 2) Duplicatas
dup_rows = int(df.duplicated().sum())
dup_ids = int(df['id'].duplicated().sum()) if 'id' in df.columns else None
dups_text = (
    f"Duplicatas (linhas): {dup_rows}"
    + (f"\nDuplicatas por id: {dup_ids}" if dup_ids is not None else "")
)

# 3) Checagens de sanidade
def _num(s):  # garante numérico
    return pd.to_numeric(df[s], errors='coerce') if s in df.columns else pd.Series(dtype='float64')
def _exists(*cols): return all(c in df.columns for c in cols)
def _count(cond): return int(pd.Series(cond).sum())

sanity_lines = []
if _exists('ap_hi', 'ap_lo'):
    ap_hi = _num('ap_hi'); ap_lo = _num('ap_lo')
    sanity_lines.append(f"ap_hi < ap_lo: {_count(ap_hi < ap_lo)}")
    sanity_lines.append(f"ap_hi fora [80, 250]: {_count((ap_hi < 80) | (ap_hi > 250))}")
    sanity_lines.append(f"ap_lo fora [40, 150]: {_count((ap_lo < 40) | (ap_lo > 150))}")
else:
    sanity_lines.append('ap_hi/ap_lo: coluna(s) ausente(s)')
if 'height' in df.columns:
    h = _num('height'); sanity_lines.append(f"height fora [120, 220] cm: {_count((h < 120) | (h > 220))}")
else:
    sanity_lines.append('height: coluna ausente')
if 'weight' in df.columns:
    w = _num('weight'); sanity_lines.append(f"weight fora [30, 200] kg: {_count((w < 30) | (w > 200))}")
else:
    sanity_lines.append('weight: coluna ausente')
sanity_text = "Checagens de sanidade (contagens de violações):\n" + "\n".join(f" - {line}" for line in sanity_lines)

# ====== App ======
app = Dash()

app.layout = html.Div([
    html.H2('Doença Cardiovascular — EDA básica'),

    html.Hr(),
    html.H3('1) Visão geral'),
    html.P('Quantidade de linhas e colunas do dataset.'),
    html.Pre(shape_text),

    html.Hr(),
    html.H3('2) Informações do DataFrame (df.info)'),
    html.P('Tipos de dados e contagem de valores não nulos por coluna.'),
    html.Pre(info_text, style={'whiteSpace': 'pre-wrap'}),

    # ---- Qualidade dos dados ANTES da amostra ----
    html.Hr(),
    html.H3('3) Qualidade dos dados'),
    html.P('Valores ausentes, duplicatas e checagens de sanidade.'),
    html.H4('3.1) Valores ausentes por coluna'),
    html.Pre(missing_text),
    html.H4('3.2) Duplicatas'),
    html.Pre(dups_text),
    html.H4('3.3) Checagens de sanidade'),
    html.Pre(sanity_text),

    html.Hr(),
    html.H3('4) Tabela de dados (amostra)'),
    html.P('Primeiros 200 registros para inspeção rápida.'),
    dash_table.DataTable(
        data=df.head(200).to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'}
    ),

    html.Hr(),
    html.H3('5) Positivos vs. Negativos — empilhado por gênero'),
    html.P('Uma barra para Positivo (1) e outra para Negativo (0); dentro de cada barra, a divisão por Feminino e Masculino.'),
    dcc.RadioItems(
        id='metric-radio-stacked',
        options=[
            {'label': 'Fumante (smoke)', 'value': 'smoke'},
            {'label': 'Álcool (alco)', 'value': 'alco'},
            {'label': 'Ativo fisicamente (active)', 'value': 'active'},
            {'label': 'Cardiopatia (cardio)', 'value': 'cardio'},
        ],
        value='cardio',
        inline=True
    ),
    dcc.Graph(id='graph-stacked-binary'),

    html.Hr(),
    html.H3('6) Métrica por faixa etária (por gênero)'),
    html.P('Escolha a métrica e o status (1=Positivo, 0=Negativo). O gráfico mostra a proporção do status escolhido por faixa etária e gênero.'),
    dcc.RadioItems(
        id='metric-radio-age',
        options=[
            {'label': 'Fumante (smoke)', 'value': 'smoke'},
            {'label': 'Álcool (alco)',  'value': 'alco'},
            {'label': 'Ativo fisicamente (active)', 'value': 'active'},
            {'label': 'Cardiopatia (cardio)', 'value': 'cardio'},
        ],
        value='cardio',
        inline=True
    ),
    dcc.Dropdown(
        id='status-dd-age',
        options=[{'label': 'Positivos (1)', 'value': 1},
                 {'label': 'Negativos (0)', 'value': 0}],
        value=1,
        clearable=False,
        style={'maxWidth': '320px'}
    ),
    dcc.Graph(id='graph-by-age'),

    html.Hr(),
    html.H3('7) Distribuições categóricas — mostrar 1 por vez (empilhado por gênero)'),
    html.P('Selecione qual distribuição deseja visualizar: PA (codificada), Colesterol (níveis) ou Glicose (níveis).'),
    dcc.RadioItems(
        id='cat-radio',
        options=[
            {'label': 'Categoria de PA (codificada)', 'value': 'bp'},
            {'label': 'Níveis de Colesterol',         'value': 'chol'},
            {'label': 'Níveis de Glicose',            'value': 'gluc'},
        ],
        value='bp',
        inline=True
    ),
    dcc.Graph(id='graph-categorical-counts'),

    html.Hr(),
    html.H3('8) Categorias por faixa etária (por gênero)'),
    html.P('Selecione a variável categórica e a categoria específica; veremos a proporção dessa categoria por faixa etária e gênero.'),
    dcc.RadioItems(
        id='cat-var-age',
        options=[
            {'label': 'Categoria de PA (codificada)', 'value': 'bp'},
            {'label': 'Níveis de Colesterol',         'value': 'chol'},
            {'label': 'Níveis de Glicose',            'value': 'gluc'},
        ],
        value='chol',
        inline=True
    ),
    dcc.Dropdown(
        id='cat-value-age',
        options=[],   # preenchido dinamicamente
        value=None,   # padrão vem do callback
        clearable=False,
        placeholder='Selecione a categoria…',
        style={'maxWidth': '420px'}
    ),
    dcc.Graph(id='graph-cat-by-age'),
])

# ====== Cores e ordem comuns ======
COLOR_MAP = {'Masculino': '#93c5fd', 'Feminino': '#d8b4fe'}  # azul claro / roxo claro
GENDER_ORDER = ['Feminino', 'Masculino']

# ====== Callback: Positivo vs Negativo empilhado por gênero ======
@callback(
    Output('graph-stacked-binary', 'figure'),
    Input('metric-radio-stacked', 'value')
)
def update_graph_stacked(metric):
    tmp = df[['gender_label', metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors='coerce')
    tmp = tmp.dropna(subset=[metric]).astype({metric: int})

    g = tmp.groupby(['gender_label', metric]).size().rename('count').reset_index()

    status_codes = [1, 0]
    full_index = pd.MultiIndex.from_product([GENDER_ORDER, status_codes], names=['gender_label', metric])
    g = g.set_index(['gender_label', metric]).reindex(full_index, fill_value=0).reset_index()

    g['status'] = g[metric].map({1: 'Positivo (1)', 0: 'Negativo (0)'})
    totals = g.groupby('status', as_index=False)['count'].sum().rename(columns={'count': 'total_status'})
    g = g.merge(totals, on='status', how='left')
    g['perc'] = (g['count'] / g['total_status']).fillna(0)

    fig = px.bar(
        g, x='status', y='count',
        color='gender_label', text='count',
        color_discrete_map=COLOR_MAP,
        category_orders={'status': ['Positivo (1)', 'Negativo (0)'], 'gender_label': GENDER_ORDER},
        labels={'status': 'Valor', 'count': 'Contagem', 'gender_label': 'Gênero'},
        title=f'Positivos vs. Negativos — {metric} (empilhado por gênero)',
        custom_data=['gender_label', 'perc']
    )
    fig.update_layout(barmode='stack', legend_title_text='Gênero')
    fig.update_traces(
        hovertemplate='Status: %{x}<br>Gênero: %{customdata[0]}<br>Contagem: %{y}<br>Proporção no status: %{customdata[1]:.1%}<extra></extra>',
        textposition='inside'
    )
    fig.update_yaxes(title='Contagem')

    for _, row in totals.iterrows():
        fig.add_annotation(x=row['status'], y=row['total_status'],
                           text=f"Total: {int(row['total_status'])}",
                           showarrow=False, yshift=12, font=dict(size=12))
    return fig

# ====== Callback: Métrica por faixa etária (por gênero) — hover na ordem: Faixa, Gênero, Contagem, Total, Proporção
@callback(
    Output('graph-by-age', 'figure'),
    Input('metric-radio-age', 'value'),
    Input('status-dd-age', 'value')
)
def update_graph_by_age(metric, status_value):
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    age_labels = ['<30', '30–39', '40–49', '50–59', '60–69', '70+']

    tmp = df[['age_years', 'gender_label', metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors='coerce')
    tmp = tmp.dropna(subset=['age_years', metric])
    tmp['age_bin'] = pd.cut(tmp['age_years'], bins=age_bins, labels=age_labels, right=False)

    tmp['is_status'] = (tmp[metric].astype(int) == int(status_value)).astype(int)

    agg = (
        tmp.groupby(['age_bin', 'gender_label'])['is_status']
           .agg(n_total='count', n_status='sum')
           .reset_index()
    )
    agg['proportion'] = (agg['n_status'] / agg['n_total']).fillna(0)

    status_label_short = {
        'smoke': {1: 'Fumantes',              0: 'Não fumantes'},
        'alco':  {1: 'Quem consome álcool',   0: 'Quem não consome álcool'},
        'active':{1: 'Ativos fisicamente',    0: 'Não ativos'},
        'cardio':{1: 'Com cardiopatia',       0: 'Sem cardiopatia'},
    }.get(metric, {1: 'Positivos', 0: 'Negativos'})[int(status_value)]

    fig = px.line(
        agg, x='age_bin', y='proportion',
        color='gender_label', markers=True,
        color_discrete_map=COLOR_MAP,
        category_orders={'age_bin': age_labels, 'gender_label': GENDER_ORDER},
        labels={'age_bin': 'Faixa etária', 'proportion': f'Proporção de {status_label_short}', 'gender_label': 'Gênero'},
        title=f'Proporção de {status_label_short} em {metric} por faixa etária e gênero',
        custom_data=['gender_label', 'n_status', 'n_total']
    )
    fig.update_yaxes(range=[0, 1], tickformat='.0%', title=f'Proporção de {status_label_short}')
    fig.update_traces(
        mode='lines+markers',
        hovertemplate=(
            'Faixa: %{x}<br>'
            'Gênero: %{customdata[0]}<br>'
            f'Contagem de {status_label_short}: %{{customdata[1]}}<br>'
            'Total na faixa/gênero: %{customdata[2]}<br>'
            f'Proporção de {status_label_short}: %{{y:.1%}}'
            '<extra></extra>'
        )
    )
    return fig

# ====== Callback: Distribuições categóricas (empilhado por gênero) ======
@callback(
    Output('graph-categorical-counts', 'figure'),
    Input('cat-radio', 'value')
)
def update_categorical_graph(which):
    if which == 'chol':
        order = ['Normal', 'Acima do normal', 'Muito acima do normal']
        x_label = 'Nível de Colesterol'
        title = 'Nº de pessoas por nível de Colesterol (empilhado por gênero)'
        tmp = df[['gender_label', 'chol_label']].copy().dropna(subset=['chol_label'])
        tmp['cat'] = pd.Categorical(tmp['chol_label'], categories=order, ordered=True)

    elif which == 'gluc':
        order = ['Normal', 'Acima do normal', 'Muito acima do normal']
        x_label = 'Nível de Glicose'
        title = 'Nº de pessoas por nível de Glicose (empilhado por gênero)'
        tmp = df[['gender_label', 'gluc_label']].copy().dropna(subset=['gluc_label'])
        tmp['cat'] = pd.Categorical(tmp['gluc_label'], categories=order, ordered=True)

    else:
        title = 'Nº de pessoas por categoria de PA (codificada) — empilhado por gênero'
        x_label = 'Categoria de PA'
        if 'bp_category_encoded' not in df.columns:
            empty = pd.DataFrame({'cat': ['Coluna "bp_category_encoded" não encontrada'],
                                  'gender_label': ['Feminino'], 'count': [0]})
            fig = px.bar(empty, x='cat', y='count', color='gender_label',
                         color_discrete_map=COLOR_MAP, text='count', title=title)
            return fig

        ser = df['bp_category_encoded']
        if pd.api.types.is_numeric_dtype(ser):
            tmp = df[['gender_label', 'bp_category_encoded']].copy()
            tmp['cat_code'] = pd.to_numeric(tmp['bp_category_encoded'], errors='coerce')
            tmp = tmp.dropna(subset=['cat_code'])
            order_codes = sorted(tmp['cat_code'].unique())
            tmp['cat'] = pd.Categorical(tmp['cat_code'], categories=order_codes, ordered=True)
            tmp['cat'] = tmp['cat'].astype('Int64').astype(str)
            order = [str(c) for c in order_codes]
            x_label = 'Código da categoria de PA'
        else:
            labels_txt = ser.astype('string').str.strip()
            tmp = df[['gender_label']].copy()
            tmp['cat'] = labels_txt
            tmp = tmp.dropna(subset=['cat'])
            rank_map = {
                'normal': 0, 'elevated': 1,
                'hypertension stage 1': 2, 'stage 1': 2, 'htn stage 1': 2,
                'hypertension stage 2': 3, 'stage 2': 3, 'htn stage 2': 3,
                'hypertensive crisis': 4, 'crisis': 4
            }
            seen = sorted(tmp['cat'].unique(),
                          key=lambda v: (rank_map.get(str(v).lower(), 99), str(v)))
            order = seen
            tmp['cat'] = pd.Categorical(tmp['cat'], categories=order, ordered=True)

    g = tmp.groupby(['cat', 'gender_label']).size().rename('count').reset_index()

    if hasattr(tmp['cat'].dtype, 'categories'):
        cat_order = [c for c in tmp['cat'].cat.categories if c in g['cat'].unique()]
    else:
        cat_order = order
    full_index = pd.MultiIndex.from_product([cat_order, GENDER_ORDER], names=['cat', 'gender_label'])
    g = g.set_index(['cat', 'gender_label']).reindex(full_index, fill_value=0).reset_index()

    totals = g.groupby('cat', as_index=False)['count'].sum().rename(columns={'count': 'total_cat'})
    g = g.merge(totals, on='cat', how='left')
    g['perc'] = (g['count'] / g['total_cat']).fillna(0)

    fig = px.bar(
        g, x='cat', y='count',
        color='gender_label', text='count',
        color_discrete_map=COLOR_MAP,
        category_orders={'cat': cat_order, 'gender_label': GENDER_ORDER},
        labels={'cat': x_label, 'count': 'Contagem', 'gender_label': 'Gênero'},
        title=title,
        custom_data=['gender_label', 'perc']
    )
    fig.update_layout(barmode='stack', legend_title_text='Gênero')
    fig.update_traces(
        hovertemplate='Categoria: %{x}<br>Gênero: %{customdata[0]}<br>Contagem: %{y}<br>Participação na categoria: %{customdata[1]:.1%}<extra></extra>',
        textposition='inside'
    )
    fig.update_yaxes(title='Contagem')

    for _, row in totals.iterrows():
        fig.add_annotation(x=row['cat'], y=row['total_cat'],
                           text=f"Total: {int(row['total_cat'])}",
                           showarrow=False, yshift=12, font=dict(size=12))
    return fig

# ====== Helpers para opções da sub-categoria (padrão garantido) ======
def _options_for_cat_var(var_key):
    if var_key == 'chol':
        order = ['Normal', 'Acima do normal', 'Muito acima do normal']
        vals = [v for v in order if v in set(df['chol_label'].dropna().unique())]
        return [{'label': v, 'value': v} for v in vals], (vals[0] if vals else None)

    if var_key == 'gluc':
        order = ['Normal', 'Acima do normal', 'Muito acima do normal']
        vals = [v for v in order if v in set(df['gluc_label'].dropna().unique())]
        return [{'label': v, 'value': v} for v in vals], (vals[0] if vals else None)

    if 'bp_category_encoded' not in df.columns:
        return [], None

    ser = df['bp_category_encoded']
    if pd.api.types.is_numeric_dtype(ser):
        codes = pd.to_numeric(ser, errors='coerce').dropna().astype('Int64').astype(str).unique().tolist()
        vals = sorted(codes, key=lambda x: int(x))
        return [{'label': f'Código {v}', 'value': v} for v in vals], (vals[0] if vals else None)
    else:
        labels = ser.astype('string').str.strip().dropna().unique().tolist()
        rank_map = {
            'normal': 0, 'elevated': 1,
            'hypertension stage 1': 2, 'stage 1': 2, 'htn stage 1': 2,
            'hypertension stage 2': 3, 'stage 2': 3, 'htn stage 2': 3,
            'hypertensive crisis': 4, 'crisis': 4
        }
        vals = sorted(labels, key=lambda v: (rank_map.get(str(v).lower(), 99), str(v)))
        return [{'label': v, 'value': v} for v in vals], (vals[0] if vals else None)

# Popula dropdown da sub-categoria (padrão vem do callback)
@callback(
    Output('cat-value-age', 'options'),
    Output('cat-value-age', 'value'),
    Input('cat-var-age', 'value')
)
def populate_cat_value_options(var_key):
    options, default = _options_for_cat_var(var_key)
    return options, default

# ====== Callback: Categoria por faixa etária (por gênero) — hover na ordem: Faixa, Gênero, Contagem, Total, Proporção
@callback(
    Output('graph-cat-by-age', 'figure'),
    Input('cat-var-age', 'value'),
    Input('cat-value-age', 'value')
)
def update_graph_cat_by_age(var_key, cat_value):
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    age_labels = ['<30', '30–39', '40–49', '50–59', '60–69', '70+']

    title_var = {'bp': 'Categoria de PA (codificada)',
                 'chol': 'Níveis de Colesterol',
                 'gluc': 'Níveis de Glicose'}.get(var_key, 'Categoria')

    if cat_value is None:
        fig = px.line(title=f'{title_var} por faixa etária e gênero — selecione uma categoria')
        fig.update_layout(xaxis_title='Faixa etária', yaxis_title='Proporção', yaxis_range=[0, 1])
        return fig

    if var_key == 'chol':
        tmp = df[['age_years', 'gender_label', 'chol_label']].copy().dropna(subset=['age_years', 'chol_label'])
        tmp['cat_label'] = tmp['chol_label'].astype(str)
    elif var_key == 'gluc':
        tmp = df[['age_years', 'gender_label', 'gluc_label']].copy().dropna(subset=['age_years', 'gluc_label'])
        tmp['cat_label'] = tmp['gluc_label'].astype(str)
    else:
        if 'bp_category_encoded' not in df.columns:
            return px.line(title='Coluna "bp_category_encoded" não encontrada')
        ser = df['bp_category_encoded']
        if pd.api.types.is_numeric_dtype(ser):
            tmp = df[['age_years', 'gender_label', 'bp_category_encoded']].copy()
            tmp['cat_label'] = pd.to_numeric(tmp['bp_category_encoded'], errors='coerce').astype('Int64').astype(str)
        else:
            tmp = df[['age_years', 'gender_label']].copy()
            tmp['cat_label'] = ser.astype('string').str.strip()
        tmp = tmp.dropna(subset=['age_years', 'cat_label'])

    tmp['age_bin'] = pd.cut(tmp['age_years'], bins=age_bins, labels=age_labels, right=False)
    tmp['is_sel'] = (tmp['cat_label'] == str(cat_value)).astype(int)

    agg = tmp.groupby(['age_bin', 'gender_label'])['is_sel'].agg(n_total='count', n_sel='sum').reset_index()
    agg['proportion'] = (agg['n_sel'] / agg['n_total']).fillna(0)

    fig = px.line(
        agg, x='age_bin', y='proportion',
        color='gender_label', markers=True,
        color_discrete_map=COLOR_MAP,
        category_orders={'age_bin': age_labels, 'gender_label': GENDER_ORDER},
        labels={'age_bin': 'Faixa etária', 'proportion': 'Proporção', 'gender_label': 'Gênero'},
        title=f'Proporção da categoria "{cat_value}" — {title_var} por faixa etária e gênero',
        custom_data=['gender_label', 'n_sel', 'n_total']
    )
    fig.update_yaxes(range=[0, 1], tickformat='.0%', title='Proporção')
    fig.update_traces(
        mode='lines+markers',
        hovertemplate=(
            'Faixa: %{x}<br>'
            'Gênero: %{customdata[0]}<br>'
            f'Contagem em "{cat_value}": %{{customdata[1]}}<br>'
            'Total na faixa/gênero: %{customdata[2]}<br>'
            f'Proporção em "{cat_value}": %{{y:.1%}}'
            '<extra></extra>'
        )
    )
    return fig

# ====== Run ======
if __name__ == '__main__':
    app.run(debug=True)
