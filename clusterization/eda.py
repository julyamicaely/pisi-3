import pandas as pd


# PARTE 1: PREPARAÇÃO DOS DADOS
df = pd.read_parquet('./EDA/cardio_data.parquet')
#mostrar o dataset no dash

bp_category_order = {
    'Normal': 0,
    'Elevated': 1,
    'Hypertension Stage 1': 2,
    'Hypertension Stage 2': 3,
    'Hypertensive Crisis': 4
}

df['bp_category_encoded'] = df['bp_category'].map(bp_category_order)

df_clean = df.copy()
df_clean = df_clean[(df_clean['ap_hi'] >= 80) & (df_clean['ap_hi'] <= 250)]
df_clean = df_clean[(df_clean['ap_lo'] >= 50) & (df_clean['ap_lo'] <= 150)]
df_clean = df_clean[(df_clean['height'] >= 120) & (df_clean['height'] <= 220)]
df_clean = df_clean[(df_clean['weight'] >= 40) & (df_clean['weight'] <= 200)]

parquet_file = './clusterization/cardio_data_processed.parquet'
df_clean.to_parquet(parquet_file, index=False)
print(f'Arquivo convertido com sucesso: {parquet_file}')

# Selecionar colunas para clusterização
clustering_cols = [
    'age_years', 'gender', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]
df_cluster = df_clean[clustering_cols].copy()


# Arquivos
parquet_file = './clusterization/data_clusters.parquet'

df_cluster.to_parquet(parquet_file, index=False)

print(f'Arquivo convertido com sucesso: {parquet_file}')