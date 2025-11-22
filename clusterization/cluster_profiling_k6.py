import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import warnings

# --- 1. Configuração ---
warnings.filterwarnings('ignore')
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

K_ESCOLHIDO = 6

# --- CAMINHOS ---
cluster_data_path = './clusterization/data_clusters.parquet'
label_data_path = './clusterization/cardio_data_processed.parquet'
output_path = './clusterization/cardio_data_processed_with_clusters.parquet'

print(f"--- INICIANDO ANÁLISE FINAL COM K={K_ESCOLHIDO} ---")

# --- 2. Carregar e Treinar o Modelo ---
print(f"Carregando dados de cluster de '{cluster_data_path}'...")
try:
    df_clusters = pd.read_parquet(cluster_data_path)
except FileNotFoundError:
    print(f"ERRO: Arquivo de cluster '{cluster_data_path}' não encontrado.")
    exit()

# Definir o Pré-processador
continuous_features = ['age_years', 'ap_hi', 'ap_lo', 'bmi']
categorical_features = ['gender', 'cholesterol', 'gluc']
binary_features = ['smoke', 'alco', 'active']

preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', StandardScaler(), continuous_features),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('binary', 'passthrough', binary_features)
    ],
    remainder='drop'
)

print("Pré-processando dados e treinando modelo K-Means...")
X_processed = preprocessor.fit_transform(df_clusters)

# Treinar o Modelo Final
kmeans_final = KMeans(n_clusters=K_ESCOLHIDO, random_state=42, n_init='auto', algorithm='lloyd')
kmeans_final.fit(X_processed)

# Obter os rótulos
labels = kmeans_final.predict(X_processed)
print(f"Modelo treinado e {len(labels)} rótulos gerados.")

# --- 3. Carregar DataFrame Principal e Anexar Rótulos ---
print(f"Carregando arquivo principal de '{label_data_path}' para anexar rótulos...")
try:
    df_final = pd.read_parquet(label_data_path)
except FileNotFoundError:
    print(f"ERRO: Arquivo principal '{label_data_path}' não encontrado.")
    exit()

# --- VERIFICAÇÃO DE SANIDADE ---
if len(df_clusters) != len(df_final):
    print("\nERRO CRÍTICO: Os arquivos têm tamanhos diferentes!")
    print(f"  {cluster_data_path} tem {len(df_clusters)} linhas.")
    print(f"  {label_data_path} tem {len(df_final)} linhas.")
    print("Não é possível anexar os rótulos. Corrija os arquivos.")
    exit()
else:
    print("Tamanhos validados. Anexando rótulos ao DataFrame principal.")

# Anexar a nova coluna 'clusterk6'
df_final['clusterk6'] = labels

# --- 4. Salvar DataFrame Final ---
df_final.to_parquet(output_path, index=False)
print(f"\nDataFrame final com clusters K=6 salvo em: '{output_path}'")

print(f"\nDistribuição dos clusters (K={K_ESCOLHIDO}):")
print(df_final['clusterk6'].value_counts().sort_index())

# --- 5. Profiling (Interpretação) dos Clusters ---
print("\n--- PASSO 3: PROFILING (QUEM SÃO OS CLUSTERS?) ---")
print("(Usando dados de 'cardio_data_processed.parquet')")

# Perfil Numérico (Médias)
numeric_cols = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight']
profile_numeric = df_final.groupby('clusterk6')[numeric_cols].mean()
print("\n### Perfil Numérico (Médias por Cluster) ###")
print(profile_numeric)

# Perfil Categórico (Percentuais)
print("\n### Perfil Categórico: Colesterol (% por Cluster) ###")
profile_chol = df_final.groupby('clusterk6')['cholesterol'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
print(profile_chol)

print("\n### Perfil Categórico: Glicose (% por Cluster) ###")
profile_gluc = df_final.groupby('clusterk6')['gluc'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
print(profile_gluc)

print("\n### Perfil Categórico: Estilo de Vida (% que SIM) ###")
lifestyle_cols = ['smoke', 'alco', 'active']
profile_lifestyle = df_final.groupby('clusterk6')[lifestyle_cols].mean().mul(100)
print(profile_lifestyle)

# --- 6. Validação com a variável 'cardio' ---
print("\n--- PASSO 4: VALIDAÇÃO (TAXA DE DOENÇA CARDÍACA) ---")

validation = df_final.groupby('clusterk6')['cardio'].mean().mul(100).sort_values(ascending=False)
print("\n### Taxa de Doença Cardíaca (%) por Cluster ###")
print(validation)

print("\n--- ANÁLISE K=6 CONCLUÍDA ---")