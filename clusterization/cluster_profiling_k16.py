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

K_ESCOLHIDO = 16

# --- CAMINHOS ---
cluster_data_path = './clusterization/data_clusters.parquet'
# Agora carregamos o arquivo que já tem os clusters K=6
input_path = './clusterization/cardio_data_processed_with_clusters.parquet'
output_path = './clusterization/cardio_data_processed_with_clusters.parquet'

print(f"--- INICIANDO ANÁLISE COM K={K_ESCOLHIDO} ---")

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
print(f"Carregando arquivo principal de '{input_path}' para anexar rótulos K=16...")
try:
    # Carrega o arquivo que já tem os clusters K=6
    df_final = pd.read_parquet(input_path)
except FileNotFoundError:
    print(f"ERRO: Arquivo principal '{input_path}' não encontrado.")
    print("Execute primeiro o script cluster_profiling_k6.py")
    exit()

# --- VERIFICAÇÃO DE SANIDADE ---
if len(df_clusters) != len(df_final):
    print("\nERRO CRÍTICO: Os arquivos têm tamanhos diferentes!")
    print(f"  {cluster_data_path} tem {len(df_clusters)} linhas.")
    print(f"  {input_path} tem {len(df_final)} linhas.")
    print("Não é possível anexar os rótulos. Corrija os arquivos.")
    exit()
else:
    print("Tamanhos validados. Anexando rótulos K=16 ao DataFrame principal.")

# Anexar a nova coluna 'clusterk16'
df_final['clusterk16'] = labels

# --- 4. Salvar DataFrame Final ---
df_final.to_parquet(output_path, index=False)
print(f"\nDataFrame final com clusters K=6 e K=16 salvo em: '{output_path}'")

print(f"\nDistribuição dos clusters K=16:")
print(df_final['clusterk16'].value_counts().sort_index())

# --- 5. Profiling (Interpretação) dos Clusters K=16 ---
print("\n--- PROFILING DOS CLUSTERS K=16 ---")

# Perfil Numérico (Médias)
numeric_cols = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'height', 'weight']
profile_numeric = df_final.groupby('clusterk16')[numeric_cols].mean()
print("\n### Perfil Numérico K=16 (Médias por Cluster) ###")
print(profile_numeric)

print("\n### Perfil Categórico K=16: Estilo de Vida (% que SIM) ###")
lifestyle_cols = ['smoke', 'alco', 'active']
profile_lifestyle = df_final.groupby('clusterk16')[lifestyle_cols].mean().mul(100)
print(profile_lifestyle)

# --- 6. Validação com a variável 'cardio' ---
print("\n--- VALIDAÇÃO K=16 (TAXA DE DOENÇA CARDÍACA) ---")

validation = df_final.groupby('clusterk16')['cardio'].mean().mul(100).sort_values(ascending=False)
print("\n### Taxa de Doença Cardíaca (%) por Cluster K=16 ###")
print(validation)

print("\n--- ANÁLISE K=16 CONCLUÍDA ---")