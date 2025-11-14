import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score  # <-- Importar a métrica
import os
import time
import warnings

# --- Configuração ---
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 10})

# --- 0. Setup de Caminhos ---
data_path = './clusterization/data_clusters.parquet'
output_dir = './clusterization/graphics'
# Criar o diretório se não existir
os.makedirs(output_dir, exist_ok=True)
print(f"Diretório de gráficos '{output_dir}' está pronto.")

# --- Bloco 1: Preparação dos Dados (Usando Opção 2: BMI) ---

print("Iniciando Bloco 1: Preparação dos Dados (com BMI)...")
try:
    df = pd.read_parquet(data_path)
    print(f"Dataset '{data_path}' carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{data_path}' não encontrado.")
    exit()

# --- Definir Listas de Features (Opção 2) ---
continuous_features_v2 = ['age_years', 'ap_hi', 'ap_lo', 'bmi']
categorical_features = ['gender', 'cholesterol', 'gluc']
binary_features = ['smoke', 'alco', 'active']

# --- Definir o Pré-processador (Opção 2) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', StandardScaler(), continuous_features_v2),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('binary', 'passthrough', binary_features)
    ],
    remainder='drop'
)

# Dados processados (para K-Means e DBI)
X = preprocessor.fit_transform(df)
print(f"Dados pré-processados (X). Shape: {X.shape}")
print("...Preparação dos dados concluída.")

# --- Bloco 2: Execução K-Means e Cálculo do Davies-Bouldin ---

range_n_clusters = range(2, 31) # Usando o range 2-30
db_scores = []  # Lista para guardar os scores

print(f"\nIniciando Bloco 2: Executando K-Means e calculando DBI (K de {range_n_clusters.start} a {range_n_clusters.stop-1})...")
start_loop_time = time.time()

for n_clusters in range_n_clusters:
    k_start_time = time.time()
    
    # 1. Rodar o K-Means
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', algorithm='lloyd')
    cluster_labels = clusterer.fit_predict(X)
    
    # 2. Calcular o Davies-Bouldin Score
    db_score = davies_bouldin_score(X, cluster_labels)
    db_scores.append(db_score)
    
    k_end_time = time.time()
    print(f"  ...K-Means para k={n_clusters} concluído em {k_end_time - k_start_time:.2f}s. (Score DBI: {db_score:.4f})")

total_loop_time = time.time() - start_loop_time
print(f"...Execução do K-Means finalizada em {total_loop_time:.2f}s.")

# --- Bloco 3: Plotar Gráfico de Resumo do DBI ---

print("\nIniciando Bloco 3: Plotando Gráfico Resumo do DBI...")
output_path_summary = os.path.join(output_dir, 'davies_bouldin_summary.png')
plt.figure(figsize=(12, 7))
plt.plot(range_n_clusters, db_scores, 'bo-', markersize=5)
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Índice Davies-Bouldin (Menor é Melhor)')
plt.title('Índice Davies-Bouldin vs. Número de Clusters (K) - Opção BMI')
plt.xticks(list(range(range_n_clusters.start, range_n_clusters.stop, 2))) # Ticks a cada 2
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(output_path_summary)
plt.clf()
print(f"Gráfico de resumo salvo em: {output_path_summary}")

print("\nAnálise Davies-Bouldin concluída.")