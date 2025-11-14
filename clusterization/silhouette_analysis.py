import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
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
continuous_features = ['age_years', 'ap_hi', 'ap_lo', 'bmi']
categorical_features = ['gender', 'cholesterol', 'gluc']
binary_features = ['smoke', 'alco', 'active']

# --- Definir o Pré-processador (Opção 2) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', StandardScaler(), continuous_features),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('binary', 'passthrough', binary_features)
    ],
    remainder='drop'
)

# Dados processados (para K-Means e Silhueta)
X = preprocessor.fit_transform(df)
print(f"Dados pré-processados (X). Shape: {X.shape}")

# Dados reduzidos para 2D (APENAS para visualização)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print(f"Dados reduzidos (X_pca) para visualização. Shape: {X_pca.shape}")
print("...Preparação dos dados concluída.")

# --- Bloco 2: Execução ÚNICA do K-Means (Coleta de Dados) ---

range_n_clusters = range(2, 31) # Usando o range 2-30

# Dicionário para guardar os resultados detalhados de cada 'k'
all_results = {}
# Lista para guardar a média da silhueta (para um gráfico de resumo final)
silhouette_avg_list = []

print(f"\nIniciando Bloco 2: Executando K-Means (K de {range_n_clusters.start} a {range_n_clusters.stop-1})...")
start_loop_time = time.time()

for n_clusters in range_n_clusters:
    k_start_time = time.time()
    
    # 1. Rodar o K-Means
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', algorithm='lloyd')
    cluster_labels = clusterer.fit_predict(X)
    
    # 2. Calcular todas as métricas
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    centers = clusterer.cluster_centers_
    
    # 3. Salvar os resultados
    silhouette_avg_list.append(silhouette_avg)
    all_results[n_clusters] = {
        'labels': cluster_labels,
        'sample_values': sample_silhouette_values,
        'avg_score': silhouette_avg,
        'centers': centers
    }
    
    k_end_time = time.time()
    print(f"  ...K-Means para k={n_clusters} concluído em {k_end_time - k_start_time:.2f}s. (Score Silhueta: {silhouette_avg:.4f})")

total_loop_time = time.time() - start_loop_time
print(f"...Execução do K-Means finalizada em {total_loop_time:.2f}s.")

# --- Bloco 3: Gráfico de Resumo da Silhueta ---

print("\nIniciando Bloco 3: Plotando Gráfico Resumo da Silhueta...")
output_path_summary = os.path.join(output_dir, 'silhouette_summary.png')
plt.figure(figsize=(12, 7))
plt.plot(range_n_clusters, silhouette_avg_list, 'bo-', markersize=5)
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Score Médio de Silhueta')
plt.title('Score Médio de Silhueta vs. Número de Clusters (K) - Opção BMI')
plt.xticks(list(range(range_n_clusters.start, range_n_clusters.stop, 2))) # Ticks a cada 2
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(output_path_summary)
plt.clf()
print(f"Gráfico de resumo salvo em: {output_path_summary}")


# --- Bloco 4: Gráficos Detalhados (Silhueta + PCA) ---

print("\nIniciando Bloco 4: Plotando gráficos detalhados (um por um)...")

for n_clusters in range_n_clusters:
    # 1. Recuperar os dados salvos
    results = all_results[n_clusters]
    cluster_labels = results['labels']
    sample_silhouette_values = results['sample_values']
    silhouette_avg = results['avg_score']
    centers = results['centers']
    
    # 2. Criar a figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # --- Gráfico 1: Silhueta ---
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("Gráfico de silhueta para os clusters")
    ax1.set_xlabel("Valores do coeficiente de silhueta")
    ax1.set_ylabel("Rótulo do Cluster")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # --- Gráfico 2: Visualização dos Clusters (com PCA) ---
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    
    ax2.scatter(
        X_pca[:, 0], X_pca[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    centers_pca = pca.transform(centers)
    ax2.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )
    for i, c in enumerate(centers_pca):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("Visualização dos dados clusterizados (via PCA)")
    ax2.set_xlabel("Componente Principal 1")
    ax2.set_ylabel("Componente Principal 2")

    plt.suptitle(
        f"Análise de Silhueta (Opção BMI) com n_clusters = {n_clusters}",
        fontsize=14,
        fontweight="bold",
    )
    
    # Salvar o gráfico
    filename = f'silhouette_k{n_clusters:02d}.png'
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path)
    plt.close(fig) # Fecha a figura para economizar memória
    
    print(f"  ...Gráfico detalhado salvo: {output_path}")

print(f"\nAnálise completa concluída. Todos os {len(range_n_clusters)} gráficos detalhados estão em '{output_dir}'.")