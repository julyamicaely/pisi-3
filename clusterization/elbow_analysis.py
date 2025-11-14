import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import warnings
import time
import os

# --- Configuração ---
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 10})

# --- Modificado: Definir o diretório de saída dos gráficos ---
# A pasta 'graphics' agora está dentro de 'clusterization'
output_dir = './clusterization/graphics'
# Criar o diretório se não existir
os.makedirs(output_dir, exist_ok=True)
print(f"Diretório de gráficos '{output_dir}' está pronto.")

print("Iniciando o script...")
# --- Carregar o dataset a partir do Parquet ---
data_path = './clusterization/data_clusters.parquet'
print(f"Carregando o dataset de '{data_path}'...")
try:
    df = pd.read_parquet(data_path)
    print("Dataset carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo '{data_path}' não encontrado.")
    # Se o arquivo não for encontrado, não podemos continuar
    exit()

# --- Definir Listas de Features (Excluindo 'cardio') ---
binary_features = ['smoke', 'alco', 'active']
categorical_features = ['gender', 'cholesterol', 'gluc']

# --- Opção 1: Height + Weight (Remover BMI, bp_category, cardio) ---
print("Definindo pré-processador V1 (Height + Weight)...")
continuous_features_v1 = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']

preprocessor_v1 = ColumnTransformer(
    transformers=[
        ('continuous', StandardScaler(), continuous_features_v1),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('binary', 'passthrough', binary_features)
    ],
    remainder='drop' 
)

# --- Opção 2: BMI (Remover Height, Weight, bp_category, cardio) ---
print("Definindo pré-processador V2 (BMI)...")
continuous_features_v2 = ['age_years', 'ap_hi', 'ap_lo', 'bmi']

preprocessor_v2 = ColumnTransformer(
    transformers=[
        ('continuous', StandardScaler(), continuous_features_v2),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('binary', 'passthrough', binary_features)
    ],
    remainder='drop'
)

print("Iniciando transformações de dados...")
# Aplicar as transformações
start_transform = time.time()
X_v1 = preprocessor_v1.fit_transform(df)
print(f"Transformação V1 (Height+Weight) concluída. Shape: {X_v1.shape}")
X_v2 = preprocessor_v2.fit_transform(df)
print(f"Transformação V2 (BMI) concluída. Shape: {X_v2.shape}")
print(f"Tempo de transformação: {time.time() - start_transform:.2f}s")


# --- Função para calcular Inércia (Elbow Method) ---
def calculate_inertia(X, k_range, option_name):
    """Calcula a inércia (WCSS) para um range de K."""
    print(f"Iniciando cálculo de inércia para {option_name} (K de {k_range.start} a {k_range.stop-1})...")
    inertias = []
    start_loop = time.time()
    for k in k_range:
        k_start_time = time.time()
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42, algorithm='lloyd')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        k_end_time = time.time()
        
    print(f"Cálculo de inércia para {option_name} concluído em {time.time() - start_loop:.2f}s.")
    return inertias

# --- Função para Plotar o Gráfico ---
def plot_elbow(inertias, k_range, title, filename):
    """Plota e salva o gráfico de cotovelo."""
    print(f"Gerando gráfico: {filename}")
    plt.figure(figsize=(12, 7))
    plt.plot(k_range, inertias, 'bo-', markersize=5)
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (WCSS)')
    plt.title(title)
    
    ticks = [2] + list(k_range)[1::2] 
    if len(k_range) > 20:
        ticks = [2] + list(k_range)[1::3]
        
    plt.xticks(ticks)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Salvar no diretório correto
    plt.savefig(filename)
    plt.clf() 
    print(f"Gráfico salvo: {filename}")

# --- Execução ---
k_range = range(2, 31)

# --- Definir caminhos de saída completos ---
output_path_v1 = os.path.join(output_dir, 'elbow_plot_v1.png')
output_path_v2 = os.path.join(output_dir, 'elbow_plot_v2.png')

# Opção 1
total_start_v1 = time.time()
inertias_v1 = calculate_inertia(X_v1, k_range, "Opção 1 (Height+Weight)")
plot_elbow(inertias_v1, k_range, 'Método do Cotovelo (Opção 1: Usando Height + Weight)', output_path_v1)
print(f"Tempo total para Opção 1: {time.time() - total_start_v1:.2f}s")


# Opção 2
total_start_v2 = time.time()
inertias_v2 = calculate_inertia(X_v2, k_range, "Opção 2 (BMI)")
plot_elbow(inertias_v2, k_range, 'Método do Cotovelo (Opção 2: Usando BMI)', output_path_v2)
print(f"Tempo total para Opção 2: {time.time() - total_start_v2:.2f}s")

print("\nAnálise do método do cotovelo concluída para ambas as opções.")
print(f"Gráficos salvos em '{output_dir}'.")