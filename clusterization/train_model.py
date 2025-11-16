import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib  # Biblioteca padrão para salvar modelos scikit-learn
import warnings

# --- 1. Configuração ---
warnings.filterwarnings('ignore')

K_ESCOLHIDO = 6
cluster_data_path = './clusterization/data_clusters.parquet'

# --- NOVO: Pasta para salvar os modelos ---
output_dir = './clusterization/models'
os.makedirs(output_dir, exist_ok=True)

print(f"--- SCRIPT DE TREINAMENTO E SALVAMENTO (K={K_ESCOLHIDO}) ---")

# --- 2. Carregar Dados de Treino ---
print(f"Carregando dados de cluster de '{cluster_data_path}'...")
try:
    df_clusters = pd.read_parquet(cluster_data_path)
except FileNotFoundError:
    print(f"ERRO: Arquivo de cluster '{cluster_data_path}' não encontrado.")
    exit()

# --- 3. Definir Pré-processador (Exatamente o mesmo de antes) ---
continuous_features_v2 = ['age_years', 'ap_hi', 'ap_lo', 'bmi']
categorical_features = ['gender', 'cholesterol', 'gluc']
binary_features = ['smoke', 'alco', 'active']

preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', StandardScaler(), continuous_features_v2),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('binary', 'passthrough', binary_features)
    ],
    remainder='drop'
)

# --- 4. Processar Dados e Treinar Modelo ---
print("Pré-processando dados...")
# Usamos fit_transform aqui pois estamos 'treinando' o preprocessor
X_processed = preprocessor.fit_transform(df_clusters)

print(f"Treinando modelo final K-Means com K={K_ESCOLHIDO}...")
kmeans_final = KMeans(n_clusters=K_ESCOLHIDO, random_state=42, n_init='auto', algorithm='lloyd')
kmeans_final.fit(X_processed)

print("Treinamento concluído.")

# --- 5. Salvar os Artefatos (Modelo e Pré-processador) ---

# É CRUCIAL salvar o pré-processador também!
# Senão, você não pode processar novos dados da mesma forma.
path_preprocessor = os.path.join(output_dir, 'preprocessor.joblib')
joblib.dump(preprocessor, path_preprocessor)
print(f"Pré-processador salvo em: '{path_preprocessor}'")

# Salvar o modelo K-Means
path_model = os.path.join(output_dir, f'kmeans_k{K_ESCOLHIDO}.joblib')
joblib.dump(kmeans_final, path_model)
print(f"Modelo K-Means salvo em: '{path_model}'")

print("\n--- Processo concluído com sucesso! ---")