import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_parquet('./clusterization/data_clusters.parquet')

# Definir transformações por tipo de variável
preprocessor = ColumnTransformer(
    transformers=[
        # Padronizar variáveis contínuas
        ('continuous', StandardScaler(), [
            'age_years', 'height', 'weight', 
            'ap_hi', 'ap_lo'
        ]),
        
        # Manter variáveis binárias e ordinais como estão
        ('binary_ordinal', 'passthrough', [
            'gender', 'cholesterol', 'gluc', 
            'smoke', 'alco', 'active'
        ])
    ]
)

# Padronizar os dados
X = preprocessor.fit_transform(df)

# Vamos determinar o número ideal de clusters com o método do cotovelo
inertia = []
silhouette_scores = []
range_n_clusters = range(2, 8)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plotar o método do cotovelo e silhouette score
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(range_n_clusters, inertia, 'bo-')
ax1.set_xlabel('Número de Clusters')
ax1.set_ylabel('Inércia')
ax1.set_title('Método do Cotovelo')

ax2.plot(range_n_clusters, silhouette_scores, 'ro-')
ax2.set_xlabel('Número de Clusters')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score')

plt.show()


