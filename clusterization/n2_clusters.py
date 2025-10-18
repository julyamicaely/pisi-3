import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib


# Carregar dados
df = pd.read_parquet('./clusterization/data_clusters.parquet')

# Definir o pré-processador (mesmo do n_clusters_find.py)
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

# Aplicar pré-processamento

X = preprocessor.fit_transform(df)

# CLUSTERIZAÇÃO COM 2 CLUSTERS
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Adicionar clusters ao DataFrame
df['cluster'] = clusters

# Calcular métricas
silhouette_avg = silhouette_score(X, clusters)
print(f"Clusterização concluída!")
print(f"Silhouette Score: {silhouette_avg:.4f}")

# ANÁLISE DOS CLUSTERS
print("\n" + "="*60)
print("ANÁLISE DOS CLUSTERS")
print("="*60)

# 1. Distribuição dos clusters
cluster_counts = df['cluster'].value_counts().sort_index()
print(f"\nDISTRIBUIÇÃO:")
for cluster, count in cluster_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   Cluster {cluster}: {count} pacientes ({percentage:.1f}%)")

# 2. Características por cluster
numeric_vars = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']
categorical_vars = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

cluster_means = df.groupby('cluster')[numeric_vars].mean().round(2)
cluster_categorical = df.groupby('cluster')[categorical_vars].mean().round(3)

print(f"\nVARIÁVEIS NUMÉRICAS (médias):")
print(cluster_means)

print(f"\nVARIÁVEIS CATEGÓRICAS (proporções):")
print(cluster_categorical)

# 3. INTERPRETAÇÃO DETALHADA
print(f"\nINTERPRETAÇÃO DOS CLUSTERS:")
print("-" * 50)

for cluster in cluster_means.index:
    print(f"\nCLUSTER {cluster}:")
    means = cluster_means.loc[cluster]
    cats = cluster_categorical.loc[cluster]
    
    print(f"   {cluster_counts[cluster]} pacientes ({cluster_counts[cluster]/len(df)*100:.1f}%)")
    print(f"   Idade: {means['age_years']} anos")
    print(f"   Altura: {means['height']} cm")
    print(f"   Peso: {means['weight']} kg")
    print(f"   Pressão: {means['ap_hi']}/{means['ap_lo']} mmHg")
    
    # Calcular BMI para análise (após clusterização)
    bmi = means['weight'] / ((means['height'] / 100) ** 2)
    print(f"   BMI (análise): {bmi:.1f}")
    
    # Fatores de risco
    print(f"   Colesterol nível: {cats['cholesterol']:.1f}")
    print(f"   Glicose nível: {cats['gluc']:.1f}")
    print(f"   Fumantes: {cats['smoke']*100:.1f}%")
    print(f"   Consome álcool: {cats['alco']*100:.1f}%")
    print(f"   Ativos: {cats['active']*100:.1f}%")
    
    # Classificação de risco
    risk_score = 0
    if means['age_years'] > 50: risk_score += 1
    if bmi > 25: risk_score += 1
    if means['ap_hi'] > 140: risk_score += 1
    if cats['cholesterol'] > 1.5: risk_score += 1
    if cats['smoke'] > 0.2: risk_score += 1
    
    if risk_score >= 3:
        risk_level = "ALTO RISCO CARDIOVASCULAR"
    elif risk_score >= 2:
        risk_level = "RISCO MODERADO"
    else:
        risk_level = "BAIXO RISCO"
    
    print(f"   PERFIL: {risk_level} ({risk_score}/5 fatores de risco)")

# VISUALIZAÇÕES
print(f"\nCRIANDO VISUALIZAÇÕES...")

# 1. Distribuição dos clusters
plt.figure(figsize=(10, 6))
colors = ['#66c2a5', '#fc8d62']  # Cores verdes/laranjas
bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.7)
plt.title('Distribuição dos Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Cluster')
plt.ylabel('Número de Pacientes')
plt.xticks([0, 1])

# Adicionar valores nas barras
for bar, count in zip(bars, cluster_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(cluster_counts),
             f'{count}\n({count/len(df)*100:.1f}%)', 
             ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./clusterization/cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Heatmap das diferenças entre clusters
plt.figure(figsize=(12, 6))

# Preparar dados para heatmap
comparison_data = pd.DataFrame()
for var in numeric_vars + categorical_vars:
    if var in numeric_vars:
        comparison_data[var] = cluster_means[var]
    else:
        comparison_data[var] = cluster_categorical[var]

# Calcular diferenças percentuais entre clusters
diff_percentage = ((comparison_data.loc[1] - comparison_data.loc[0]) / comparison_data.loc[0] * 100).round(1)

plt.subplot(1, 2, 1)
sns.heatmap(comparison_data.T, annot=True, fmt=".2f", cmap="YlOrRd", 
            cbar_kws={'label': 'Valor Médio'})
plt.title('Características Médias por Cluster')

plt.subplot(1, 2, 2)
sns.heatmap(diff_percentage.to_frame(), annot=True, fmt=".1f", cmap="RdYlBu_r", 
            center=0, cbar_kws={'label': 'Diferença % (Cluster1 vs Cluster0)'})
plt.title('Diferenças Percentuais entre Clusters')

plt.tight_layout()
plt.savefig('./clusterization/cluster_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. PCA para visualização 2D
print("Aplicando PCA para visualização 2D...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], 
                     cmap='viridis', alpha=0.6, s=30)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%} variância)')
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%} variância)')
plt.title('Visualização dos Clusters em 2D (PCA)')
plt.grid(alpha=0.3)
plt.savefig('./clusterization/cluster_pca.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Variância explicada pelo PCA: {pca.explained_variance_ratio_.sum():.2%}")

# 4. Boxplots das variáveis mais importantes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
important_vars = ['age_years', 'ap_hi', 'weight', 'cholesterol', 'gluc', 'smoke']

for i, var in enumerate(important_vars):
    row, col = i // 3, i % 3
    df.boxplot(column=var, by='cluster', ax=axes[row, col])
    axes[row, col].set_title(f'{var.upper()} por Cluster')
    axes[row, col].set_xlabel('Cluster')

plt.suptitle('Distribuição das Variáveis Mais Importantes por Cluster', y=1.02)
plt.tight_layout()
plt.savefig('./clusterization/cluster_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# SALVAR RESULTADOS
print(f"\n💾 SALVANDO RESULTADOS...")

# 1. Dataset com clusters
output_file = './clusterization/cardio_2clusters.parquet'
df.to_parquet(output_file, index=False)
print(f"✅ Dataset com clusters salvo em: {output_file}")

# 2. Resumo estatístico
summary_file = './clusterization/cluster_2_summary.csv'
cluster_summary = pd.concat([cluster_means, cluster_categorical], axis=1)
cluster_summary['n_pacientes'] = cluster_counts
cluster_summary['percentual'] = (cluster_summary['n_pacientes'] / len(df)) * 100
cluster_summary.to_csv(summary_file, float_format='%.3f')
print(f"✅ Resumo estatístico salvo em: {summary_file}")

# 3. Modelo e pré-processador
joblib.dump(kmeans, './clusterization/kmeans_2clusters.pkl')
joblib.dump(preprocessor, './clusterization/preprocessor_2clusters.pkl')
print(f"✅ Modelo K-Means salvo")
print(f"✅ Pré-processador salvo")

# 4. Relatório em texto
# 4. Relatório em texto
with open('./clusterization/cluster_report.txt', 'w') as f:
    f.write("RELATÓRIO DE CLUSTERIZAÇÃO - 2 CLUSTERS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
    f.write(f"Total de pacientes: {len(df)}\n\n")
    
    f.write("DISTRIBUIÇÃO:\n")
    for cluster, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        f.write(f"Cluster {cluster}: {count} pacientes ({percentage:.1f}%)\n")
    
    f.write("\nINTERPRETAÇÃO SUGERIDA:\n")
    f.write("• Cluster 0: Perfil de menor risco cardiovascular\n")
    f.write("• Cluster 1: Perfil de maior risco cardiovascular\n")
    
    f.write("\nPRINCIPAIS DIFERENÇAS:\n")
    # Usar comparison_data que contém todas as variáveis
    for var in ['age_years', 'ap_hi', 'weight', 'cholesterol']:
        if var in comparison_data.columns:
            diff = comparison_data.loc[1, var] - comparison_data.loc[0, var]
            f.write(f"- {var}: {diff:+.2f}\n")

print(f"✅ Relatório em texto salvo")
