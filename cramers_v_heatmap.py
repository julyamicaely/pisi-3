# cramers_v_heatmap.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# ====== Função Cramér's V ======
def cramers_v(x, y):
    """Calcula Cramér's V entre duas colunas categóricas."""
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0

# ====== Carregamento e pré-processamento ======
df = pd.read_parquet('cardio_data.parquet')

# Colunas categóricas legíveis
df['gender_label'] = df['gender'].map({1: 'Feminino', 2: 'Masculino'}).astype('category')
lvl_map = {1: 'Normal', 2: 'Acima do normal', 3: 'Muito acima do normal'}
df['chol_label'] = pd.to_numeric(df.get('cholesterol'), errors='coerce').map(lvl_map)
df['gluc_label']  = pd.to_numeric(df.get('gluc'), errors='coerce').map(lvl_map)

# Seleciona apenas colunas categóricas para o heatmap
cat_cols = ['gender_label', 'smoke', 'alco', 'active', 'cardio', 'chol_label', 'gluc_label']
cat_cols = [c for c in cat_cols if c in df.columns]

# ====== Matriz de Cramér's V ======
v_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

for col1 in cat_cols:
    for col2 in cat_cols:
        v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

# ====== Plot do heatmap ======
plt.figure(figsize=(10, 8))
sns.heatmap(v_matrix.astype(float), annot=True, fmt=".2f", cmap='coolwarm', vmin=0, vmax=1)
plt.title("Matriz de Cramér's V — Variáveis Categóricas")
plt.tight_layout()
plt.savefig("cramers_v_heatmap.png", dpi=300)
plt.show()

