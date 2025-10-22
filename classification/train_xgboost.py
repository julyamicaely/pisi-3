import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# =======================================================
# 1. Caminho base e carregamento dos dados
# =======================================================
# Caminho correto para o arquivo dentro de pisi-3
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # sobe 2 níveis até pisi-3
DATA_PATH = os.path.join(BASE_DIR, "EDA", "cardio_data.parquet")



def preprocess_data():
    print("🔧 Iniciando pré-processamento dos dados...")

    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")

    # Limpeza de valores inconsistentes
    df = df.dropna()
    df = df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0) & (df['ap_hi'] < 250) & (df['ap_lo'] < 200)]

    # Criação de colunas derivadas
    if "age" in df.columns and "age_years" not in df.columns:
        df["age_years"] = (df["age"] / 365).astype(int)

    df["cholesterol_high"] = (df["cholesterol"] > 1).astype(int)
    df["gluc_high"] = (df["gluc"] > 1).astype(int)

    # Seleção de colunas
    X = df[['gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'smoke', 'alco', 'active', 'age_years',
            'cholesterol_high', 'gluc_high']]
    y = df['cardio']

    # Normalização
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    enc_dir = os.path.join(os.path.dirname(__file__), "encoders")
    os.makedirs(enc_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(enc_dir, "scaler_xgb.joblib"))
    print("📏 Aplicado RobustScaler e salvo encoder.")

    # Balanceamento com SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    print(f"✅ Após SMOTE: {X_resampled.shape[0]} amostras balanceadas.")

    return X_resampled, y_resampled


# =======================================================
# 2. Treinamento do modelo XGBoost
# =======================================================
def train_xgboost():
    X, y = preprocess_data()

    print("📊 Treinando modelo XGBoost com 11 variáveis...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("✅ Treinamento concluído!")
    print("\n📈 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, y_pred, cm, report


# =======================================================
# 3. Salvamento dos resultados
# =======================================================
if __name__ == "__main__":
    print("🚀 Iniciando treinamento XGBoost...")

    model, X_test, y_test, y_pred, cm, report = train_xgboost()

    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Salvar artefatos
    joblib.dump(model, os.path.join(RESULTS_DIR, "model_xgb.joblib"))
    joblib.dump(X_test, os.path.join(RESULTS_DIR, "X_test_xgb.joblib"))
    joblib.dump(y_test, os.path.join(RESULTS_DIR, "y_test_xgb.joblib"))
    joblib.dump(y_pred, os.path.join(RESULTS_DIR, "y_pred_xgb.joblib"))
    joblib.dump(cm, os.path.join(RESULTS_DIR, "cm_xgb.joblib"))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "report": report
    }
    joblib.dump(metrics, os.path.join(RESULTS_DIR, "metrics_xgb.joblib"))

    print(f"💾 Artefatos salvos em: {RESULTS_DIR}")
