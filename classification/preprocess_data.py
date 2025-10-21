import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib


def preprocess_data():
    """
    Pipeline de pré-processamento do dataset cardiovascular:
      - Limpa valores inconsistentes (pressão arterial, etc.)
      - Remove outliers (IQR)
      - Transforma variáveis categóricas em dummies
      - Cria colunas binárias de risco (ex: colesterol alto)
      - Escala os dados mantendo os nomes
      - Aplica balanceamento com SMOTE
    Retorna:
      X_train, X_test, y_train, y_test, scaler, label_encoders
    """

    print("📥 Carregando dados...")
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "EDA", "cardio_data.csv")
    df = pd.read_csv(data_path)

    print(f"✅ Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")

    # Normalizar nomes das colunas
    df.columns = df.columns.str.strip().str.lower()

    # Validar target
    if "cardio" not in df.columns:
        raise ValueError("❌ Coluna 'cardio' (alvo) não encontrada no dataset!")

    # --- LIMPEZA DE VALORES ---
    print("🧽 Limpando valores inconsistentes...")

    # Corrigir idades se estiverem em dias
    if "age" in df.columns:
        df["age_years"] = (df["age"] // 365).astype(int)

    # Remover id (não ajuda no modelo)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Remover registros com pressões absurdas (não fisiológicas)
    df = df[(df["ap_hi"].between(80, 250)) & (df["ap_lo"].between(40, 180))]

    # Remover registros onde pressão sistólica < diastólica
    df = df[df["ap_hi"] >= df["ap_lo"]]

    # --- ENGENHARIA DE VARIÁVEIS ---
    print("⚙️ Criando variáveis binárias de risco...")

    # Transformar colesterol e glicose em variáveis binárias de risco
    if "cholesterol" in df.columns:
        df["cholesterol_high"] = (df["cholesterol"] > 1).astype(int)
    if "gluc" in df.columns:
        df["gluc_high"] = (df["gluc"] > 1).astype(int)

    # Remover colunas irrelevantes ou duplicadas
    cols_to_drop = [
        "age",              # já temos age_years
        "bmi",              # substituído por weight/height
        "bp_category",
        "bp_category_encoded",
        "cholesterol",      # substituído por cholesterol_high
        "gluc"              # substituído por gluc_high
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # --- REMOÇÃO DE OUTLIERS ---
    print("✂️ Removendo outliers com método IQR...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("cardio", errors="ignore")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (df[col] >= lower) & (df[col] <= upper)
        df = df[mask]

    print(f"✅ Após remoção de outliers: {df.shape[0]} linhas restantes.")

    # --- SEPARAÇÃO FEATURES / TARGET ---
    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    # --- TRATAMENTO CATEGÓRICAS ---
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # --- ESCALONAMENTO ---
    print("📏 Aplicando RobustScaler...")
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # --- BALANCEAMENTO ---
    print("⚖️ Aplicando SMOTE para balancear classes...")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_scaled, y)

    print(f"✅ Após SMOTE: {X_bal.shape[0]} amostras balanceadas.")

    # --- DIVISÃO TREINO / TESTE ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # --- SALVAR OBJETOS ---
    enc_dir = os.path.join(os.path.dirname(__file__), "encoders")
    os.makedirs(enc_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(enc_dir, "scaler.joblib"))
    for name, enc in label_encoders.items():
        joblib.dump(enc, os.path.join(enc_dir, f"encoder_{name}.joblib"))

    print("💾 Pré-processamento salvo em /classification/encoders/")
    print(f"📊 Colunas finais usadas no modelo: {list(X_train.columns)}")

    return X_train, X_test, y_train, y_test, scaler, label_encoders
