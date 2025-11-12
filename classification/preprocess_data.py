import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib


def preprocess_data():
    """
    Pipeline de pré-processamento do dataset cardiovascular SEM VAZAMENTO DE DADOS.
    
    Ordem correta para prevenir data leakage:
      1. Limpa valores inconsistentes (pressão arterial, etc.)
      2. Remove outliers (IQR)
      3. Transforma variáveis categóricas com LabelEncoder
      4. Cria colunas binárias de risco (ex: colesterol alto)
      5. ✅ DIVIDE train/test ANTES de escalonar ou balancear
      6. ✅ Ajusta RobustScaler APENAS em X_train
      7. ✅ Aplica SMOTE APENAS em X_train, y_train
      8. ✅ Avalia modelo em X_test original (sem SMOTE)
    
    Retorna:
      X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names
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
    print("⚙️ Preparando variáveis...")

    # Transformar colesterol e glicose em variáveis binárias de risco
    if "cholesterol" in df.columns:
        df["cholesterol_high"] = (df["cholesterol"] > 1).astype(int)
    if "gluc" in df.columns:
        df["gluc_high"] = (df["gluc"] > 1).astype(int)

    # Remover colunas irrelevantes ou duplicadas
    cols_to_drop = [
        "age",              # já temos age_years
        "bp_category",      # categórica textual
        "bp_category_encoded",  # redundante com ap_hi e ap_lo
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

    # --- TRATAMENTO CATEGÓRICAS (antes do split) ---
    # ✅ OK: Label encoding antes do split não causa leakage, apenas transforma categorias
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Guardar nomes das features originais
    feature_names = X.columns.tolist()

    # ✅ PASSO CRÍTICO: DIVIDIR TREINO/TESTE ANTES DE ESCALAR OU BALANCEAR
    print("✂️ Dividindo dados em treino/teste (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste:  {X_test.shape[0]} amostras")

    # ✅ ESCALONAMENTO: fit apenas no treino, transform em ambos
    print("📏 Aplicando RobustScaler (fit apenas em X_train)...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),       # ✅ apenas transform, sem fit
        columns=X_test.columns, 
        index=X_test.index
    )

    # ✅ BALANCEAMENTO: aplicar SMOTE apenas no conjunto de treino
    print("⚖️ Aplicando SMOTE apenas no conjunto de treino...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"✅ Após SMOTE: {X_train_bal.shape[0]} amostras de treino balanceadas.")
    print(f"   Teste permanece original: {X_test_scaled.shape[0]} amostras (sem SMOTE).")

    # --- SALVAR OBJETOS ---
    scalers_dir = os.path.join(os.path.dirname(__file__), "scalers")
    encoders_dir = os.path.join(os.path.dirname(__file__), "encoders")
    os.makedirs(scalers_dir, exist_ok=True)
    os.makedirs(encoders_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(scalers_dir, "robust_scaler.joblib"))
    for name, enc in label_encoders.items():
        joblib.dump(enc, os.path.join(encoders_dir, f"encoder_{name}.joblib"))

    print("💾 Scaler salvo em: /classification/scalers/")
    print("💾 Encoders salvos em: /classification/encoders/")
    print(f"📊 Features finais: {feature_names}")

    return X_train_bal, X_test_scaled, y_train_bal, y_test, scaler, label_encoders, feature_names
