import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "EDA" / "cardio_data.parquet"


def train_xgboost():
    print("=== Treinando modelo XGBoost ===")

    df = pd.read_parquet(DATA_PATH)
    df = df.drop(columns=["id", "bp_category", "bp_category_encoded"], errors="ignore")

    X = df[["age_years", "ap_hi", "ap_lo", "cholesterol", "gluc", "weight", "height"]]
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Gera relatório
    report = classification_report(y_test, y_pred)
    print("\n📊 Relatório de Classificação:\n")
    print(report)

    # Cria pasta de resultados
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Gera timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Salva relatório com timestamp
    report_path = results_dir / f"xgboost_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Relatório de Classificação - XGBoost ===\n\n")
        f.write(report)

    print(f"💾 Relatório salvo em: {report_path}")

    return model, X_test, y_test, X.columns

if __name__ == "__main__":
    train_xgboost()
