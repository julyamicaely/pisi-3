# model_xgboost_simple.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime
import os

def main():
    print("Iniciando treinamento do XGBoost")
    
    try:
        # Carregar dados de exemplo
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar modelo
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Salvar modelo
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"xgboost_model_{timestamp}.joblib"
        joblib.dump(model, model_filename)
        
        # Gerar relatório
        report = classification_report(y_test, y_pred)
        report_filename = f"xgboost_report_{timestamp}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"XGBoost Classification Report\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write(report)
        
        print(f"Modelo salvo: {model_filename}")
        print(f"Relatorio salvo: {report_filename}")
        print("Treinamento concluido com sucesso")
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()