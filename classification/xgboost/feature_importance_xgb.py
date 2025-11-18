# feature_importance_simple.py
import os
from datetime import datetime

def main():
    print("Gerando importancia de features")
    
    try:
        # Verificar se matplotlib está disponível
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False
            print("Matplotlib nao disponivel - gerando apenas relatorio textual")
        
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.datasets import make_classification
        
        # Dados de exemplo
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Treinar modelo
        model = XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Calcular importancia
        importance_scores = model.feature_importances_
        feature_importance = list(zip(feature_names, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Gerar imagem se matplotlib disponível
        if matplotlib_available:
            # Top 10 features
            top_features = feature_importance[:10]
            features, scores = zip(*top_features)
            
            plt.figure(figsize=(10, 6))
            plt.barh(features, scores)
            plt.xlabel('Importancia')
            plt.title('Top 10 Features - XGBoost')
            plt.gca().invert_yaxis()
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            img_filename = f"xgb_feature_importance_{timestamp}.png"
            plt.savefig(img_filename, bbox_inches='tight', dpi=100)
            plt.close()
            
            print(f"Imagem salva: {img_filename}")
        
        # Salvar relatório
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"feature_importance_{timestamp}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("Importancia de Features - XGBoost\n\n")
            f.write("Top 10 Features:\n")
            for feature, score in feature_importance[:10]:
                f.write(f"{feature}: {score:.6f}\n")
        
        print(f"Relatorio salvo: {report_filename}")
        print("Analise de features concluida")
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()