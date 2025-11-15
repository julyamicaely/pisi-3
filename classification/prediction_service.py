"""
🔮 Serviço de Predição - Random Forest Pipeline

Este módulo fornece uma interface simplificada para fazer predições com
o pipeline de Random Forest treinado. O pipeline é portável e pode ser
usado em qualquer aplicação que tenha acesso ao arquivo .joblib.

⚠️ IMPORTANTE:
   - O pipeline espera dados já preparados (features na ordem correta)
   - Use build_feature_frame() de preprocess_data para preparar dados brutos
   - O pipeline já contém o scaler, não é necessário escalar manualmente

📦 Uso típico:
   >>> from classification.prediction_service import load_model, predict_proba
   >>> from classification.preprocess_data import build_feature_frame
   >>> 
   >>> # Preparar dados
   >>> df = pd.DataFrame({...})  # dados brutos
   >>> X = build_feature_frame(df)
   >>> 
   >>> # Fazer predição
   >>> result = predict_proba(X)
   >>> print(result['probability'], result['risk_label'])
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union


# Cache do modelo para evitar recarregamento
_MODEL_CACHE = None


def get_model_path() -> Path:
    """Retorna o caminho do pipeline de produção."""
    base_path = Path(__file__).parent
    return base_path / 'models' / 'random_forest_pipeline.joblib'


def load_model(force_reload: bool = False):
    """
    Carrega o pipeline de Random Forest treinado.
    
    O modelo é carregado uma vez e mantido em cache para performance.
    Se o pipeline não existir, lança erro informativo.
    
    Args:
        force_reload (bool): Se True, recarrega o modelo mesmo se estiver em cache
        
    Returns:
        Pipeline: Pipeline scikit-learn com scaler + classificador
        
    Raises:
        FileNotFoundError: Se o pipeline não foi treinado ainda
    """
    global _MODEL_CACHE
    
    # Retornar do cache se disponível
    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE
    
    # Verificar se o pipeline existe
    pipeline_path = get_model_path()
    
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"❌ Pipeline não encontrado em: {pipeline_path}\n\n"
            "Para criar o pipeline de produção, execute:\n"
            "   python -m classification.model_random_forest --pipeline\n\n"
            "Ou com tuning de hiperparâmetros:\n"
            "   python -m classification.model_random_forest --pipeline --tune --n_iter=10 --cv=3"
        )
    
    # Carregar pipeline
    print(f"📦 Carregando pipeline de: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    
    # Salvar no cache
    _MODEL_CACHE = pipeline
    
    print("✅ Pipeline carregado com sucesso!")
    return pipeline


def predict_single(features: Dict[str, Union[int, float]]) -> Dict[str, Union[float, str, int]]:
    """
    Faz predição para um único paciente.
    
    Args:
        features (dict): Dicionário com as features do paciente
            Exemplo:
            {
                'gender': 0,  # 0=feminino, 1=masculino
                'ap_hi': 120,  # pressão sistólica
                'ap_lo': 80,   # pressão diastólica
                'smoke': 0,    # 0=não, 1=sim
                'alco': 0,     # 0=não, 1=sim
                'active': 1,   # 0=não, 1=sim
                'age_years': 50,
                'bmi': 25.5,
                'cholesterol_high': 0,  # 0=normal, 1=alto
                'gluc_high': 0          # 0=normal, 1=alto
            }
    
    Returns:
        dict: Resultado da predição com campos:
            - probability: probabilidade de doença cardíaca (0-100%)
            - risk_label: classificação de risco ('Baixo', 'Médio', 'Alto')
            - class: classe predita (0 ou 1)
    """
    # Criar DataFrame com ordem correta das features
    feature_order = [
        'gender', 'ap_hi', 'ap_lo', 
        'smoke', 'alco', 'active', 
        'age_years', 'bmi', 
        'cholesterol_high', 'gluc_high'
    ]
    
    # Validar que todas as features estão presentes
    missing = set(feature_order) - set(features.keys())
    if missing:
        raise ValueError(f"❌ Features faltando: {missing}")
    
    # Criar DataFrame
    df = pd.DataFrame([features], columns=feature_order)
    
    # Fazer predição
    return predict_proba(df)


def predict_proba(X: pd.DataFrame) -> Union[Dict, list]:
    """
    Faz predição de probabilidade para um ou mais pacientes.
    
    ⚠️ IMPORTANTE:
       - X deve conter as features na ordem correta (ver feature_order)
       - X NÃO deve estar escalonado (o pipeline faz isso automaticamente)
       - Para dados brutos do CSV, use build_feature_frame() antes
    
    Args:
        X (pd.DataFrame): DataFrame com features preparadas
            Colunas esperadas (nesta ordem):
            ['gender', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active',
             'age_years', 'bmi', 'cholesterol_high', 'gluc_high']
    
    Returns:
        - Se X tem 1 linha: dict com resultado único
        - Se X tem múltiplas linhas: list de dicts com resultados
        
        Each resultado contém:
        - probability: probabilidade de doença cardíaca (0-100%)
        - risk_label: classificação de risco ('Baixo', 'Médio', 'Alto')
        - class: classe predita (0 ou 1)
    """
    # Carregar modelo
    pipeline = load_model()
    
    # Validar input
    expected_features = [
        'gender', 'ap_hi', 'ap_lo', 
        'smoke', 'alco', 'active', 
        'age_years', 'bmi', 
        'cholesterol_high', 'gluc_high'
    ]
    
    if not all(col in X.columns for col in expected_features):
        missing = set(expected_features) - set(X.columns)
        raise ValueError(
            f"❌ DataFrame com colunas incorretas!\n"
            f"   Faltando: {missing}\n"
            f"   Esperado: {expected_features}\n"
            f"   Recebido: {X.columns.tolist()}"
        )
    
    # Garantir ordem correta
    X = X[expected_features]
    
    # Fazer predição
    probas = pipeline.predict_proba(X)[:, 1]  # probabilidade da classe 1 (doença)
    classes = pipeline.predict(X)
    
    # Classificar risco
    def classify_risk(proba):
        if proba < 0.4:
            return 'Baixo'
        elif proba < 0.7:
            return 'Médio'
        else:
            return 'Alto'
    
    # Obter importâncias das features do modelo
    classifier = pipeline.named_steps['classifier']
    feature_importances = classifier.feature_importances_
    
    # Preparar resultados
    results = []
    for i in range(len(X)):
        # Criar dicionário de contribuições ESPECÍFICAS do paciente
        # Importância * valor da feature normalizado (quanto mais alto, maior contribuição)
        feature_contributions = {}
        patient_values = X.iloc[i]
        
        for j, feature in enumerate(expected_features):
            # Para features binárias (0/1), usar direto o valor
            # Para features contínuas, usar valor normalizado
            value = patient_values[feature]
            importance = feature_importances[j]
            
            # Contribuição = importância * indicador de risco
            # Features binárias: se = 1, contribui com importância total
            # Features contínuas: normalizar para escala 0-1
            if feature in ['smoke', 'alco', 'cholesterol_high', 'gluc_high', 'gender']:
                contribution = importance * value * 100  # 0 ou 100% da importância
            elif feature == 'active':
                contribution = importance * (1 - value) * 100  # sedentarismo é risco
            elif feature == 'age_years':
                # Normalizar idade (30-65 anos)
                normalized = (value - 30) / (65 - 30)
                contribution = importance * normalized * 100
            elif feature == 'bmi':
                # Normalizar BMI (18-40)
                normalized = min((value - 18) / (40 - 18), 1.0)
                contribution = importance * normalized * 100
            elif feature in ['ap_hi', 'ap_lo']:
                # Normalizar pressão (ap_hi: 90-180, ap_lo: 60-120)
                if feature == 'ap_hi':
                    normalized = min((value - 90) / (180 - 90), 1.0)
                else:
                    normalized = min((value - 60) / (120 - 60), 1.0)
                contribution = importance * normalized * 100
            else:
                contribution = importance * 100
            
            feature_contributions[feature] = float(contribution)
        
        result = {
            'probability': float(probas[i] * 100),  # converter para porcentagem
            'risk_label': classify_risk(probas[i]),
            'class': int(classes[i]),
            'feature_contributions': feature_contributions,  # Contribuições específicas!
            'risk_factors': _analyze_risk_factors(
                X.iloc[i], 
                expected_features, 
                feature_importances
            )
        }
        results.append(result)
    
    # Retornar dict único se input tinha 1 linha
    if len(results) == 1:
        return results[0]
    
    return results


def _analyze_risk_factors(patient_data, feature_names, importances):
    """
    Analisa quais fatores mais contribuem para o risco do paciente.
    
    Args:
        patient_data: Series com dados do paciente
        feature_names: Lista de nomes das features
        importances: Array com importâncias das features do modelo
    
    Returns:
        list: Top 3 fatores de risco com descrição e impacto
    """
    # Mapeamento de features para descrições amigáveis
    feature_labels = {
        'gender': ('Gênero', {0: 'Feminino', 1: 'Masculino'}),
        'ap_hi': ('Pressão Sistólica', lambda x: f'{int(x)} mmHg'),
        'ap_lo': ('Pressão Diastólica', lambda x: f'{int(x)} mmHg'),
        'smoke': ('Tabagismo', {0: 'Não fuma', 1: 'Fumante'}),
        'alco': ('Álcool', {0: 'Não consome', 1: 'Consome'}),
        'active': ('Atividade Física', {0: 'Sedentário', 1: 'Ativo'}),
        'age_years': ('Idade', lambda x: f'{int(x)} anos'),
        'bmi': ('IMC', lambda x: f'{x:.1f} kg/m²'),
        'cholesterol_high': ('Colesterol', {0: 'Normal', 1: 'Alto'}),
        'gluc_high': ('Glicose', {0: 'Normal', 1: 'Alta'})
    }
    
    # Calcular score de contribuição (importância * valor normalizado)
    contributions = []
    for i, feature in enumerate(feature_names):
        value = patient_data[feature]
        importance = importances[i]
        
        # Para features binárias, só considerar se valor = 1 (fator de risco presente)
        # Para features numéricas, normalizar
        if feature in ['smoke', 'alco', 'cholesterol_high', 'gluc_high']:
            # Fatores de risco: só contribuem se presentes
            contribution = importance if value == 1 else 0
        elif feature == 'active':
            # Atividade física: contribui negativamente se sedentário
            contribution = importance if value == 0 else 0
        elif feature == 'gender':
            # Gênero: considera a importância
            contribution = importance * (value / 1.0)  # normalizar 0-1
        else:
            # Features numéricas: normalizar por faixas esperadas
            if feature == 'ap_hi':
                # Pressão sistólica: risco aumenta acima de 120
                contribution = importance * max(0, (value - 120) / 100)
            elif feature == 'ap_lo':
                # Pressão diastólica: risco aumenta acima de 80
                contribution = importance * max(0, (value - 80) / 50)
            elif feature == 'age_years':
                # Idade: risco aumenta com a idade
                contribution = importance * (value / 100)
            elif feature == 'bmi':
                # IMC: risco aumenta acima de 25
                contribution = importance * max(0, (value - 25) / 15)
            else:
                contribution = importance * value
        
        # Formatar valor para exibição
        label, formatter = feature_labels[feature]
        if isinstance(formatter, dict):
            display_value = formatter.get(value, str(value))
        else:
            display_value = formatter(value)
        
        contributions.append({
            'feature': feature,
            'label': label,
            'value': display_value,
            'importance': float(importance * 100),  # porcentagem
            'contribution': float(contribution)
        })
    
    # Ordenar por contribuição e pegar top 3
    contributions.sort(key=lambda x: x['contribution'], reverse=True)
    top_factors = contributions[:3]
    
    return top_factors


def get_model_info() -> Dict:
    """
    Retorna informações sobre o modelo treinado.
    
    Returns:
        dict: Metadados do pipeline (features, métricas, etc)
    """
    metadata_path = Path(__file__).parent / 'models' / 'pipeline_metadata.json'
    
    if not metadata_path.exists():
        return {
            'error': 'Metadados não encontrados',
            'pipeline_exists': get_model_path().exists()
        }
    
    import json
    with open(metadata_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Teste básico do serviço
    print("=" * 70)
    print("🧪 TESTE DO SERVIÇO DE PREDIÇÃO")
    print("=" * 70)
    
    # Exemplo 1: Paciente de baixo risco
    print("\n📋 Exemplo 1: Paciente de baixo risco")
    print("-" * 70)
    
    low_risk_patient = {
        'gender': 0,           # feminino
        'ap_hi': 110,          # pressão normal
        'ap_lo': 70,
        'smoke': 0,            # não fuma
        'alco': 0,             # não bebe
        'active': 1,           # ativo fisicamente
        'age_years': 35,       # jovem
        'bmi': 22.5,           # peso normal
        'cholesterol_high': 0, # colesterol normal
        'gluc_high': 0         # glicose normal
    }
    
    try:
        result = predict_single(low_risk_patient)
        print(f"✅ Predição bem-sucedida:")
        print(f"   Probabilidade: {result['probability']:.2f}%")
        print(f"   Risco: {result['risk_label']}")
        print(f"   Classe: {result['class']}")
        if 'risk_factors' in result:
            print(f"\n   📊 Principais fatores de risco:")
            for idx, factor in enumerate(result['risk_factors'], 1):
                print(f"      {idx}. {factor['label']}: {factor['value']} (Impacto: {factor['importance']:.1f}%)")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
    
    # Exemplo 2: Paciente de alto risco
    print("\n📋 Exemplo 2: Paciente de alto risco")
    print("-" * 70)
    
    high_risk_patient = {
        'gender': 1,           # masculino
        'ap_hi': 160,          # hipertensão
        'ap_lo': 100,
        'smoke': 1,            # fumante
        'alco': 1,             # consome álcool
        'active': 0,           # sedentário
        'age_years': 65,       # idoso
        'bmi': 32.0,           # obesidade
        'cholesterol_high': 1, # colesterol alto
        'gluc_high': 1         # glicose alta
    }
    
    try:
        result = predict_single(high_risk_patient)
        print(f"✅ Predição bem-sucedida:")
        print(f"   Probabilidade: {result['probability']:.2f}%")
        print(f"   Risco: {result['risk_label']}")
        print(f"   Classe: {result['class']}")
        if 'risk_factors' in result:
            print(f"\n   📊 Principais fatores de risco:")
            for idx, factor in enumerate(result['risk_factors'], 1):
                print(f"      {idx}. {factor['label']}: {factor['value']} (Impacto: {factor['importance']:.1f}%)")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
    
    # Informações do modelo
    print("\n📊 Informações do modelo")
    print("-" * 70)
    info = get_model_info()
    if 'error' not in info:
        print(f"Features: {info.get('n_features', 'N/A')}")
        print(f"Treinado em: {info.get('timestamp', 'N/A')}")
        if 'test_metrics' in info:
            metrics = info['test_metrics']
            print(f"\nMétricas de teste:")
            print(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            print(f"   Acurácia: {metrics.get('accuracy', 0):.4f}")
            print(f"   Precisão: {metrics.get('precision', 0):.4f}")
    else:
        print(f"⚠️ {info['error']}")
    
    print("\n" + "=" * 70)
