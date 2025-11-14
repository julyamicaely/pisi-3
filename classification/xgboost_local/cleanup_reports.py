"""
Script utilitário para limpar relatórios antigos do XGBoost.
Mantém apenas o relatório mais recente de cada tipo.
"""

from pathlib import Path
import glob

def clean_all_old_reports():
    """
    Remove TODOS os relatórios antigos, mantendo apenas os mais recentes.
    """
    reports_dir = Path(__file__).parent / '../reports'
    
    patterns = [
        'xgboost_confusion_matrix_*.png',
        'xgboost_roc_curve_*.png',
        'xgboost_precision_recall_*.png',
        'xgboost_metrics_report_*.txt',
        'xgboost_metrics_*.json',
        'xgboost_feature_importance_*.png',
        'xgboost_feature_importance_*.json'
    ]
    
    print("=" * 70)
    print("LIMPEZA DE RELATÓRIOS ANTIGOS - XGBOOST")
    print("=" * 70)
    
    print(f"\n📂 Diretório: {reports_dir}")
    
    total_removed = 0
    keep_latest = 1
    
    for pattern in patterns:
        files = sorted(glob.glob(str(reports_dir / pattern)))
        
        if len(files) > keep_latest:
            files_to_remove = files[:-keep_latest]
            pattern_name = pattern.replace('xgboost_', '').replace('_*.', '.')
            print(f"\n📊 {pattern_name}:")
            print(f"   Total: {len(files)} | Mantendo: {keep_latest} | Removendo: {len(files_to_remove)}")
            
            for file in files_to_remove:
                try:
                    Path(file).unlink()
                    total_removed += 1
                    print(f"   🗑️  {Path(file).name}")
                except Exception as e:
                    print(f"   ⚠️  Erro ao remover {Path(file).name}: {e}")
        elif len(files) == keep_latest:
            pattern_name = pattern.replace('xgboost_', '').replace('_*.', '.')
            print(f"\n📊 {pattern_name}: ✅ Apenas {len(files)} arquivo (OK)")
        elif len(files) > 0:
            pattern_name = pattern.replace('xgboost_', '').replace('_*.', '.')
            print(f"\n📊 {pattern_name}: ℹ️  {len(files)} arquivo(s) encontrado(s) (nenhum para remover)")
    
    print("\n" + "=" * 70)
    if total_removed > 0:
        print(f"✅ LIMPEZA CONCLUÍDA: {total_removed} arquivo(s) removido(s)")
    else:
        print("ℹ️  NENHUM ARQUIVO ANTIGO ENCONTRADO")
    print("=" * 70)

if __name__ == "__main__":
    clean_all_old_reports()
