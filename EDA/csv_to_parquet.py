import pandas as pd

# Arquivos
csv_file = './EDA/cardio_data.csv'
parquet_file = './EDA/cardio_data.parquet'

# Ler CSV e salvar em Parquet
df = pd.read_csv(csv_file)
df.to_parquet(parquet_file, index=False)

print(f'Arquivo convertido com sucesso: {parquet_file}')
