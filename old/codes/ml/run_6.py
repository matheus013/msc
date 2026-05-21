import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------
# Carregamento dos dados
# -------------------------
print("🔄 Carregando dados...")
df = pd.read_parquet('data/raw/base.parquet')

# Codificação dos atributos categóricos
le_produto = LabelEncoder()
le_store = LabelEncoder()
df['produto_encoded'] = le_produto.fit_transform(df['produto_id'])
df['store_encoded'] = le_store.fit_transform(df['store_id'])

# Criação da coluna de grupo e grupo ordinal (para ordenação temporal)
df['grupo'] = df['ano'].astype(str) + '_' + df['campanha'].astype(str).str.zfill(2)
df['grupo_ordinal'] = df['ano'] * 100 + df['campanha']  # ex: 202304

# Ordenar grupos para simular tempo
unique_groups = sorted(df['grupo_ordinal'].unique())

# Definição de features
X_all = df[['ano', 'campanha', 'produto_encoded', 'store_encoded']]
y_all = df['demanda']
df['grupo_ordinal'] = df['grupo_ordinal'].astype(int)

# -------------------------
# Hiperparâmetros otimizados
# -------------------------
best_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 1.0,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'verbosity': 0
}

# -------------------------
# Validação Temporal
# -------------------------
print("📈 Iniciando avaliação temporal...")

resultados = []
for i in range(3, len(unique_groups)):
    treino_ate = unique_groups[i - 1]
    teste_grupo = unique_groups[i]

    treino_mask = df['grupo_ordinal'] <= treino_ate
    teste_mask = df['grupo_ordinal'] == teste_grupo

    X_train, y_train = X_all[treino_mask], y_all[treino_mask]
    X_test, y_test = X_all[teste_mask], y_all[teste_mask]
    grupo_nome = df.loc[teste_mask, 'grupo'].iloc[0]

    if len(y_test) == 0:
        continue  # pular se não houver teste nesse grupo

    modelo = XGBRegressor(**best_params)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"✅ Avaliado grupo {grupo_nome} | MSE: {mse:.2f}")
    resultados.append({'grupo': grupo_nome, 'mse_xgboost': mse})

# -------------------------
# Resultados
# -------------------------
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('grupo')
print("\n📊 Resultados Finais:")
print(df_resultados)

# Salvar para CSV
df_resultados.to_csv('resultados_xgboost_temporal.csv', index=False)
print("\n💾 Resultados salvos em 'resultados_xgboost_temporal.csv'")
