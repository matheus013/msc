import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------
# Carregamento dos dados
# -------------------------
df = pd.read_parquet('data/raw/base.parquet')

# Codificação dos atributos categóricos
le_produto = LabelEncoder()
le_store = LabelEncoder()
df['produto_encoded'] = le_produto.fit_transform(df['produto_id'])
df['store_encoded'] = le_store.fit_transform(df['store_id'])

# Criação da coluna de grupo
df['grupo'] = df['ano'].astype(str) + '_' + df['campanha'].astype(str).str.zfill(2)

# Definição de X, y, groups
X = df[['ano', 'campanha', 'produto_encoded', 'store_encoded']]
y = df['demanda']
groups = df['grupo']

# -------------------------
# Hiperparâmetros otimizados (exemplo: substitua pelos seus)
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
# Avaliação com GroupKFold
# -------------------------
gkf = GroupKFold(n_splits=df['grupo'].nunique())
mse_scores = []
grupos_avaliados = []

for train_idx, test_idx in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    grupo_teste = groups.iloc[test_idx].unique()[0]

    modelo_xgb = XGBRegressor(**best_params)
    modelo_xgb.fit(X_train, y_train)
    y_pred = modelo_xgb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mse_scores.append(mse)
    grupos_avaliados.append(grupo_teste)

# -------------------------
# Resultados
# -------------------------
df_resultados = pd.DataFrame({
    'grupo': grupos_avaliados,
    'mse_xgboost': mse_scores
})

# Exibir e salvar resultados
print(df_resultados)
df_resultados.to_csv('resultados_xgboost_final.csv', index=False)
