import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------
# Carregamento e pré-processamento
# -------------------------
df = pd.read_parquet('data/raw/base.parquet')

le_produto = LabelEncoder()
le_store = LabelEncoder()
df['produto_encoded'] = le_produto.fit_transform(df['produto_id'])
df['store_encoded'] = le_store.fit_transform(df['store_id'])

df['grupo'] = df['ano'].astype(str) + '_' + df['campanha'].astype(str).str.zfill(2)

X = df[[
    # 'ano', 
    'campanha', 'produto_encoded', 'store_encoded']]
y = df['demanda']
groups = df['grupo']

# -------------------------
# Definir busca de hiperparâmetros
# -------------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# GroupKFold
gkf = GroupKFold(n_splits=df['grupo'].nunique())

# Score negativo do MSE (padrão do sklearn para problemas de minimização)
neg_mse = make_scorer(mean_squared_error, greater_is_better=False)

# Estimador
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    scoring=neg_mse,
    cv=gkf.split(X, y, groups),
    n_iter=30,  # Você pode aumentar para maior qualidade
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# -------------------------
# Execução
# -------------------------
random_search.fit(X, y)

# -------------------------
# Resultados
# -------------------------
print("Melhores hiperparâmetros encontrados:")
print(random_search.best_params_)

print("\nMelhor score (MSE negativo):")
print(random_search.best_score_)

# Salvar resultados detalhados em CSV
resultados = pd.DataFrame(random_search.cv_results_)
resultados.to_csv("resultados_xgb_hiperparametros.csv", index=False)
