import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0'=todos, '1'=info, '2'=warning, '3'=error

# Carregamento dos dados
df = pd.read_parquet('data/raw/base.parquet')

# Codificação das variáveis categóricas
le_produto = LabelEncoder()
le_store = LabelEncoder()
df['produto_encoded'] = le_produto.fit_transform(df['produto_id'])
df['store_encoded'] = le_store.fit_transform(df['store_id'])

# Criação da coluna de grupo combinando 'ano' e 'campanha' com zero à esquerda
df['grupo'] = df['ano'].astype(str) + '_' + df['campanha'].astype(str).str.zfill(2)

# Definição das variáveis independentes e dependente
X = df[['ano', 'campanha', 'produto_encoded', 'store_encoded']]
y = df['demanda']
groups = df['grupo']

# Inicialização do GroupKFold com número de splits igual ao número de grupos únicos
n_splits = df['grupo'].nunique()
gkf = GroupKFold(n_splits=n_splits)

# Listas para armazenar os MSEs de cada modelo
mse_scores_lr = []
mse_scores_xgb = []
mse_scores_arima = []
mse_scores_lstm = []

# Loop sobre os folds
for train_idx, test_idx in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Regressão Linear
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mse_scores_lr.append(mse_lr)
    
    # XGBoost
    modelo_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    modelo_xgb.fit(X_train, y_train)
    y_pred_xgb = modelo_xgb.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mse_scores_xgb.append(mse_xgb)


# Cálculo da média dos MSEs
mse_medio_lr = np.mean(mse_scores_lr)
mse_medio_xgb = np.mean(mse_scores_xgb)


# Exibição dos resultados
print(f"Média do Erro Quadrático Médio (MSE) - Regressão Linear: {mse_medio_lr:.2f}")
print(f"Média do Erro Quadrático Médio (MSE) - XGBoost: {mse_medio_xgb:.2f}")
