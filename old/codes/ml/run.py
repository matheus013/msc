import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Suponha que 'df' seja o seu DataFrame com as colunas: 'ano', 'campanha', 'produto_id', 'demanda'
# Exemplo de criação do DataFrame:
df = pd.read_parquet('data/raw/base.parquet')

# Preprocessamento
df = df.copy()
df['ano'] = df['ano'].astype(int)
df['campanha'] = df['campanha'].astype(int)
df['produto_id'] = df['produto_id'].astype(str)

# Codificação de variáveis categóricas
le_produto = LabelEncoder()
df['produto_encoded'] = le_produto.fit_transform(df['produto_id'])

# Features e target
X = df[['ano', 'campanha', 'produto_encoded']]
y = df['demanda']

# Divisão em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Regressão Linear
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 2. ARIMA
# Para ARIMA, precisamos de uma série temporal univariada. Vamos agrupar por ano e campanha.
df_arima = df.groupby(['ano', 'campanha'])['demanda'].sum().reset_index()
df_arima['periodo'] = df_arima['ano'].astype(str) + '-' + df_arima['campanha'].astype(str)
df_arima.set_index('periodo', inplace=True)
model_arima = ARIMA(df_arima['demanda'], order=(1,1,1))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=5)

# 3. LSTM
# Preparação dos dados para LSTM
# Agrupar por ano e campanha e normalizar
df_lstm = df.groupby(['ano', 'campanha'])['demanda'].sum().reset_index()
scaler = MinMaxScaler()
df_lstm['demanda_scaled'] = scaler.fit_transform(df_lstm[['demanda']])

# Criar sequências
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 3
X_lstm, y_lstm = create_sequences(df_lstm['demanda_scaled'].values, seq_length)

# Redimensionar para [amostras, tempo, características]
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

# Dividir em treino e teste
X_train_lstm, X_test_lstm = X_lstm[:-1], X_lstm[-1:]
y_train_lstm, y_test_lstm = y_lstm[:-1], y_lstm[-1:]

# Construir o modelo LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

# Treinar o modelo
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=200, verbose=0)

# Previsão
y_pred_lstm = model_lstm.predict(X_test_lstm)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

# Exibir resultados
print("Previsão Regressão Linear:", y_pred_lr[:5])
print("Previsão ARIMA:", forecast_arima)
print("Previsão LSTM:", y_pred_lstm.flatten())
