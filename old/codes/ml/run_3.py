import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Suprimir logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Carregamento dos dados
df = pd.read_parquet('data/raw/base.parquet')

# Codificação das variáveis categóricas
le_produto = LabelEncoder()
le_store = LabelEncoder()
df['produto_encoded'] = le_produto.fit_transform(df['produto_id'])
df['store_encoded'] = le_store.fit_transform(df['store_id'])

# Criação da coluna de grupo
df['grupo'] = df['ano'].astype(str) + '_' + df['campanha'].astype(str).str.zfill(2)

# Features e target
X = df[['ano', 'campanha', 'produto_encoded', 'store_encoded']]
y = df['demanda']
groups = df['grupo']
unique_grupos = df['grupo'].unique()

# Cross-validation
n_splits = len(unique_grupos)
gkf = GroupKFold(n_splits=n_splits)

# MSEs por modelo
mse_lr, mse_ridge, mse_lasso = [], [], []
mse_knn, mse_rf, mse_gb, mse_xgb = [], [], [], []

print(f"🔄 Iniciando validação cruzada com {n_splits} folds...\n")

for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"🔹 Fold {i+1} - Grupo: {unique_grupos[i]}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    mse = mean_squared_error(y_test, lr.predict(X_test))
    mse_lr.append(mse)
    print(f"  📈 Linear Regression MSE: {mse:.4f}")

    # Ridge
    ridge = Ridge().fit(X_train, y_train)
    mse = mean_squared_error(y_test, ridge.predict(X_test))
    mse_ridge.append(mse)
    print(f"  📈 Ridge Regression MSE: {mse:.4f}")

    # Lasso
    lasso = Lasso().fit(X_train, y_train)
    mse = mean_squared_error(y_test, lasso.predict(X_test))
    mse_lasso.append(mse)
    print(f"  📈 Lasso Regression MSE: {mse:.4f}")

    # KNN
    knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    mse = mean_squared_error(y_test, knn.predict(X_test))
    mse_knn.append(mse)
    print(f"  📈 KNN MSE: {mse:.4f}")

    # Random Forest
    # rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    # mse = mean_squared_error(y_test, rf.predict(X_test))
    # mse_rf.append(mse)
    # print(f"  🌲 Random Forest MSE: {mse:.4f}")

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    mse = mean_squared_error(y_test, gb.predict(X_test))
    mse_gb.append(mse)
    print(f"  📊 Gradient Boosting MSE: {mse:.4f}")

    # XGBoost
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    mse = mean_squared_error(y_test, xgb.predict(X_test))
    mse_xgb.append(mse)
    print(f"  ⚡ XGBoost MSE: {mse:.4f}\n")

# Criar DataFrame de resultados
df_resultados = pd.DataFrame({
    'grupo': unique_grupos,
    'Linear': mse_lr,
    'Ridge': mse_ridge,
    'Lasso': mse_lasso,
    'KNN': mse_knn,
    # 'RandomForest': mse_rf,
    'GradientBoost': mse_gb,
    'XGBoost': mse_xgb
})

# Exibir e salvar
print("✅ Resultados finais por grupo:\n")
print(df_resultados)

# Salvando
output_path = 'results/mse_por_modelo.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_resultados.to_csv(output_path, index=False)
print(f"\n📁 Resultados salvos em: {output_path}")
