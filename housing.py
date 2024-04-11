import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Housing.csv")

print(df.isnull().sum())

binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))
df = pd.get_dummies(df, columns=['furnishingstatus'])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('price', axis=1))

X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['price'], test_size=0.2, random_state=42)

models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]

for model in models:
    model.fit(X_train, y_train)

for model in models:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model: {type(model).__name__}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print()
