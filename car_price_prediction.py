

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
# Pour outliers avancés :
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory




df = pd.read_csv("bmw.csv")
df_encoded = pd.get_dummies(df, columns=['model', 'transmission', 'fuelType', 'engineSize','mpg','mileage','tax'])

# Features & target
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Spliting train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# traîning the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# prediction over a handset 
y_pred = model.predict(X_test)
# Evaluating of the performances
print('MAE :', mean_absolute_error(y_test, y_pred))
print('MSE :', mean_squared_error(y_test, y_pred))
print('R² :', r2_score(y_test, y_pred))

pred_df = X_test.copy()
pred_df['price_true'] = y_test
pred_df['price_pred'] = y_pred
print(pred_df[['price_true', 'price_pred']].head())
