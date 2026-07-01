import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
df=pd.read_csv("house-price-prediction/data/raw/train.csv")
df.head()
df.shape
df.columns
df.dtypes
df.isnull().sum()
df.duplicated().sum()
df.describe()
df.isnull().sum().sort_values(ascending=False).head(20)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

print("Numerical columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
df.isnull().sum().sum()
#2
le=LabelEncoder()
for col in cat_cols:
  df[col]=le.fit_transform(df[col])
df.head()
#3
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)
#4
model=LinearRegression()
model.fit(X_train,y_train)      
y_pred=model.predict(X_test)
y_pred[:5]
#5
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)
#6
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred=rf_model.predict(X_test)
rf_pred[:5]
#7
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, y_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)
#8
joblib.dump(rf_model, "house-price-prediction.pkl")
id_test=X_test.iloc[0]
prediction=rf_model.predict([id_test])
print("Predicted Price:",prediction[0])
print("Actual Price:",y_test.iloc[0])
#8
joblib.dump(rf_model, "house-price-prediction/models/house_price_model.pkl")
#done
print("Model trained and saved succesfly!")
