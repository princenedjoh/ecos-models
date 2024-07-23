import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Step 1: Load the data
data = pd.read_csv('airqualityDataset.csv')

# Step 2: Data Preprocessing
# Handle missing values if any
data = data.dropna()

# Convert date to datetime format if needed
data['date'] = pd.to_datetime(data['date'])

# Step 3: Feature Selection
features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
X = data[features]
y = data['aqi']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Selection and Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Step 8: Make Predictions
sample_data = [[213.62,0.01,3.94,88.69,1.89,0.5,0.71,1.11]]  # Example sample data
predicted_aqi = model.predict(sample_data)
print(f'Predicted AQI: {predicted_aqi[0]}')

pickle.dump(model, open('model.pkl','wb'))








#random forest
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# Model training
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

# Evaluate the model
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

pickle.dump(rfr, open('rfr.pkl','wb'))
