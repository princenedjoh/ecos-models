import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics


data = pd.read_csv("sampled_points.csv")

data = np.array(data)

# data cleaning
df = pd.DataFrame(data)
df_cleaned = df.fillna(0)
data_cleaned = df_cleaned.to_numpy()

#data splitting
X = data_cleaned[:, 0:-1]
y = data_cleaned[:, -1]
y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print("Mean Squared Error (MSE):", mse)
# print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

pickle.dump(model, open('model.pkl','wb'))
