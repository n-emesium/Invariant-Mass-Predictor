import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load and preprocess your data as before
data = pd.read_csv('dielectron.csv', na_values=['NA'])
data.drop(['Run', 'Event'], axis=1, inplace=True)
data.dropna(inplace=True)

X = data.drop('M', axis=1)
y = data['M']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors

# Train the model
knn_regressor.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn_regressor.predict(X_test)

# Calculate mean squared error
mse_knn = mean_squared_error(y_test, y_pred_knn)
print("KNN Mean Squared Error:", mse_knn)
