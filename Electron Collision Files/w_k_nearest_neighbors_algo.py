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

# Create a Weighted KNN regressor with distance-based weights
weighted_knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')
# You can adjust the number of neighbors and other parameters

# Train the model
weighted_knn_regressor.fit(X_train, y_train)

# Make predictions
y_pred_weighted_knn = weighted_knn_regressor.predict(X_test)

# Calculate mean squared error
mse_weighted_knn = mean_squared_error(y_test, y_pred_weighted_knn)
print("Weighted KNN Mean Squared Error:", mse_weighted_knn)
