import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('dielectron.csv', na_values=['NA'])  # Specify NA as missing value indicator if necessary
data.drop(['Run', 'Event'], axis=1, inplace=True)  # Drop 'Run' and 'Event' columns
data.dropna(inplace=True)  # Drop rows with missing values, if any

X = data.drop('M', axis=1)
y = data['M']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(1)  # Output layer with 1 neuron for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=500, batch_size=32)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Extract the mean squared error from the history
mse = history.history['loss']

# Plotting the graph
epochs = range(1, len(mse) + 1)
plt.plot(epochs, mse, marker='o')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Epoch vs. Mean Square Error')

# Displaying the graph
plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = round(mse, 2)
print("Mean Squared Error:", rmse)
