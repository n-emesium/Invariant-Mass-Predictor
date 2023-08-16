#Hücre 1


import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('dielectron.csv', na_values=['NA'])
data.drop(['Run', 'Event'], axis=1, inplace=True)
data.dropna(inplace=True)

X = data.drop('M', axis=1)
y = data['M']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

activation_functions = ['relu', 'sigmoid', 'tanh']

for activation in activation_functions:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=activation, input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(32, activation=activation),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Activation Function:", activation)
    print("Mean Squared Error:", mse)
    print()

    # Extract the mean squared error from the history
    mse_values = history.history['loss']

    # Plotting the graph
    epochs = range(1, len(mse_values) + 1)
    plt.plot(epochs, mse_values, marker='o')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Epoch vs. Mean Squared Error')

    # Displaying the graph
    plt.show()