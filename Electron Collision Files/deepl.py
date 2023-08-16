import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import time

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('dielectron.csv', na_values=['NA'])
data.drop(['Run', 'Event'], axis=1, inplace=True)
data.dropna(inplace=True)
print("Data loaded and preprocessed.")

# Scale the features using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data.drop('M', axis=1))

y = data['M']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize results list
results = []

optimizers = ['SGD', 'Adam']
extra_dense_layers = ['Yes', 'No']
last_layer_activations = ['relu', 'sigmoid']

# Create a table header
print("{:<10} {:<10} {:<20} {:<20}".format("Model Name", "Optimizer", "Extra Dense Layer", "Last Layer Activation"))
print("="*60)

# Iterate through combinations
for optimizer in optimizers:
    for dense_layer in extra_dense_layers:
        for last_activation in last_layer_activations:
            model_name = f"Model_{optimizer}{dense_layer}{last_activation.capitalize()}"
            
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(48, activation='relu') if dense_layer == 'Yes' else tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation=last_activation)
            ])
            
            model.compile(optimizer=optimizer.lower(), loss='mean_squared_error')
            
            print("{:<10} {:<10} {:<20} {:<20}".format(model_name, optimizer, dense_layer, last_activation))
            
            start_time = time.time()
            history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            results.append((model_name, mse))
            
            end_time = time.time()
            training_time = end_time - start_time
            print("Model trained. Training time:", training_time, "seconds")
            print("="*60)

# Print results
sorted_results = sorted(results, key=lambda x: x[1])
for result in sorted_results:
    print(f'{result[0]} - Error/Accuracy: {result[1]}')
