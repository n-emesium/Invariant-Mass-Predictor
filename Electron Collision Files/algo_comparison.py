import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import time

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('dielectron.csv', na_values=['NA'])
data.drop(['Run', 'Event'], axis=1, inplace=True)
data.dropna(inplace=True)
print("Data loaded and preprocessed.")

X = data.drop('M', axis=1)
y = data['M']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize results list
results = []

# MLP model
print("Training MLP...")
model_mlp = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_mlp.compile(optimizer='adam', loss='mean_squared_error')
start_time = time.time()
history_mlp = model_mlp.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)
y_pred_mlp = model_mlp.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
results.append(('MLP', mse_mlp))
end_time = time.time()
mlp_training_time = end_time - start_time
print("MLP trained. Training time:", mlp_training_time, "seconds")

# CNN model
print("Training CNN...")
X_train_cnn = X_train.values.reshape((-1, X_train.shape[1], 1))
X_test_cnn = X_test.values.reshape((-1, X_test.shape[1], 1))
model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_cnn.compile(optimizer='adam', loss='mean_squared_error')
start_time = time.time()
history_cnn = model_cnn.fit(X_train_cnn, y_train, epochs=100, batch_size=32, verbose=0)
y_pred_cnn = model_cnn.predict(X_test_cnn)
mse_cnn = mean_squared_error(y_test, y_pred_cnn)
results.append(('CNN', mse_cnn))
end_time = time.time()
cnn_training_time = end_time - start_time
print("CNN trained. Training time:", cnn_training_time, "seconds")

# RNN model
print("Training RNN...")
X_train_rnn = X_train.values.reshape((-1, X_train.shape[1], 1))
X_test_rnn = X_test.values.reshape((-1, X_test.shape[1], 1))
model_rnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(X_train_rnn.shape[1], 1)),
    tf.keras.layers.Dense(1)
])
model_rnn.compile(optimizer='adam', loss='mean_squared_error')
start_time = time.time()
history_rnn = model_rnn.fit(X_train_rnn, y_train, epochs=100, batch_size=32, verbose=0)
y_pred_rnn = model_rnn.predict(X_test_rnn)
mse_rnn = mean_squared_error(y_test, y_pred_rnn)
results.append(('RNN', mse_rnn))
end_time = time.time()
rnn_training_time = end_time - start_time
print("RNN trained. Training time:", rnn_training_time, "seconds")

# LSTM model
print("Training LSTM...")
model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(X_train_rnn.shape[1], 1)),
    tf.keras.layers.Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
start_time = time.time()
history_lstm = model_lstm.fit(X_train_rnn, y_train, epochs=100, batch_size=32, verbose=0)
y_pred_lstm = model_lstm.predict(X_test_rnn)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
results.append(('LSTM', mse_lstm))
end_time = time.time()
lstm_training_time = end_time - start_time
print("LSTM trained. Training time:", lstm_training_time, "seconds")

# Print results
print("Training times:")
print("MLP:", mlp_training_time, "seconds")
print("CNN:", cnn_training_time, "seconds")
print("RNN:", rnn_training_time, "seconds")
print("LSTM:", lstm_training_time, "seconds")

sorted_results = sorted(results, key=lambda x: x[1])
for result in sorted_results:
    print(f'{result[0]} - Error/Accuracy: {result[1]}')
