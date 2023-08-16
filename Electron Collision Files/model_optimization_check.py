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
learning_rates = [0.001, 0.01, 0.1]  # Add the learning rates you want to test
models = []

for activation in activation_functions:
    for learning_rate in learning_rates:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, activation=activation, input_shape=(X_train.shape[1],)))
        model.add(tf.keras.layers.Dense(64, activation=activation))
        model.add(tf.keras.layers.Dense(32, activation=activation))
        model.add(tf.keras.layers.Dense(16, activation=activation))
        model.add(tf.keras.layers.Dense(1))
        
        add_layer = True  # Set this to True or False based on whether you want to use dropout
        model.add(tf.keras.layers.Dropout(0.5)) if add_layer else None  # Example dropout usage
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        history = model.fit(X_train, y_train, epochs=500, batch_size=32)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        
        models.append({
            'Model Name': f'Model with {activation} activation',
            'Dropout': 'Yes' if add_layer else 'No',
            'Learning Rate': learning_rate,
            'Optimizer': 'Adam'
        })

results_df = pd.DataFrame(models)
results_df.to_excel('optimized_models_parameters.xlsx', index=False)
