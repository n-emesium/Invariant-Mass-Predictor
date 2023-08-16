import time
import pandas as pd
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dielectron.csv', na_values=['NA'])
df.drop(['Run', 'Event'], axis=1, inplace=True)
df.dropna(inplace=True)

x = df.drop(columns=['M']) 
y = df['M']
st_x= StandardScaler()  

x= st_x.fit_transform(x) 

rs = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = rs)


# Defining different configurations
configs = [
    {"learning_rate": 0.01, "batch_size": 32, "hidden_layers": 2},
    {"learning_rate": 0.01, "batch_size": 32, "hidden_layers": 3},
    {"learning_rate": 0.01, "batch_size": 64, "hidden_layers": 2},
    {"learning_rate": 0.01, "batch_size": 64, "hidden_layers": 3},
    {"learning_rate": 0.001, "batch_size": 32, "hidden_layers": 2},
    {"learning_rate": 0.001, "batch_size": 32, "hidden_layers": 3},
    {"learning_rate": 0.001, "batch_size": 64, "hidden_layers": 2},
    {"learning_rate": 0.001, "batch_size": 64, "hidden_layers": 3}
]

# A DataFrame to store the results
results_df = pd.DataFrame(columns=["Model Name", "Mean Squared Error", "Training Time (seconds)"])

# Iterate through configurations and create models
for idx, config in enumerate(configs):
    model_name = f"Model_LR{config['learning_rate']}_BS{config['batch_size']}_HL{config['hidden_layers']}"
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu')
    ])
    
    for _ in range(config['hidden_layers'] - 1):
        model.add(tf.keras.layers.Dense(32, activation='relu'))
    
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    start_time = time.time()
    model.fit(x_train, y_train, epochs=500, batch_size=config['batch_size'])
    training_time = time.time() - start_time

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    
    results_df.loc[idx] = [model_name, mse, training_time]

print(results_df)