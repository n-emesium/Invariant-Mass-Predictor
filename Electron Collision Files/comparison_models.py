import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('dielectron.csv', na_values=['NA'])
data.drop(['Run', 'Event'], axis=1, inplace=True)
data.dropna(inplace=True)

X = data.drop('M', axis=1)
y = data['M']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

activation_functions = ['relu', 'sigmoid', 'tanh']
models = []

for activation in activation_functions:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=activation, input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(32, activation=activation),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    for epoch in range(1, 501):
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        if epoch % 50 == 0:
            print(f"Activation: {activation} - Epoch: {epoch}/{500}")

    y_pred = model.predict(X_test)
    y_binary = [1 if val >= 0.5 else 0 for val in y_test]
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

    accuracy = accuracy_score(y_binary, y_pred_binary)
    precision = precision_score(y_binary, y_pred_binary)
    recall = recall_score(y_binary, y_pred_binary)
    f1 = f1_score(y_binary, y_pred_binary)

    models.append({
        'Activation Function': activation,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    print(f'Activation: {activation} - Completed')

results_df = pd.DataFrame(models)
results_df.to_excel('model_results.xlsx', index=False)
