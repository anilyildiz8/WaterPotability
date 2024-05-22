import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from keras import layers, models
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')

X = data.drop('Potability', axis=1)
y = data['Potability']

X_mean = X.mean()
X_std = X.std()

X_normalized = (X - X_mean) / X_std

learning_rate = 0.001

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

avg_val_losses = []
avg_val_accuracies = []

for train_index, test_index in kfold.split(X_normalized):
    X_train, X_test = X_normalized.iloc[train_index], X_normalized.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3), 
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3), 
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer , loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    
    avg_val_losses.append(val_loss)
    avg_val_accuracies.append(val_accuracy)

avg_val_losses = np.mean(avg_val_losses, axis=0)
avg_val_accuracies = np.mean(avg_val_accuracies, axis=0)

plt.figure(figsize=(8, 6))
plt.plot(avg_val_losses, label='Validation Loss', color='blue')
plt.title('Average Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(avg_val_accuracies, label='Validation Accuracy', color='green')
plt.title('Average Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
