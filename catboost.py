import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')

X = data.drop('Potability', axis=1)
y = data['Potability']

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

losses = []
accuracies = []
iteration = 0

for train_index, test_index in kfold.split(X_normalized):
    iteration += 1
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        eval_metric='Logloss',
        random_seed=42,
        logging_level='Silent'
    )

    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_test, y_test)

    model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    loss = log_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Iteration {iteration}: Test Loss: {loss}, Test Accuracy: {accuracy}')
    
    losses.append(loss)
    accuracies.append(accuracy)

avg_loss = np.mean(losses)
avg_accuracy = np.mean(accuracies)

print(f'Average Test Loss: {avg_loss}, Average Test Accuracy: {avg_accuracy}')

plt.figure(figsize=(8, 6))
plt.plot(range(1, iteration + 1), accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy on Each Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.xticks(range(1, iteration + 1))
plt.grid(True)
plt.show()

with open('catboost_loss_accuracy.txt', 'a') as file:
    file.write(f'Average Test Loss: {avg_loss}, Average Test Accuracy: {avg_accuracy}\n')

model.save_model('catboost_model.cbm')
