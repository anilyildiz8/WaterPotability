import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

data = pd.read_csv('dataset.csv')

X = data.drop('Potability', axis=1)
y = data['Potability']

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

losses = []
accuracies = []

for train_index, test_index in kfold.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.01, n_estimators=2000)

    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    loss = log_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    
    losses.append(loss)
    accuracies.append(accuracy)

avg_loss = np.mean(losses)
avg_accuracy = np.mean(accuracies)

print(f'Average Test Loss: {avg_loss}, Average Test Accuracy: {avg_accuracy}')

with open('xgboost_loss_accuracy.txt', 'a') as file:
    file.write(f'Average Test Loss: {avg_loss}, Average Test Accuracy: {avg_accuracy}\n')
