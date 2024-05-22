import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

data = pd.read_csv('dataset.csv')

X = data.drop('Potability', axis=1)
y = data['Potability']

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

gb_losses = []
gb_accuracies = []

for train_index, test_index in kfold.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    
    gb_y_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    gb_y_pred = gb_model.predict(X_test)

    gb_loss = log_loss(y_test, gb_y_pred_proba)
    gb_accuracy = accuracy_score(y_test, gb_y_pred)
    
    print(f'Gradient Boosting - Test Loss: {gb_loss}, Test Accuracy: {gb_accuracy}')
    
    gb_losses.append(gb_loss)
    gb_accuracies.append(gb_accuracy)

avg_gb_loss = np.mean(gb_losses)
avg_gb_accuracy = np.mean(gb_accuracies)

print(f'Gradient Boosting - Average Test Loss: {avg_gb_loss}, Average Test Accuracy: {avg_gb_accuracy}')

with open('gb_loss_accuracy.txt', 'a') as file:
    file.write(f'Gradient Boosting - Average Test Loss: {avg_gb_loss}, Average Test Accuracy: {avg_gb_accuracy}\n')
