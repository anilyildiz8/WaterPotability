from flask import Flask, request, render_template
import numpy as np
import joblib
import json
from tensorflow.keras.models import model_from_json
import os

app = Flask(__name__)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

model.load_weights('model_fold_5.keras')

scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ph = float(request.form['ph'])
        hardness = float(request.form['hardness'])
        solids = float(request.form['solids'])
        chloramines = float(request.form['chloramines'])
        sulfate = float(request.form['sulfate'])
        conductivity = float(request.form['conductivity'])
        organic_carbon = float(request.form['organic_carbon'])
        trihalomethanes = float(request.form['trihalomethanes'])
        turbidity = float(request.form['turbidity'])
        
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        
        input_data = scaler.transform(input_data)
        
        prediction = model.predict(input_data)
        prediction = (prediction > 0.5).astype(int)
        
        return render_template('index.html', prediction_text=f'Potability: {"Potable" if prediction[0][0] == 1 else "Not Potable"}')

if __name__ == '__main__':
    app.run(debug=True)
