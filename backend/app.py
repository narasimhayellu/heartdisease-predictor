from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('../ml/heart_disease_model.pkl')
scaler = joblib.load('../ml/scaler.pkl')
feature_names = joblib.load('../ml/feature_names.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        features = []
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing required field: {feature}'}), 400
            features.append(float(data[feature]))
        
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'risk': 'High' if prediction == 1 else 'Low',
            'probability': {
                'low_risk': float(probability[0]),
                'high_risk': float(probability[1])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/features', methods=['GET'])
def get_features():
    feature_info = {
        'age': {'type': 'number', 'label': 'Age', 'min': 20, 'max': 80},
        'sex': {'type': 'select', 'label': 'Sex', 'options': [{'value': 0, 'label': 'Female'}, {'value': 1, 'label': 'Male'}]},
        'cp': {'type': 'select', 'label': 'Chest Pain Type', 'options': [
            {'value': 0, 'label': 'Typical Angina'},
            {'value': 1, 'label': 'Atypical Angina'},
            {'value': 2, 'label': 'Non-Anginal Pain'},
            {'value': 3, 'label': 'Asymptomatic'}
        ]},
        'trestbps': {'type': 'number', 'label': 'Resting Blood Pressure (mm Hg)', 'min': 90, 'max': 200},
        'chol': {'type': 'number', 'label': 'Cholesterol (mg/dl)', 'min': 125, 'max': 400},
        'fbs': {'type': 'select', 'label': 'Fasting Blood Sugar > 120 mg/dl', 'options': [{'value': 0, 'label': 'No'}, {'value': 1, 'label': 'Yes'}]},
        'restecg': {'type': 'select', 'label': 'Resting ECG Results', 'options': [
            {'value': 0, 'label': 'Normal'},
            {'value': 1, 'label': 'ST-T Wave Abnormality'},
            {'value': 2, 'label': 'Left Ventricular Hypertrophy'}
        ]},
        'thalach': {'type': 'number', 'label': 'Maximum Heart Rate', 'min': 70, 'max': 200},
        'exang': {'type': 'select', 'label': 'Exercise Induced Angina', 'options': [{'value': 0, 'label': 'No'}, {'value': 1, 'label': 'Yes'}]},
        'oldpeak': {'type': 'number', 'label': 'ST Depression', 'min': 0, 'max': 6, 'step': 0.1},
        'slope': {'type': 'select', 'label': 'Slope of Peak Exercise ST Segment', 'options': [
            {'value': 0, 'label': 'Upsloping'},
            {'value': 1, 'label': 'Flat'},
            {'value': 2, 'label': 'Downsloping'}
        ]},
        'ca': {'type': 'number', 'label': 'Number of Major Vessels', 'min': 0, 'max': 3},
        'thal': {'type': 'select', 'label': 'Thalassemia', 'options': [
            {'value': 0, 'label': 'Normal'},
            {'value': 1, 'label': 'Fixed Defect'},
            {'value': 2, 'label': 'Reversible Defect'}
        ]}
    }
    return jsonify(feature_info)

if __name__ == '__main__':
    app.run(debug=True, port=5001)