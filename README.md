# Heart Disease Predictor

A simple web application that predicts heart disease risk using machine learning.

## Architecture

```
[User Browser]
   ⬇️⬆️ (HTTP)
[HTML/JS Frontend]
   ⬇️⬆️ (POST JSON)
[Flask API]
   ⬇️
[Scikit-learn Model]
```

## Setup

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Start the Flask backend:
```bash
cd backend
python3 app.py
```

3. Open the frontend:
- Open `frontend/index.html` in your browser
- Or serve it using Python: `python3 -m http.server 8080` from the frontend directory

## Usage

1. Fill in the health metrics form
2. Click "Predict Risk"
3. View the prediction result

## Features Used

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope of Peak Exercise ST Segment
- Number of Major Vessels
- Thalassemia

## Deployment

For production deployment on Render or Heroku:

1. Create a `Procfile`:
```
web: gunicorn --chdir backend app:app
```

2. Update the API_URL in `frontend/index.html` to your deployed backend URL.