import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

np.random.seed(42)

n_samples = 1000
data = pd.DataFrame({
    'age': np.random.randint(20, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),
    'trestbps': np.random.randint(90, 200, n_samples),
    'chol': np.random.randint(125, 400, n_samples),
    'fbs': np.random.randint(0, 2, n_samples),
    'restecg': np.random.randint(0, 3, n_samples),
    'thalach': np.random.randint(70, 200, n_samples),
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6, n_samples).round(1),
    'slope': np.random.randint(0, 3, n_samples),
    'ca': np.random.randint(0, 4, n_samples),
    'thal': np.random.randint(0, 3, n_samples)
})

risk_score = (
    (data['age'] > 50).astype(int) * 0.3 +
    (data['trestbps'] > 140).astype(int) * 0.2 +
    (data['chol'] > 240).astype(int) * 0.2 +
    (data['fbs'] == 1).astype(int) * 0.1 +
    (data['thalach'] < 120).astype(int) * 0.2 +
    np.random.uniform(0, 0.3, n_samples)
)
data['target'] = (risk_score > 0.5).astype(int)

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'ml/heart_disease_model.pkl')
joblib.dump(scaler, 'ml/scaler.pkl')
joblib.dump(list(X.columns), 'ml/feature_names.pkl')

print("\nModel saved successfully!")
print("Features used:", list(X.columns))