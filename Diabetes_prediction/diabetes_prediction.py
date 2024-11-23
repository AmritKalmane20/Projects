# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the Datasetc:\Users\Amrit Kalmane\Downloads\diabetes.csv
data = pd.read_csv('C:/Users/Amrit Kalmane/Downloads/diabetes.csv')

# Data Exploration
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Replace zeros with NaN in specific columns (except 'Pregnancies' and 'Outcome')
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_to_replace] = data[columns_to_replace].replace(0, np.nan)

# Fill missing values with the median
data.fillna(data.median(), inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('Outcome', axis=1))

# Prepare Features and Target
X = pd.DataFrame(scaled_data, columns=data.columns[:-1])
y = data['Outcome']

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the Model
joblib.dump(model, 'diabetes_prediction_model.pkl')
print("\nModel saved as 'diabetes_prediction_model.pkl'")
