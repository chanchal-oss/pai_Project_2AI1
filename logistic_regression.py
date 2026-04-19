import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib   # for saving model

# Step 1: Dataset (Student Performance)
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 6, 7, 9, 10, 3, 5],
    'Attendance': [50, 60, 65, 70, 75, 80, 85, 90, 55, 68, 82, 88, 92, 95, 66, 78],
    'PreviousMarks': [40, 45, 50, 55, 60, 65, 70, 75, 48, 52, 67, 72, 80, 85, 53, 62],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Step 2: Features and target
X = df[['StudyHours', 'Attendance', 'PreviousMarks']]
y = df['Pass']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Save model and scaler (IMPORTANT for Flask)
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Step 10: Predict New Data (FIXED - no warning)
new_data = pd.DataFrame([[4, 72, 58]], 
                        columns=['StudyHours', 'Attendance', 'PreviousMarks'])

new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)
print("Prediction (0=Fail, 1=Pass):", prediction[0])