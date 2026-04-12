# Logistic Regression on Titanic Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Create simple dataset manually
data = {
    'Pclass': [1, 3, 2, 1, 3, 2, 1, 3],
    'Age': [22, 38, 26, 35, 28, 19, 40, 30],
    'Sex': [0, 1, 1, 1, 0, 0, 1, 0],  # male=0, female=1
    'Survived': [0, 1, 1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# Step 2: Features and target
X = df[['Pclass', 'Age', 'Sex']]
y = df['Survived']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test)

# Step 6: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))