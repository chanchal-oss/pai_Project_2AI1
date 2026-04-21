import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#linear_regression
#dataset of ....
# Dataset
#satart...
data = {
    'StudyHours': [2, 4, 6, 8, 3, 5, 7, 9],
    'Attendance': [60, 70, 80, 90, 65, 75, 85, 95],
    'SleepHours': [5, 6, 7, 8, 5, 6, 7, 8],
    'Marks': [50, 60, 70, 85, 55, 65, 75, 90]
}

df = pd.DataFrame(data)

X = df[['StudyHours', 'Attendance', 'SleepHours']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)