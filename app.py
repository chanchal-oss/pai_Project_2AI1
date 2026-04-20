from flask import Flask, render_template, request, redirect
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# 👉 Home automatically redirects to your named URL
@app.route('/')
def home():
    return redirect('/chanchal-linear-regression')

# 👉 Your CUSTOM URL (name included)
@app.route('/chanchal-linear-regression')
def main_page():
    return render_template('index.html')

# 👉 Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        study = float(request.form['study'])
        attendance = float(request.form['attendance'])
        sleep = float(request.form['sleep'])

        data = np.array([[study, attendance, sleep]])
        prediction = model.predict(data)

        result = round(prediction[0], 2)

    except:
        result = "Invalid Input!"

    return render_template('index.html', result=result)

# Run app
if __name__ == "__main__":
    app.run(debug=True)