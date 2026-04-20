from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# model load
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study = float(request.form['study'])
    attendance = float(request.form['attendance'])
    sleep = float(request.form['sleep'])

    data = np.array([[study, attendance, sleep]])
    prediction = model.predict(data)

    return render_template('index.html', result=round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)