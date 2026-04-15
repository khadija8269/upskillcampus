# app.py

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/crop_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        values = [

            float(request.form['year1']),
            float(request.form['year2']),
            float(request.form['year3']),
            float(request.form['year4']),
            float(request.form['year5'])

        ]

        features = np.array([values])

        prediction = model.predict(features)

        return render_template(

            'index.html',

            prediction_text=
            f"📈 Predicted Next Year Production: {prediction[0]:.2f}"

        )

    except:

        return render_template(

            'index.html',

            prediction_text="❌ Invalid Input"

        )


if __name__ == "__main__":
    app.run(debug=True)