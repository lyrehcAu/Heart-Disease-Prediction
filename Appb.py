from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model_path = r"C:\Users\cheryl\Documents\Oding\BINUS\CODE S3\AI\AOL\model\HeartDisease_rf.pkl" 
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # AMBIL DATA YG DIISI
    age = int(request.form['age'])
    gender = 0
    if request.form['gender'] == 'male':
        gender = 1
    else:
        gender = 0
    resting_ecg = int(request.form['resting_ecg'])
    blood_pressure = int(request.form['blood_pressure'])
    chest_pain = 0 
    if request.form['chest_pain'] == 'typical':
        chest_pain = 1
    elif request.form['chest_pain'] == 'atypical_angina':
        chest_pain = 2
    elif request.form['chest_pain'] == 'non_anginal':
        chest_pain = 3
    elif request.form['chest_pain'] == 'asymptomatic':
        chest_pain = 4
    max_heart_rate = int(request.form['max_heart_rate'])
    cholesterol = int(request.form['cholesterol'])
    fasting_sugar = 0 
    if request.form['fasting_sugar'] == 'yes':
        fasting_sugar = 1
    else: 
        fasting_sugar = 0
    exercise_angina = 0
    if request.form['exercise_angina'] == 'yes':
        exercise_angina = 1
    else:
        exercise_angina = 0
    st_depression = float(request.form['st_depression'])
    st_slope = 0 
    if request.form['st_slope'] == 'normal':
        st_slope = 0
    elif request.form['st_slope'] == 'upsloping':
        st_slope = 1
    elif request.form['st_slope'] == 'flat':
        st_slope = 2
    elif request.form['st_slope'] == 'downsloping':
        st_slope = 3

    # PERSIAPAN DATA
    new_data = np.array([
        age, gender, chest_pain, blood_pressure, cholesterol, fasting_sugar, resting_ecg,
        max_heart_rate, exercise_angina, st_depression, st_slope
    ]).reshape(1, -1)

    # PREDIKSI
    prediction = model.predict(new_data)[0]

    # HASIL
    if prediction == 0:
        result = "No heart disease detected."
    else:
        result = "Heart disease detected."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
