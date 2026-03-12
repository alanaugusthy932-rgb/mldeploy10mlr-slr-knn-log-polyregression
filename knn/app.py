from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', title="KNN Classification")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form.get('hours'))
        sleep = float(request.form.get('sleep'))
        prediction = model.predict([[hours, sleep]])[0]
        result = "PASS" if prediction == 1 else "FAIL"
        
        return render_template('index.html', 
                             title="KNN Classification",
                             prediction_text=f'Prediction: {result}',
                             hours=hours,
                             sleep=sleep)
    except Exception as e:
        return render_template('index.html', title="KNN Classification", error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
