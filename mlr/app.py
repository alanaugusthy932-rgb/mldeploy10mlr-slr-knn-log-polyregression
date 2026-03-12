from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', title="Multiple Linear Regression")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form.get('hours'))
        attendance = float(request.form.get('attendance'))
        prediction = model.predict([[hours, attendance]])[0]
        return render_template('index.html', 
                             title="Multiple Linear Regression",
                             prediction_text=f'Predicted Exam Score: {prediction:.2f}',
                             hours=hours,
                             attendance=attendance)
    except Exception as e:
        return render_template('index.html', title="Multiple Linear Regression", error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
