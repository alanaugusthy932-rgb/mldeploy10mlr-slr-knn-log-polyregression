from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
poly_path = os.path.join(os.path.dirname(__file__), 'poly.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(poly_path, 'rb') as f:
    poly = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', title="Polynomial Regression")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form.get('hours'))
        poly_hours = poly.transform([[hours]])
        prediction = model.predict(poly_hours)[0]
        return render_template('index.html', 
                             title="Polynomial Regression",
                             prediction_text=f'Predicted Exam Score: {prediction:.2f}',
                             hours=hours)
    except Exception as e:
        return render_template('index.html', title="Polynomial Regression", error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
