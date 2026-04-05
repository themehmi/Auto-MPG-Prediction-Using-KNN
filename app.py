from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and the scaler
# Ensure these files are in the same directory as app.py
model = joblib.load('model2.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in the correct order: 
        # cylinders, displacement, horsepower, weight, acceleration, model year, origin
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        
        # Scale the data before prediction
        scaled_features = scaler.transform(final_features)
        
        # Predict using the k-NN regressor
        prediction = model.predict(scaled_features)
        output = round(prediction[0], 2)

        return render_template('index.html', 
                               prediction_text=f'Predicted Efficiency: {output} MPG')
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Please check your input values.")

if __name__ == "__main__":
    app.run(debug=True)