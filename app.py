from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    # data from the request (JSON format)
    data = request.get_json()  
    
    
    features = np.array([[
        float(data['nitrogen']),
        float(data['phosphorus']),
        float(data['potassium']),
        float(data['ph']),
        float(data['temperature']),
        float(data['humidity']),
        float(data['rainfall'])
    ]])
    
    # Making the prediction
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
