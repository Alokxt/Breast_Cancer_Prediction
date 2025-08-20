from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.decomposition import PCA


# Load the trained model
model_path = 'model2.pkl'
scalar = 'scalar2.pkl'
pcs = 'PCAs2.pkl'

with open(model_path, 'rb') as file:
    svm = pickle.load(file)
with open(scalar,'rb') as fL:
    sc = pickle.load(fL)
with open(pcs,'rb') as f:
    pc = pickle.load(f)




app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form

    int_features = [float(x) for x in request.form.values()]

    int_features2 = np.array(int_features).reshape(1,-1)


    # Make prediction
    x_centred = sc.transform(int_features2)
    x_transformed = x_centred.dot(pc)
    prediction = svm.predict(x_transformed)
    output = "⚠️ Malignant tumor detected (Breast Cancer)." if prediction[0] == 1 else "✅ Benign tumor detected (No signs of breast cancer)."

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)