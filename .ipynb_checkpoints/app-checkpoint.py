from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import cv2
import os
import scipy

app = Flask(__name__)

# Load the saved models
diabetes_model = joblib.load('static/diabetes_rf_model.pkl')
tumor_model = joblib.load('static/tumor_detection_svm_model.pkl')


# Function to extract features from the image for tumor detection
def extract_features(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Example code to extract features (actual logic will vary based on your feature extraction):
    mean = np.mean(img)
    variance = np.var(img)
    std_dev = np.std(img)
    skewness = scipy.stats.skew(img.flatten())
    kurtosis = scipy.stats.kurtosis(img.flatten())

    # Placeholder for second-order features (e.g., contrast, energy, etc.)
    contrast = 0  # Replace with actual calculation
    energy = 0  # Replace with actual calculation
    asm = 0  # Replace with actual calculation
    entropy = 0  # Replace with actual calculation
    homogeneity = 0  # Replace with actual calculation
    dissimilarity = 0  # Replace with actual calculation
    correlation = 0  # Replace with actual calculation
    coarseness = 0  # Replace with actual calculation

    # Create a numpy array of all features
    features = np.array(
        [mean, variance, std_dev, skewness, kurtosis, contrast, energy, asm, entropy, homogeneity, dissimilarity,
         correlation, coarseness])

    return features


# Home page
@app.route('/')
def index():
    return render_template('index.html')

# About Us route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact route
@app.route('/contact')
def contact():
    return render_template('contact.html')


# Diabetes prediction page
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        gender = request.form['gender'].lower()
        if gender == 'male':
            gender = 1
        else:
            gender = 0

        age = int(request.form['age'])
        hypertension = 1 if request.form['hypertension'].lower() == 'yes' else 0
        heart_disease = 1 if request.form['heart_disease'].lower() == 'yes' else 0
        smoking_history = int(request.form['smoking_history'])
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])

        # Create input array for prediction
        input_features = np.array(
            [[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])

        # Make prediction
        prediction = diabetes_model.predict(input_features)
        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
        return render_template('diabetes.html', result=result)

    return render_template('diabetes.html')


# Tumor detection page
@app.route('/tumor', methods=['GET', 'POST'])
def tumor():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded image
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)

        # Extract features from the image
        features = extract_features(file_path)

        # Check if features is a 1D array
        if len(features.shape) == 1:
            # Reshape features to be 2D with shape (1, N) for model input
            features = features.reshape(1, -1)

        # Double check feature shape
        if features.shape[1] != 13:
            return "Error: Feature extraction failed. Incorrect number of features."

        # Make prediction using the SVM model
        prediction = tumor_model.predict(features)
        probability = tumor_model.predict_proba(features)

        if prediction[0] == 1:
            result = f"Tumor detected with {probability[0][1] * 100:.2f}% confidence."
        else:
            result = f"No tumor detected with {probability[0][0] * 100:.2f}% confidence."

        return render_template('tumor.html', result=result, image_path=file.filename)

    return render_template('tumor.html')


if __name__ == '__main__':
    app.run(debug=True)
