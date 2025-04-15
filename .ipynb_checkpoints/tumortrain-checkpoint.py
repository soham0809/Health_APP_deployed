from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Load your dataset
data = pd.read_csv('braintumor.csv')

# Drop non-numeric columns such as 'Image' if it exists
X = data.drop(['Class', 'Image'], axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model with probability estimates
svm_model = svm.SVC(probability=True)
svm_model.fit(X_train, y_train)

# Save the model
joblib.dump(svm_model, 'static/tumor_detection_svm_model.pkl')

# Test the model
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Placeholder function to detect tumor region (this would usually involve a deep learning model)
def detect_tumor_region(image_path):
    # Load the MRI image in grayscale
    img = cv2.imread(image_path, 0)

    # Perform thresholding to highlight the tumor region (this is just a placeholder method)
    _, threshold = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to color for drawing
    highlighted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if contours:
        # Find the largest contour (assuming it's the tumor)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the moments to calculate the center of the contour
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])  # X coordinate of the tumor center
            cY = int(M["m01"] / M["m00"])  # Y coordinate of the tumor center
        else:
            cX, cY = 0, 0  # In case the contour area is zero
        
        # Draw the contours
        cv2.drawContours(highlighted_img, [largest_contour], -1, (0, 255, 0), 2)
        
        # Draw an arrow pointing to the tumor center
        img_height, img_width = highlighted_img.shape[:2]
        start_point = (img_width // 2, img_height - 20)  # Start of the arrow at the bottom center
        end_point = (cX, cY)  # Pointing to the tumor center
        cv2.arrowedLine(highlighted_img, start_point, end_point, (0, 0, 255), 3, tipLength=0.05)

        # Draw a small circle at the tumor center
        cv2.circle(highlighted_img, (cX, cY), 10, (255, 0, 0), -1)  # Blue circle on the tumor
    
    # Save the result
    result_image_path = os.path.join('static', 'uploads', 'highlighted_tumor_with_arrow.png')
    cv2.imwrite(result_image_path, highlighted_img)

    return result_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tumor', methods=['POST'])
def detect_tumor():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file:
        image_path = os.path.join('static', 'uploads', file.filename)
        file.save(image_path)
        
        # Detect the tumor region in the image
        highlighted_image_path = detect_tumor_region(image_path)

        # For simplicity, assume we run the SVM model here on structured data
        # In a real-world case, we would integrate this with image data preprocessing
        # Using pre-trained deep learning model for tumor detection

        return render_template('index.html', result='Tumor detected!', image_path='highlighted_tumor.png')

if __name__ == "__main__":
    app.run(debug=True)
