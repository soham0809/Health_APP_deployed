# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
# Make sure your dataset is in a CSV file
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Preprocessing
# Encode categorical features (if necessary)
label_encoder = LabelEncoder()

data['gender'] = label_encoder.fit_transform(data['gender'])  # Convert gender to numeric
data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'])  # Convert smoking history to numeric

# Define features and target
X = data.drop(columns=['diabetes'])  # All columns except the target
y = data['diabetes']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(rf_model, 'static/diabetes_rf_model.pkl')

print("Model trained and saved successfully!")
