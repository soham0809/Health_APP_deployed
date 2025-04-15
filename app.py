from flask import Flask, request, render_template, redirect, url_for, session, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import os
import cv2
import scipy.stats
import csv
import io
from datetime import datetime
import pytz

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

DB_USER = "medicareadmin"
DB_PASSWORD = "7w6A8}hS<(*N"
DB_HOST = "restored-db.cj4qkmm6ap1p.ap-south-1.rds.amazonaws.com"
DB_NAME = "medicareDB"
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?sslmode=require"
    
# # With a direct connection string:
# app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://medicareadmin:7w6A8}hS<(*N@restored-db.cj4qkmm6ap1p.ap-south-1.rds.amazonaws.com/medicareDB?sslmode=require"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    diabetes_result = db.Column(db.Boolean, default=None)
    tumor_result = db.Column(db.Boolean, default=None)
    heart_result = db.Column(db.Boolean, default=None)
    
    def is_admin(self):
        return self.email == 'admin@gmail.com' and self.name == 'admin'

# Load ML models
diabetes_model = joblib.load('static/diabetes_rf_model.pkl')
tumor_model = joblib.load('static/tumor_detection_svm_model.pkl')
heart_attack_model = joblib.load('static/heart_model.pkl')

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean = np.mean(img)
    variance = np.var(img)
    std_dev = np.std(img)
    skewness = scipy.stats.skew(img.flatten())
    kurtosis = scipy.stats.kurtosis(img.flatten())
    contrast = 0
    energy = 0
    asm = 0
    entropy = 0
    homogeneity = 0
    dissimilarity = 0
    correlation = 0
    coarseness = 0
    features = np.array([mean, variance, std_dev, skewness, kurtosis, contrast, energy, asm, entropy, homogeneity, dissimilarity, correlation, coarseness])
    return features

@app.route('/')
def index():
    is_admin = False
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            is_admin = user.is_admin()
    return render_template('index.html', logged_in='user_id' in session, username=session.get('name', ''), is_admin=is_admin)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Special case for admin login
        if email == 'admin' and password == 'admin':
            # Check if admin exists, if not create it
            admin = User.query.filter_by(email='admin').first()
            if not admin:
                admin = User(name='admin', email='admin', password=generate_password_hash('admin'))
                db.session.add(admin)
                db.session.commit()
            
            session['user_id'] = admin.id
            session['name'] = admin.name
            india = pytz.timezone('Asia/Kolkata')
            admin.last_login = datetime.now(india)
            db.session.commit()
            flash('Admin login successful!', 'success')
            return redirect(url_for('index'))
            
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['name'] = user.name
            user.last_login = datetime.utcnow()
            db.session.commit()
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'error')
        elif password != confirm_password:
            flash('Passwords do not match.', 'error')
        else:
            hashed_pw = generate_password_hash(password)
            new_user = User(name=name, email=email, password=hashed_pw)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return "About Us Page"

@app.route('/contact')
def contact():
    return "Contact Page"

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            gender = 1 if request.form['gender'].lower() == 'male' else 0
            age = int(request.form['age'])
            hypertension = 1 if request.form['hypertension'].lower() == 'yes' else 0
            heart_disease = 1 if request.form['heart_disease'].lower() == 'yes' else 0
            smoking_history = int(request.form['smoking_history'])
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])
            input_features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
            prediction = diabetes_model.predict(input_features)
            result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"

            if 'user_id' in session:
                user = User.query.get(session['user_id'])
                user.diabetes_result = bool(prediction[0])
                db.session.commit()

            return render_template('diabetes.html', result=result)
        except Exception as e:
            return render_template('diabetes.html', error=f"Error: {str(e)}")

    return render_template('diabetes.html')

@app.route('/tumor', methods=['GET', 'POST'])
def tumor():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        features = extract_features(file_path).reshape(1, -1)
        prediction = tumor_model.predict(features)
        probability = tumor_model.predict_proba(features)

        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            user.tumor_result = bool(prediction[0])
            db.session.commit()

        if prediction[0] == 1:
            result = f"Tumor detected with {probability[0][1] * 100:.2f}% confidence."
        else:
            result = f"No tumor detected with {probability[0][0] * 100:.2f}% confidence."
        return render_template('tumor.html', result=result, image_path=file.filename)
    return render_template('tumor.html')

@app.route('/heart_attack', methods=['GET', 'POST'])
def heart_attack():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = 1 if request.form['sex'].lower() == 'male' else 0
        cholesterol = float(request.form['cholesterol'])
        blood_pressure = request.form['blood_pressure'].split('/')
        systolic_bp = float(blood_pressure[0])
        diastolic_bp = float(blood_pressure[1])
        heart_rate = float(request.form['heart_rate'])
        diabetes = int(request.form['diabetes'])
        family_history = int(request.form['family_history'])
        smoking = int(request.form['smoking'])
        obesity = int(request.form['obesity'])
        alcohol = int(request.form['alcohol'])
        exercise_hours = float(request.form['exercise_hours'])
        diet = int(request.form['diet'])
        previous_heart_problems = int(request.form['previous_heart_problems'])
        medication_use = int(request.form['medication_use'])
        stress_level = float(request.form['stress_level'])
        sedentary_hours = float(request.form['sedentary_hours'])
        income = float(request.form['income'])
        bmi = float(request.form['bmi'])
        triglycerides = float(request.form['triglycerides'])
        physical_activity_days = int(request.form['physical_activity_days'])
        sleep_hours = float(request.form['sleep_hours'])

        input_features = np.array([[age, sex, cholesterol, systolic_bp, diastolic_bp, heart_rate, diabetes,
                                    family_history, smoking, obesity, alcohol, exercise_hours, diet,
                                    previous_heart_problems, medication_use, stress_level, sedentary_hours,
                                    income, bmi, triglycerides, physical_activity_days, sleep_hours]])

        prediction = heart_attack_model.predict(input_features)
        result = "Heart Attack Risk Detected" if prediction[0] == 1 else "No Heart Attack Risk Detected"

        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            user.heart_result = bool(prediction[0])
            db.session.commit()

        return render_template('heart_attack.html', result=result)

    return render_template('heart_attack.html')

@app.route('/records')
def records():
    # Check if user is logged in and is admin
    if 'user_id' not in session:
        flash('You must be logged in to access this page.', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin():
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('index'))
    
    # Get all users and their test results
    patients = User.query.filter(User.email != 'admin').all()
    return render_template('records.html', patients=patients, is_admin=True)

@app.route('/download_csv')
def download_csv():
    # Check if user is logged in and is admin
    if 'user_id' not in session:
        flash('You must be logged in to access this page.', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin():
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('index'))
    
    # Get all patients (excluding admin)
    patients = User.query.filter(User.email != 'admin').all()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Name', 'Email', 'Last Login', 'Diabetes Result', 'Tumor Result', 'Heart Attack Risk'])
    
    # Write data
    for patient in patients:
        diabetes_result = "Positive" if patient.diabetes_result else "Negative" if patient.diabetes_result is not None else "Not Tested"
        tumor_result = "Positive" if patient.tumor_result else "Negative" if patient.tumor_result is not None else "Not Tested"
        heart_result = "Positive" if patient.heart_result else "Negative" if patient.heart_result is not None else "Not Tested"
        
        writer.writerow([
            patient.id,
            patient.name,
            patient.email,
            patient.last_login.strftime('%Y-%m-%d %H:%M:%S') if patient.last_login else "Never",
            diabetes_result,
            tumor_result,
            heart_result
        ])
    
    # Prepare response
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=patient_records.csv"}
    )

@app.route('/privacy')
def privacy():
    return "Privacy Policy Page"

@app.route('/terms')
def terms():
    return "Terms of Service Page"

@app.route('/forgot-password')
def forgot_password():
    return "Password Reset Page"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)