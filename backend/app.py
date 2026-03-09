from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import random
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

# ---------------- AI MODEL PREDICTION ----------------
def predict_tumor(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]  # [prob_no, prob_yes]
        class_index = np.argmax(prediction)

        if class_index == 1:
            result = "Tumor Detected"
            confidence = round(prediction[1] * 100, 2)
        else:
            result = "No Tumor"
            confidence = round(prediction[0] * 100, 2)

        return result, confidence
    except Exception as e:
        return "Error", 0
    
# Load the model once at startup
model = load_model("brain_tumor_model.h5")

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# Step 1: Define upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')  # Folder where uploaded images will be saved
os.makedirs(UPLOAD_FOLDER, exist_ok=True)   

# Make session permanent for 7 days if remember me checked
app.permanent_session_lifetime = 604800  # 7 days in seconds

DB_NAME = 'database.db'

# Get email credentials from .env
EMAIL_ADDRESS = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS")

# ---------------- DATABASE CONNECTION ----------------
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- CREATE TABLE ----------------
def create_table():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
    
        )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS mri_uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        uploaded_at TEXT,
        predicted_label TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
""")

    conn.commit()
    conn.close()

create_table()

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']

    msg = EmailMessage()
    msg['Subject'] = f"New Contact Form Message from {name}"
    msg['From'] = EMAIL_ADDRESS   # Your admin email
    msg['To'] = EMAIL_ADDRESS     # Admin receives it
    msg.set_content(f"""
    You have received a new message from your website contact form.

    Name: {name}
    Email: {email}
    Message:
    {message}
    """)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        flash("Your message has been sent successfully!", "success")
    except Exception as e:
        flash(f"Failed to send message. Error: {e}", "error")

    return redirect(url_for('contact'))

# ---------------- SIGNUP ----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if len(password) < 6:
            flash("Password must be at least 6 characters long!", "error")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('signup'))

        conn = get_db_connection()
        existing_user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if existing_user:
            conn.close()
            flash("Email already registered!", "error")
            return redirect(url_for('signup'))

        password_hash = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        conn.close()

        flash("Signup successful! You can now log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember = request.form.get('remember')

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']

            if remember:
                session.permanent = True

            flash(f"Welcome, {user['username']}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password!", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()

    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (session['user_id'],)
    ).fetchone()

    uploads = conn.execute(
        "SELECT * FROM mri_uploads WHERE user_id = ?",
        (session['user_id'],)
    ).fetchall()

    conn.close()

    return render_template(
        'dashboard.html',
        user=user,
        uploads=uploads,
        total_uploads=len(uploads)
    )

#------------PREDICT MRI IMAGE----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('mri_image')
        if not file:
            flash("Please upload an image.", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result, confidence = predict_tumor(filepath)

        conn = get_db_connection()
        uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            "INSERT INTO mri_uploads (user_id, filename, predicted_label, uploaded_at) VALUES (?, ?, ?, ?)",
            (session['user_id'], filename, result, uploaded_at)
        )

        conn.commit()
        conn.close()

        return render_template(
            'predict.html',
            result=result,
            filename=filename,
            confidence=confidence
        )

    return render_template('predict.html')

#-----------------UPLOAD HISTORY----------------
@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    uploads = conn.execute(
        "SELECT * FROM mri_uploads WHERE user_id = ? ORDER BY uploaded_at DESC",
        (session['user_id'],)
    ).fetchall()
    conn.close()

    return render_template('history.html', uploads=uploads)

#---------PROFILE-----
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (session['user_id'],)
    ).fetchone()

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if password:  # Only update password if entered
            hashed_password = generate_password_hash(password)
            conn.execute(
                "UPDATE users SET username = ?, email = ?, password = ? WHERE id = ?",
                (username, email, hashed_password, session['user_id'])
            )
        else:
            conn.execute(
                "UPDATE users SET username = ?, email = ? WHERE id = ?",
                (username, email, session['user_id'])
            )

        conn.commit()
        conn.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    conn.close()
    return render_template('profile.html', user=user)

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))

# ---------------- FORGOT PASSWORD ----------------
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if not user:
            flash("Email not registered!", "error")
            return redirect(url_for('forgot_password'))

        otp = random.randint(100000, 999999)
        session['otp'] = str(otp)
        session['reset_email'] = email
        session['otp_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # OTP timestamp

        try:
            msg = EmailMessage()
            msg['Subject'] = 'Brain Tumor Detection - Password Reset OTP'
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            msg.set_content(f'Your OTP for password reset is: {otp}')

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)

            flash("OTP sent to your email. Check inbox.", "success")
            return redirect(url_for('verify_otp'))

        except Exception as e:
            flash(f"Failed to send email: {e}", "error")
            return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')

# ---------------- VERIFY OTP ----------------
@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form['otp']

        if 'otp' in session and 'otp_time' in session:
            otp_time = datetime.strptime(session['otp_time'], "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            if now - otp_time > timedelta(minutes=1):
                session.pop('otp', None)
                session.pop('otp_time', None)
                session.pop('reset_email', None)
                flash("OTP expired. Please request a new one.", "error")
                return redirect(url_for('forgot_password'))

            if session['otp'] == entered_otp:
                flash("OTP verified! Reset your password.", "success")
                return redirect(url_for('reset_password'))
            else:
                flash("Invalid OTP. Try again.", "error")
                return redirect(url_for('verify_otp'))

    return render_template('verify_otp.html')

# ---------------- RESET PASSWORD ----------------
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = session.get('reset_email')

        if not email:
            flash("Session expired. Please request a new OTP.", "error")
            return redirect(url_for('forgot_password'))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('reset_password'))

        if len(password) < 6:
            flash("Password must be at least 6 characters long!", "error")
            return redirect(url_for('reset_password'))

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        # Check if new password is same as old password
        if user and check_password_hash(user['password'], password):
            conn.close()
            flash("New password shouldn't match the old. Please choose a different password.", "error")
            return redirect(url_for('reset_password'))

        password_hash = generate_password_hash(password)
        conn.execute("UPDATE users SET password = ? WHERE email = ?", (password_hash, email))
        conn.commit()
        conn.close()

        # Clear OTP session
        session.pop('otp', None)
        session.pop('otp_time', None)
        session.pop('reset_email', None)

        flash("Password reset successful! You can login now.", "success")
        return redirect(url_for('login'))

    return render_template('reset_password.html')

# ---------------- ADMIN CREDENTIALS ----------------
ADMIN_USERNAME = os.getenv("ADMIN_USER") 
ADMIN_PASSWORD = os.getenv("ADMIN_PASS") 

# ---------------- ADMIN LOGIN ----------------
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if email == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash("Admin logged in successfully!", "success")
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid admin credentials!", "error")
            return redirect(url_for('admin_login'))

    return render_template('admin_login.html')

# ---------------- ADMIN DASHBOARD ----------------
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    conn = get_db_connection()

    users = conn.execute(
        "SELECT id, username, email FROM users"
    ).fetchall()

    uploads = conn.execute(
        """
        SELECT mri_uploads.*, users.username 
        FROM mri_uploads
        JOIN users ON mri_uploads.user_id = users.id
        ORDER BY uploaded_at DESC
        """
    ).fetchall()

    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    total_uploads = conn.execute(
        "SELECT COUNT(*) FROM mri_uploads"
    ).fetchone()[0]

    conn.close()

    return render_template(
        'admin_dashboard.html',
        users=users,
        uploads=uploads,
        total_users=total_users,
        total_uploads=total_uploads
    )

# ---------------- DELETE USER ----------------
@app.route('/admin/delete_user/<int:user_id>')
def delete_user(user_id):
    if not session.get('admin_logged_in'):
        flash("Please log in as admin!", "error")
        return redirect(url_for('admin_login'))

    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    flash("User deleted successfully!", "success")
    return redirect(url_for('admin_dashboard'))

# ---------------- ADMIN LOGOUT ----------------
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash("Admin logged out successfully!", "success")
    return redirect(url_for('admin_login'))

# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)
