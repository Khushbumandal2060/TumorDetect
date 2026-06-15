import asyncio
import os
import random
import smtplib
import threading
import uuid
import requests
from datetime import datetime, timedelta
from email.message import EmailMessage

import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session
from prisma import Prisma
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.security import generate_password_hash, check_password_hash
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

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Email credentials (used by contact form & password reset) ───
EMAIL_ADDRESS = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS")

MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model.h5")
model = load_model(MODEL_PATH)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "templates"),
    static_folder=os.path.join(BASE_DIR, "..", "static")
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.secret_key = os.getenv("SECRET_KEY", "fallback-dev-key")
app.permanent_session_lifetime = timedelta(days=7)

# ─── Upload limits ───
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "16"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ─── Allowed upload extensions ───
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required. Configure it to point at your Postgres database.")

db = Prisma()

# ─── Persistent event loop for async Prisma calls ───
_loop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_loop_thread.start()


def run_db(operation):
    """Run an async Prisma operation on the persistent background event loop."""
    future = asyncio.run_coroutine_threadsafe(operation, _loop)
    return future.result()


def fetch_user_by_email(email):
    return run_db(db.user.find_unique(where={"email": email}))


def fetch_user_by_id(user_id):
    return run_db(db.user.find_unique(where={"id": user_id}))


def fetch_users():
    return run_db(db.user.find_many())


def fetch_uploads_for_user(user_id):
    return run_db(db.mri_upload.find_many(where={"user_id": user_id}))


def fetch_all_uploads():
    return run_db(db.mri_upload.find_many())


def seed_admin_user():
    admin_email = os.getenv("ADMIN_SEED_EMAIL")
    admin_password = os.getenv("ADMIN_SEED_PASSWORD")
    admin_username = os.getenv("ADMIN_SEED_USERNAME", "Admin")

    if not admin_email or not admin_password:
        return

    password_hash = generate_password_hash(admin_password)
    existing_admin = fetch_user_by_email(admin_email)

    if existing_admin:
        run_db(
            db.user.update(
                where={"id": existing_admin.id},
                data={
                    "username": admin_username,
                    "password_hash": password_hash,
                    "role": "admin",
                },
            )
        )
        return

    run_db(
        db.user.create(
            data={
                "username": admin_username,
                "email": admin_email,
                "password_hash": password_hash,
                "role": "admin",
            }
        )
    )


def initialize_database():
    run_db(db.connect())
    seed_admin_user()


try:
    initialize_database()
except Exception as e:
    print(f"[WARNING] Database init failed: {e}")
    print("The app will start but database features won't work until the DB is reachable.")


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return fetch_user_by_id(user_id)


def current_user_is_admin():
    user = current_user()
    return bool(user and getattr(user, "role", "user") == "admin")


def require_admin_access():
    if not current_user_is_admin():
        flash("Please log in as admin!", "error")
        return False
    return True

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

        existing_user = fetch_user_by_email(email)
        if existing_user:
            flash("Email already registered!", "error")
            return redirect(url_for('signup'))

        password_hash = generate_password_hash(password)
        run_db(
            db.user.create(
                data={
                    "username": username,
                    "email": email,
                    "password_hash": password_hash,
                    "role": "user",
                }
            )
        )

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

        user = fetch_user_by_email(email)

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = getattr(user, 'role', 'user')

            if remember:
                session.permanent = True

            if getattr(user, 'role', 'user') == 'admin':
                session['admin_logged_in'] = True
                flash(f"Welcome back, {user.username}!", "success")
                return redirect(url_for('admin_dashboard'))

            flash(f"Welcome, {user.username}!", "success")
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

    user = fetch_user_by_id(session['user_id'])
    uploads = fetch_uploads_for_user(session['user_id'])

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
        if not file or file.filename == '':
            flash("Please upload an image.", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff", "error")
            return redirect(request.url)

        # UUID prefix prevents filename collisions between users
        safe_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        result, confidence = predict_tumor(filepath)

        # ─── UploadThing Integration ───
        uploadthing_token = os.getenv("UPLOADTHING_TOKEN")
        final_file_path = unique_name
        
        if uploadthing_token and uploadthing_token != "your_uploadthing_secret_token":
            try:
                headers = {
                    "x-uploadthing-api-key": uploadthing_token,
                    "x-uploadthing-version": "7"
                }
                with open(filepath, "rb") as f:
                    ut_response = requests.post(
                        "https://uploadthing.com/api/uploadFiles",
                        headers=headers,
                        files={"files": (safe_name, f)}
                    )
                if ut_response.status_code == 200:
                    ut_data = ut_response.json()
                    if ut_data and len(ut_data) > 0 and 'url' in ut_data[0]:
                        final_file_path = ut_data[0]['url']
                        
                        # Optionally delete local file to save space
                        # if os.path.exists(filepath):
                        #     os.remove(filepath)
            except Exception as e:
                print(f"[WARNING] UploadThing upload failed: {e}")

        run_db(
            db.mri_upload.create(
                data={
                    "user_id": session['user_id'],
                    "filename": final_file_path,
                    "predicted_label": result,
                    "confidence": confidence,
                    "uploaded_at": datetime.now(),
                }
            )
        )

        return render_template(
            'predict.html',
            result=result,
            filename=final_file_path,
            confidence=confidence
        )

    return render_template('predict.html')

#-----------------UPLOAD HISTORY----------------
@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))

    uploads = fetch_uploads_for_user(session['user_id'])

    return render_template('history.html', uploads=uploads)

#---------PROFILE-----
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))

    user = fetch_user_by_id(session['user_id'])

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user = fetch_user_by_email(email)
        if existing_user and existing_user.id != session['user_id']:
            flash("Email already registered!", "error")
            return redirect(url_for('profile'))

        update_data = {
            "username": username,
            "email": email,
        }

        if password:  # Only update password if entered
            update_data["password_hash"] = generate_password_hash(password)

        run_db(
            db.user.update(
                where={"id": session['user_id']},
                data=update_data,
            )
        )

        session['username'] = username
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

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

        user = fetch_user_by_email(email)

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

        user = fetch_user_by_email(email)

        # Check if new password is same as old password
        if user and check_password_hash(user.password_hash, password):
            flash("New password shouldn't match the old. Please choose a different password.", "error")
            return redirect(url_for('reset_password'))

        password_hash = generate_password_hash(password)
        run_db(
            db.user.update(
                where={"id": user.id},
                data={"password_hash": password_hash},
            )
        )

        # Clear OTP session
        session.pop('otp', None)
        session.pop('otp_time', None)
        session.pop('reset_email', None)

        flash("Password reset successful! You can login now.", "success")
        return redirect(url_for('login'))

    return render_template('reset_password.html')

# ---------------- ADMIN LOGIN ----------------
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = fetch_user_by_email(email)

        if user and getattr(user, 'role', 'user') == 'admin' and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['admin_logged_in'] = True
            flash("Admin logged in successfully!", "success")
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid admin credentials or insufficient privileges!", "error")
            return redirect(url_for('admin_login'))

    return render_template('admin_login.html')

# ---------------- ADMIN DASHBOARD ----------------
@app.route('/admin/dashboard')
def admin_dashboard():
    if not require_admin_access():
        return redirect(url_for('admin_login'))

    users = fetch_users()
    uploads = fetch_all_uploads()
    user_lookup = {user.id: user.username for user in users}
    users_view = [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": getattr(user, 'role', 'user'),
        }
        for user in users
    ]
    uploads_view = [
        {
            "id": upload.id,
            "username": user_lookup.get(upload.user_id, "Unknown"),
            "filename": upload.filename,
            "predicted_label": upload.predicted_label,
            "confidence": getattr(upload, 'confidence', None),
            "uploaded_at": upload.uploaded_at,
        }
        for upload in sorted(uploads, key=lambda entry: entry.uploaded_at, reverse=True)
    ]

    return render_template(
        'admin_dashboard.html',
        users=users_view,
        uploads=uploads_view,
        total_users=len(users_view),
        total_uploads=len(uploads_view)
    )

# ---------------- DELETE USER ----------------
@app.route('/admin/delete_user/<int:user_id>')
def delete_user(user_id):
    if not require_admin_access():
        return redirect(url_for('admin_login'))

    run_db(db.user.delete(where={"id": user_id}))

    flash("User deleted successfully!", "success")
    return redirect(url_for('admin_dashboard'))

# ---------------- ADMIN LOGOUT ----------------
@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash("Admin logged out successfully!", "success")
    return redirect(url_for('admin_login'))

# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")
