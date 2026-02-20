from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import smtplib
import random
from email.message import EmailMessage

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages and sessions

DB_NAME = 'database.db'

# ---------------- DATABASE CONNECTION ----------------
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- CREATE TABLE ----------------
def create_table():
    if not os.path.exists(DB_NAME):
        conn = get_db_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
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
    flash("Thank you! We received your message.", "success")
    return redirect(url_for('contact'))

# ---------------- SIGNUP ----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Password length check
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

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

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

        # Check if user exists
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if not user:
            flash("Email not registered!", "error")
            return redirect(url_for('forgot_password'))

        # Generate OTP
        otp = random.randint(100000, 999999)
        session['otp'] = otp
        session['reset_email'] = email

        # Send OTP via Gmail
        try:
            msg = EmailMessage()
            msg['Subject'] = 'Brain Tumor Detection - Password Reset OTP'
            msg['From'] = 'your_email@gmail.com'       # Replace with your Gmail
            msg['To'] = email
            msg.set_content(f'Your OTP for password reset is: {otp}')

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login('your_email@gmail.com', 'your_app_password')  # Use App Password
                smtp.send_message(msg)

            flash("OTP sent to your email. Check your inbox.", "success")
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
        if 'otp' in session and str(session['otp']) == entered_otp:
            flash("OTP verified! You can reset your password now.", "success")
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

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('reset_password'))

        if len(password) < 6:
            flash("Password must be at least 6 characters long!", "error")
            return redirect(url_for('reset_password'))

        # Update password in DB
        password_hash = generate_password_hash(password)
        email = session.get('reset_email')
        if email:
            conn = get_db_connection()
            conn.execute("UPDATE users SET password = ? WHERE email = ?", (password_hash, email))
            conn.commit()
            conn.close()
            session.pop('otp', None)
            session.pop('reset_email', None)
            flash("Password reset successful! You can now login.", "success")
            return redirect(url_for('login'))

    return render_template('reset_password.html')

# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)