import logging
from flask import Flask, request, render_template, redirect, session, Response, flash
from flask_sqlalchemy import SQLAlchemy
from model import db, User
import bcrypt
import joblib
import numpy as np
import pandas as pd
import os
import time
import sklearn
from threading import Thread
import matplotlib
matplotlib.use('Agg')   # Prevent GUI backend issue
import io
import base64
import matplotlib.pyplot as plt
import re
import threading
from datetime import datetime
from flask_socketio import SocketIO

from flow_engine import FlowEngine
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP

# =========================
# Global / misc
# =========================
filename = ""

# =========================
# Config
# =========================
INTERFACE = None  # set e.g. "eth0" / "wlan0" / "Ethernet" if needed
FLOW_TIMEOUT = 5.0
INACTIVE_TIMEOUT = 3.0
EXPIRE_CHECK_INTERVAL = 1.0

# ==========================================================
# ✅ TWO-STAGE MODEL FILES (Put all these inside /model folder)
# ==========================================================
MODEL_DIR = "models"  # keep relative so deployment is easy

STAGE1_PATH = os.path.join(MODEL_DIR, "randomforest_stage1_with_20_features.joblib")             # Stage-1: BENIGN vs ATTACK
STAGE2_PATH = os.path.join(MODEL_DIR, "stage2_catboost.joblib")             # Stage-2: AttackGroup classifier
#SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")                   # scaler for top20 features
TOPFEAT_PATH = os.path.join(MODEL_DIR, "most_imp20_features.joblib")          # list of top 20 feature names
LE2_PATH = os.path.join(MODEL_DIR, "label_encoder_stage2_top20_feature.joblib")        # label encoder for stage-2 outputs

# =========================
# Flask app
# =========================
app = Flask(__name__)

# =========================
# Load model assets (two-stage)
# =========================
# NOTE:
# - Stage-1 expects scaled top_features
# - Stage-2 expects the SAME scaled top_features (X2_train was built from same top features)
stage1_model = joblib.load(STAGE1_PATH)
stage2_model = joblib.load(STAGE2_PATH)
#scaler = joblib.load(SCALER_PATH)
top_features = joblib.load(TOPFEAT_PATH)
le2 = joblib.load(LE2_PATH)

# Use top_features in forms too (manual predict page)
feature_order = top_features

# =========================
# Database config
# =========================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

app.secret_key = 'secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Session config
app.config['SESSION_TYPE'] = 'filesystem'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Optional: set label column constant
LABEL_COL = "label"

with app.app_context():
    db.create_all()

@app.context_processor
def inject_user():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        return dict(user=user)
    return dict(user=None)

from functools import wraps

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'email' not in session:
            flash("Please login first", "warning")
            return redirect('/login')
        return view(*args, **kwargs)
    return wrapped

# =========================
# Home
# =========================
@app.route('/')
def home():
    return render_template('home.html')

# =========================
# Password validator
# =========================
def is_strong_password(password):
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

# =========================
# Register
# =========================
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password')

        if not name or not email or not password or not confirm_password:
            flash("Please fill in all fields", "danger")
            return redirect('/register')

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect('/register')

        if not is_strong_password(password):
            flash(
                "Password must be at least 8 characters long and include "
                "uppercase, lowercase, number, and special character.",
                "danger"
            )
            return redirect('/register')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please login.", "warning")
            return redirect('/register')

        new_user = User(name=name, email=email)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect('/login')

    return render_template('register.html')

# =========================
# Login
# =========================
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            flash("Login successful!", "success")
            return redirect('/dashboard')
        else:
            flash("Invalid email or password", "danger")
            return redirect('/login')

    return render_template('login.html')

# =========================
# Logout
# =========================
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect('/')

# Choose a threshold you tuned (start with 0.20–0.35 for many NIDS datasets)
STAGE1_THRESHOLD = 0.25
from joblib import load
import os

import warnings
warnings.filterwarnings(
    "error",
    message="X does not have valid feature names*",
    category=UserWarning
)



# if saved as dict
if isinstance(top_features, dict):
    top_features = top_features.get("features") or top_features.get("top_features")

top_features = [str(c).strip() for c in top_features]



import pandas as pd
import numpy as np


def prepare_X(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    df = df.copy()

    # normalize incoming column names
    df.columns = df.columns.astype(str).str.strip()

    # add missing top-20 columns as 0
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # select ONLY top-20 in EXACT training order
    X = df.loc[:, feature_list].copy()

    # force numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # handle inf/-inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # keep this ONLY if you trained with this rule:
    X = X.replace(-1, np.nan)

    # fill missing
    X = X.fillna(0)

    return X


def normalize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)  # collapse multiple spaces
    return s


def harmonize_columns(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Standardize spacing in uploaded CSV column names so they match training feature names.
    """
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]

    feature_list_norm = [normalize_name(f) for f in feature_list]

    norm_to_actual_df = {normalize_name(c): c for c in df.columns}
    rename_dict = {}

    for f_exact, f_norm in zip(feature_list, feature_list_norm):
        if f_norm in norm_to_actual_df:
            current = norm_to_actual_df[f_norm]
            if current != f_exact:
                rename_dict[current] = f_exact

    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df
def predict_stage1_binary(df: pd.DataFrame) -> pd.DataFrame:
    df_in = df.copy()
    df_in.columns = df_in.columns.astype(str).str.strip()
    df_in = harmonize_columns(df_in, top_features)

    X = prepare_X(df_in, top_features)

    # hard check
    if X.shape[1] != len(top_features) or list(X.columns) != list(top_features):
        raise RuntimeError("Feature alignment failed vs top_features list.")

    if not hasattr(stage1_model, "predict_proba"):
        raise RuntimeError("stage1_model does not support predict_proba().")

    # ✅ use DataFrame (keeps feature names => no warning)
    p_attack = stage1_model.predict_proba(X)[:, 1]

    y_pred = (p_attack >= STAGE1_THRESHOLD).astype(int)
    stage1_name = np.where(y_pred == 1, "ATTACK", "BENIGN")

    out = df_in.copy()
    out["AttackProb"] = p_attack.astype(float)
    out["Stage1"] = stage1_name
    return out
def predict_two_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage-1: BENIGN vs ATTACK (RF)
    Stage-2: If ATTACK, classify attack type (CatBoost)
    Output columns:
      - AttackProb
      - Stage1
      - FinalLabel
      - AttackTypeProb
    """
    out = predict_stage1_binary(df)

    out["FinalLabel"] = "BENIGN"
    out["AttackTypeProb"] = 0.0

    attack_mask = (out["Stage1"] == "ATTACK")
    if attack_mask.any():
        df_attack = out.loc[attack_mask].copy()

        # Stage-2 features (same top_features order)
        X2 = prepare_X(df_attack, top_features)

        # CatBoost predict_proba
        proba2 = np.asarray(stage2_model.predict_proba(X2), dtype=float)

        argmax_idx = np.argmax(proba2, axis=1)
        cls_prob = proba2[np.arange(len(argmax_idx)), argmax_idx]

        # ✅ map indices -> actual class labels from model
        if hasattr(stage2_model, "classes_"):
            classes2 = np.asarray(stage2_model.classes_)
            pred_class_labels = classes2[argmax_idx]
        else:
            # fallback: assume 0..K-1
            pred_class_labels = argmax_idx

        # ✅ decode to string labels if label encoder exists
        try:
            attack_types = le2.inverse_transform(pred_class_labels.astype(int))
        except Exception:
            attack_types = pred_class_labels.astype(str)

        out.loc[attack_mask, "FinalLabel"] = attack_types
        out.loc[attack_mask, "AttackTypeProb"] = cls_prob.astype(float)

    return out

# ==========================================================
# Upload CSV (ONE-STAGE inference)
# ==========================================================
last_prediction = []
cumulative_predictions = []

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    global last_prediction, cumulative_predictions

    if request.method == 'GET':
        return render_template('upload.html')

    if 'file' not in request.files:
        return render_template('upload.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error='No selected file')

    try:
        df = pd.read_csv(file)

        # normalize + harmonize columns
        df.columns = df.columns.astype(str).str.strip()
        df = harmonize_columns(df, top_features)

        present = [c for c in top_features if c in df.columns]
        missing = [c for c in top_features if c not in df.columns]

        print(f"Top20 present: {len(present)}/20")
        if missing:
            print("Missing top_features (will be filled with 0):", missing)

        # guard
        if len(present) < 15:
            return render_template(
                'upload.html',
                error=f"Uploaded CSV schema mismatch: only {len(present)}/20 required features found."
            )

        pred_df = predict_two_stage(df)

        response = []
        for i, row in pred_df.iterrows():
            final_label = row["FinalLabel"]  # BENIGN or attack-type

            response.append({
                "sr_no": int(i) + 1,
                "class_index": 0 if final_label == "BENIGN" else 1,
                "class_name": final_label,                 # ✅ shows BENIGN OR attack-type
                "attack_prob": float(row["AttackProb"]),   # Stage-1 probability
                 # optional extra (won't break frontend if ignored)
                "stage1": row["Stage1"],
                "attack_type_prob": float(row.get("AttackTypeProb", 0.0))
    })


        last_prediction.clear()
        last_prediction.extend(response)
        cumulative_predictions.extend(response)

        class_counts = pred_df["FinalLabel"].value_counts()


        plt.figure(figsize=(6, 4))
        plt.title("Prediction Distribution (Stage-1 Binary)")
        plt.xlabel("Label")
        plt.ylabel("Count")
        class_counts.plot(kind='bar')
        plt.xticks(rotation=0)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template(
            'upload.html',
            predictions=response,
            plot_url=plot_url,
            attack_counts=class_counts.to_dict()
        )

    except Exception as e:
        return render_template('upload.html', error=f"Error processing file: {str(e)}")
# =========================
# Manual Predict (top20 feature form)
# =========================
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        try:
            user_input = {}
            for col in feature_order:
                v = request.form.get(col)
                if v is None or str(v).strip() == "":
                    v = 0
                user_input[col] = float(v)

            df_one = pd.DataFrame([user_input])

            # ✅ TWO-STAGE inference
            pred_df = predict_two_stage(df_one)

            stage1 = pred_df.loc[0, "Stage1"]          # BENIGN / ATTACK
            final_label = pred_df.loc[0, "FinalLabel"] # BENIGN or attack type

            attack_prob = float(pred_df.loc[0, "AttackProb"])
            attack_type_prob = float(pred_df.loc[0, "AttackTypeProb"]) if "AttackTypeProb" in pred_df.columns else 0.0

            # Result string (frontend unchanged)
            if stage1 == "BENIGN":
                result = f"Stage-1: BENIGN | Final: BENIGN | AttackProb: {attack_prob:.4f}"
            else:
                result = (
                    f"Stage-1: ATTACK | Final: {final_label} | "
                    f"AttackProb: {attack_prob:.4f} | TypeProb: {attack_type_prob:.4f}"
                )

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('predict.html', feature_order=feature_order, result=result)

# @app.route('/dashboard')
# # @login_required
# # def dashboard():
# #     global cumulative_predictions
# #     user = User.query.filter_by(email=session['email']).first()
    
    
# #     if not cumulative_predictions:
# #         return render_template('dashboard.html',
# #                                user=user,
# #                                total_predictions=0,
# #                                attack_count=0,
# #                                normal_count=0,
# #                                accuracy=0,
# #                                attack_counts={},
# #                                plot_url_bar=None,
# #                                plot_url_pie=None,
# #                                plot_url_line=None)

# #     # Convert cumulative predictions into DataFrame
# #     df = pd.DataFrame(cumulative_predictions)

# #     # --- Metrics ---
# #     total_predictions = len(df)
# #     normal_count = (df['class_name'] == 'BENIGN').sum()
# #     attack_count = total_predictions - normal_count

# #     # --- Real Accuracy (if you have true labels, replace here) ---
# #     # For now, we calculate "model certainty" as % of BENIGN predictions
# #     accuracy = (normal_count / total_predictions) * 100 if total_predictions > 0 else 0

# #     # --- Class Distribution ---
# #     class_counts = df['class_name'].value_counts()
# #     # Ensure BENIGN first
# #     if 'BENIGN' in class_counts:
# #         class_counts = class_counts.reindex(['BENIGN'] + [c for c in class_counts.index if c != 'BENIGN'], fill_value=0)

# #     # =========================================
# #     # 1️⃣ Bar Chart — Class Distribution
# #     # =========================================
# #     plt.figure(figsize=(6, 4))
# #     class_counts.plot(kind='bar',
# #                       color=['#34d399' if c == 'BENIGN' else '#f87171' for c in class_counts.index])
# #     # plt.title("Cumulative Prediction Distribution")
# #     plt.xlabel("Class Name")
# #     plt.ylabel("Count")
# #     plt.xticks(rotation=45)
# #     plt.tight_layout()

# #     img_bar = io.BytesIO()
# #     plt.savefig(img_bar, format='png')
# #     img_bar.seek(0)
# #     plot_url_bar = base64.b64encode(img_bar.getvalue()).decode()
# #     plt.close()

# #     # =========================================
# #     # 2️⃣ Pie Chart — Normal vs Attack Ratio
# #     # =========================================
# #     plt.figure(figsize=(5, 5))
# #     labels = ['BENIGN', 'ATTACK']
# #     sizes = [normal_count, attack_count]
# #     colors = ['#34d399', '#f87171']
# #     plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
# #     # plt.title("Normal vs Attack Distribution")
# #     plt.tight_layout()

# #     img_pie = io.BytesIO()
# #     plt.savefig(img_pie, format='png')
# #     img_pie.seek(0)
# #     plot_url_pie = base64.b64encode(img_pie.getvalue()).decode()
# #     plt.close()

# #     # =========================================
# #     # 3️⃣ Line Chart — Cumulative BENIGN vs ATTACK
# #     # =========================================
# #     df['benign_cum'] = (df['class_name'] == 'BENIGN').cumsum()
# #     df['attack_cum'] = (df['class_name'] != 'BENIGN').cumsum()
# #     df['index'] = range(1, len(df) + 1)  # Use index as X-axis (like time)

# #     plt.figure(figsize=(7, 4))
# #     plt.plot(df['index'], df['benign_cum'], label='BENIGN', color='#10b981', linewidth=2)
# #     plt.plot(df['index'], df['attack_cum'], label='ATTACK', color='#ef4444', linewidth=2)
# #     # plt.title("Cumulative BENIGN vs ATTACK Over Time")
# #     plt.xlabel("Prediction Index")
# #     plt.ylabel("Cumulative Count")
# #     plt.legend()
# #     plt.xticks(rotation=30)
# #     plt.tight_layout()

# #     img_line = io.BytesIO()
# #     plt.savefig(img_line, format='png')
# #     img_line.seek(0)
# #     plot_url_line = base64.b64encode(img_line.getvalue()).decode()
# #     plt.close()

# #     # --- Chart.js or HTML use ---
# #     attack_counts = class_counts.to_dict()

# #     return render_template('dashboard.html',
# #                            total_predictions=total_predictions,
# #                            attack_count=attack_count,
# #                            normal_count=normal_count,
# #                            accuracy=round(accuracy, 2),
# #                            attack_counts=attack_counts,
# #                            plot_url_bar=plot_url_bar,
# #                            plot_url_pie=plot_url_pie,
# #                            plot_url_line=plot_url_line)
@app.route('/dashboard')
@login_required
def dashboard():
    global cumulative_predictions
    user = User.query.filter_by(email=session['email']).first()

    if not cumulative_predictions:
        return render_template(
            'dashboard.html',
            user=user,
            total_predictions=0,
            attack_count=0,
            normal_count=0,
            attack_rate=0,
            attack_counts={},
            plot_url_bar=None,
            plot_url_pie=None,
            plot_url_line=None
        )

    df = pd.DataFrame(cumulative_predictions)

    total_predictions = len(df)
    normal_count = (df['class_name'] == 'BENIGN').sum()
    attack_count = total_predictions - normal_count

    # ✅ Only new metric you keep
    attack_rate = (attack_count / total_predictions) * 100 if total_predictions else 0

    # --- Class Distribution (keep for charts) ---
    class_counts = df['class_name'].value_counts()

    if 'BENIGN' in class_counts:
        class_counts = class_counts.reindex(
            ['BENIGN'] + [c for c in class_counts.index if c != 'BENIGN'],
            fill_value=0
        )

    # 1) Bar Chart
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind='bar',
                      color=['#34d399' if c == 'BENIGN' else '#f87171' for c in class_counts.index])
    plt.xlabel("Class Name")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img_bar = io.BytesIO()
    plt.savefig(img_bar, format='png')
    img_bar.seek(0)
    plot_url_bar = base64.b64encode(img_bar.getvalue()).decode()
    plt.close()

    # 2) Pie Chart
    plt.figure(figsize=(5, 5))
    labels = ['BENIGN', 'ATTACK']
    sizes = [normal_count, attack_count]
    colors = ['#34d399', '#f87171']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.tight_layout()

    img_pie = io.BytesIO()
    plt.savefig(img_pie, format='png')
    img_pie.seek(0)
    plot_url_pie = base64.b64encode(img_pie.getvalue()).decode()
    plt.close()

    # 3) Line Chart
    df['benign_cum'] = (df['class_name'] == 'BENIGN').cumsum()
    df['attack_cum'] = (df['class_name'] != 'BENIGN').cumsum()
    df['index'] = range(1, len(df) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(df['index'], df['benign_cum'], label='BENIGN', color='#10b981', linewidth=2)
    plt.plot(df['index'], df['attack_cum'], label='ATTACK', color='#ef4444', linewidth=2)
    plt.xlabel("Prediction Index")
    plt.ylabel("Cumulative Count")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    img_line = io.BytesIO()
    plt.savefig(img_line, format='png')
    img_line.seek(0)
    plot_url_line = base64.b64encode(img_line.getvalue()).decode()
    plt.close()

    return render_template(
        'dashboard.html',
        user=user,
        total_predictions=total_predictions,
        attack_count=attack_count,
        normal_count=normal_count,
        attack_rate=round(attack_rate, 2),
        attack_counts=class_counts.to_dict(),
        plot_url_bar=plot_url_bar,
        plot_url_pie=plot_url_pie,
        plot_url_line=plot_url_line
    )

# =========================
# Download last upload report
# =========================
@app.route('/download_report')
@login_required
def download_report():
    global last_prediction
    if not last_prediction:
        return Response("No predictions available to download.", mimetype='text/plain')

    report_lines = ["Sr No,Stage1,FinalLabel\n"]
    for item in last_prediction:
        line = f"{item['sr_no']},{item.get('stage1','')},{item.get('class_name','')}\n"
        report_lines.append(line)

    report_content = ''.join(report_lines)
    return Response(
        report_content,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=network_intrusion_report.csv'}
    )

@app.route('/about')
@login_required
def about():
    return render_template('aboutus.html')

# =========================
# Profile (keep as-is)
# =========================
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    global cumulative_predictions
    if 'email' not in session:
        flash("Please login to view profile", "error")
        return redirect('/login')

    user = User.query.filter_by(email=session['email']).first()
    if not user:
        flash("User not found", "error")
        return redirect('/login')

    if request.method == 'POST':
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')

        if not old_password or not new_password:
            flash("Please fill all password fields", "error")
            return redirect('/profile')

        if not user.check_password(old_password):
            flash("Old password is incorrect", "error")
            return redirect('/profile')

        user.set_password(new_password)
        db.session.commit()

        flash("Password updated successfully", "success")
        return redirect('/profile')

    total_uploads = len(cumulative_predictions)
    total_predictions = len(cumulative_predictions)
    normal_count = sum(1 for p in cumulative_predictions if p.get('class_name') == 'BENIGN')
    total_attacks = total_predictions - normal_count

    return render_template(
        'profile.html',
        user=user,
        total_uploads=total_uploads,
        total_predictions=total_predictions,
        total_attacks=total_attacks
    )


# -------------------
# Flow engine
# -------------------
engine = FlowEngine(flow_timeout_sec=FLOW_TIMEOUT, inactive_timeout_sec=INACTIVE_TIMEOUT)


# Keep small history for initial load
RECENT_FLOW_LIMIT = 150
RECENT_PKT_LIMIT = 200

recent_flows = []
recent_pkts = []

def push_flow(evt):
    recent_flows.append(evt)
    if len(recent_flows) > RECENT_FLOW_LIMIT:
        del recent_flows[0:len(recent_flows)-RECENT_FLOW_LIMIT]

def push_pkt(evt):
    recent_pkts.append(evt)
    if len(recent_pkts) > RECENT_PKT_LIMIT:
        del recent_pkts[0:len(recent_pkts)-RECENT_PKT_LIMIT]
#STAGE1_THRESHOLD = 0.25  # tune if needed
def predict_flow(features: dict):
    """
    Returns:
      stage1_label      -> "BENIGN" or "ATTACK"
      attack_prob       -> Stage-1 P(ATTACK)
      final_label       -> "BENIGN" or attack type
      final_conf        -> confidence of final decision
      attack_type_prob  -> Stage-2 max prob (0 if BENIGN)
    """

    # 1) Build DF with correct training columns (names + order)
    X = pd.DataFrame([features], columns=top_features)

    # 2) clean numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # -------------------------
    # Stage-1 (RandomForest)
    # -------------------------
    p1 = stage1_model.predict_proba(X)[0]     # ✅ keep DF, not numpy
    attack_prob = float(p1[1])
    stage1_label = "ATTACK" if attack_prob >= STAGE1_THRESHOLD else "BENIGN"

    # If BENIGN, stop here
    if stage1_label == "BENIGN":
        final_label = "BENIGN"
        final_conf = float(np.max(p1))
        attack_type_prob = 0.0
        return stage1_label, attack_prob, final_label, final_conf, attack_type_prob

    # -------------------------
    # Stage-2 (CatBoost)
    # -------------------------
    p2 = np.asarray(stage2_model.predict_proba(X), dtype=float)[0]
    idx = int(np.argmax(p2))
    attack_type_prob = float(p2[idx])

    # decode attack type
    # If your CatBoost was trained on integer-encoded labels, le2 expects that int.
    if hasattr(stage2_model, "classes_"):
        raw_class = stage2_model.classes_[idx]      # could be int or str
        try:
            final_label = le2.inverse_transform([int(raw_class)])[0]
        except Exception:
            final_label = str(raw_class)
    else:
        # fallback: assume idx maps to encoder (sometimes true)
        final_label = le2.inverse_transform([idx])[0]

    final_conf = attack_type_prob
    return stage1_label, attack_prob, final_label, final_conf, attack_type_prob


def packet_sniffer_thread():
    def on_packet(pkt):
        # update flows
        engine.process_packet(pkt)

        # emit RAW packet stream (Wireshark-like)
        if not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        proto = "IP"
        sport = None
        dport = None
        flags = ""

        if pkt.haslayer(TCP):
            proto = "TCP"
            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)
            flags = str(pkt[TCP].flags)
        elif pkt.haslayer(UDP):
            proto = "UDP"
            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)

        evt = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "src": ip.src,
            "dst": ip.dst,
            "proto": proto,
            "sport": sport,
            "dport": dport,
            "len": int(getattr(ip, "len", len(pkt))),
            "flags": flags
        }

        push_pkt(evt)
        socketio.emit("packet_event", evt)

    sniff(iface=INTERFACE, prn=on_packet, store=False)

# -------------------
# Alerting (backend)
# -------------------
# Emits a separate Socket.IO event: "alert_event"
# so the frontend can show a banner/toast, play sound, etc.
#

# Heuristic-based PortScan detection parameters
# from collections import defaultdict, deque

# SCAN_WINDOW_SEC = 5.0
# SCAN_PORT_THRESHOLD = 25   # Nmap hits many ports quickly

# _recent_ports = defaultdict(deque)
# this is for the rule based portscan detection
# def portscan_heuristic(evt: dict) -> bool:
#     """
#     Returns True if a port scan pattern is detected.
#     """
#     now = time.time()

#     if evt.get("proto") != "TCP":
#         return False

#     src = evt.get("src")
#     dst = evt.get("dst")
#     dport = evt.get("dport")

#     if not src or not dst or not dport:
#         return False

#     key = (src, dst)
#     dq = _recent_ports[key]

#     dq.append((now, int(dport)))

#     # remove old entries
#     while dq and (now - dq[0][0]) > SCAN_WINDOW_SEC:
#         dq.popleft()

#     distinct_ports = len({p for _, p in dq})
#     return distinct_ports >= SCAN_PORT_THRESHOLD

ALERT_COOLDOWN_SEC = 1.5  # rate-limit alerts per (src,dst,label) to avoid spam
_last_alert_time = {}     # dict key -> last_time (epoch seconds)

# Optional: write alerts to console (or file if you configure logging handlers)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ANIDS_ALERTS")


def should_emit_alert(evt: dict) -> bool:
    """
    Returns True if this event should generate an alert.
    You can customize this logic later (e.g., confidence threshold, label whitelist).
    """
    return evt.get("label") and evt["label"] != "BENIGN"


def emit_alert(evt: dict):
    """
    Emit alert_event to all clients with a minimal payload.
    Also rate-limits to avoid alert spam.
    """
    now = time.time()

    key = (evt.get("src"), evt.get("dst"), evt.get("label"))
    last = _last_alert_time.get(key, 0)

    # Rate limit
    if (now - last) < ALERT_COOLDOWN_SEC:
        return

    _last_alert_time[key] = now

    alert_payload = {
        "ts": evt.get("ts"),
        "label": evt.get("label"),
        "confidence": evt.get("confidence"),
        "src": evt.get("src"),
        "dst": evt.get("dst"),
        "sport": evt.get("sport"),
        "dport": evt.get("dport"),
        "proto": evt.get("proto"),
        # Optional: include is_attack for convenience
        "is_attack": True,
    }

    # Log for future reference
    logger.warning(
        "ALERT %s %s:%s -> %s:%s proto=%s conf=%s",
        alert_payload["label"],
        alert_payload["src"], alert_payload["sport"],
        alert_payload["dst"], alert_payload["dport"],
        alert_payload["proto"],
        alert_payload["confidence"]
    )

    # Emit to frontend
    socketio.emit("alert_event", alert_payload)
def flow_expirer_thread():
    while True:
        expired = engine.expire_flows()

        for fs in expired:
            features = engine.extract_top20_features(fs)

            # ✅ two-stage prediction
            stage1_label, attack_prob, final_label, final_conf, attack_type_prob = predict_flow(features)

            # ✅ Bidirectional key unpack
            ip_a, port_a, ip_b, port_b, proto = fs.flow_key

            # ✅ Forward direction based on first packet seen
            src_ip = fs.fwd_src_ip
            sport = int(fs.fwd_src_port)
            dport = int(fs.dst_port)
            dst_ip = ip_b if src_ip == ip_a else ip_a

            is_attack = (final_label != "BENIGN")

            evt = {
                "ts": datetime.now().strftime("%H:%M:%S"),
                "src": src_ip,
                "dst": dst_ip,
                "sport": sport,
                "dport": dport,
                "proto": proto,

                # ✅ final classification (BENIGN or attack type)
                "label": final_label,
                "is_attack": is_attack,

                # ✅ expose both-stage info for UI/debug
                "stage1": stage1_label,                      # BENIGN / ATTACK
                "attack_prob": round(float(attack_prob), 4), # stage-1 prob
                "attack_type_prob": round(float(attack_type_prob), 4),  # stage-2 prob (0 if benign)

                # ✅ one confidence field kept for alerts/UI
                "confidence": round(float(final_conf), 4),

                "features": {k: round(float(v), 6) for k, v in features.items()},
            }

            push_flow(evt)
            socketio.emit("flow_event", evt)

            if should_emit_alert(evt):
                emit_alert(evt)

        socketio.sleep(EXPIRE_CHECK_INTERVAL)

@app.route("/realtime_dashboard")
@login_required
def realtime_dashboard():
    return render_template(
        "realtime_dashboard.html",
        feature_order=feature_order,
        init_flows=list(reversed(recent_flows)),
        init_pkts=list(reversed(recent_pkts)),
    )

@socketio.on("connect")
def on_connect():
    # send history so page loads filled
    socketio.emit("history", {"flows": recent_flows, "pkts": recent_pkts})

def start_threads():
    threading.Thread(target=packet_sniffer_thread, daemon=True).start()
    threading.Thread(target=flow_expirer_thread, daemon=True).start()

if __name__ == "__main__":
    start_threads()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
if __name__ == "__main__":
    # start_threads()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
