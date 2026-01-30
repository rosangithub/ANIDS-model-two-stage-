# 🚨 ANIDS  
## Anomaly-Based Network Intrusion Detection System Using Ensemble Machine Learning

---

## 📌 Overview

**ANIDS** is a **Network Intrusion Detection System (NIDS)** designed to detect abnormal and malicious network traffic using **Machine Learning and Ensemble Learning techniques**.

The system analyzes **network flow-level features** extracted from packet captures and classifies traffic as **BENIGN or various attack types**.  
To improve detection accuracy and reduce false positives, a **Stacking Ensemble Model** is used, combining multiple base classifiers.

The project also includes a **real-time intrusion detection dashboard** built with **Flask and Socket.IO**, capable of live packet capture using **Scapy**.

---

## 🎯 Objectives

- Detect anomalous and malicious network behavior
- Improve detection accuracy using ensemble learning
- Support both **offline (CSV-based)** and **real-time traffic analysis**
- Provide a real-time dashboard with alerts and visualization

---

## 🛠️ Technologies Used

### Programming & Frameworks
- Python 3.x
- Flask
- Flask-SocketIO

### Machine Learning & Data Science
- scikit-learn
- CatBoost
- pandas
- numpy
- matplotlib
- joblib

### Networking
- Scapy (for real-time packet capture)

### Development Tools
- Jupyter Notebook / Google Colab
- Git & GitHub

---

## 🤖 Machine Learning Models Implemented

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- CatBoost Classifier  
- Random Forest Classifier  

### 🔥 Stacking Ensemble Model

- Multiple base models generate predictions
- A **Random Forest Classifier** is used as the **meta-model**
- Final prediction is made using combined model outputs

This approach improves **generalization** and **reduces false positives**.

---

## 📊 Dataset Description

The dataset consists of **network flow features**, including:

- Flow Duration  
- Total Length of Forward Packets  
- Backward Packets per Second  
- Packet Length Standard Deviation  
- Forward Packet Length Mean  
- Destination Port  
- Flow Packets per Second  

### Data Preprocessing Steps

- Handling missing values  
- Feature selection  
- Normalization  
- Train-test splitting  

---

## 📈 Model Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix (per class)

---

## ⚡ Real-Time Intrusion Detection

### Features

- Live packet capture using **Scapy**
- Flow aggregation and feature extraction
- Real-time predictions using trained ML model
- Interactive dashboard:
  - Live packet stream (Wireshark-like)
  - Flow-level predictions
  - Real-time charts
  - Feature inspection
  - Attack alerts (visual + sound)

---

## 🖥️ Project Structure

ANIDS/
│
├── app.py # Flask application entry point
├── flow_engine.py # Flow aggregation & feature extraction
├── model.py # User model (SQLAlchemy)
├── model/ # Trained ML models & encoders
├── notebook/ # Jupyter notebooks
├── static/
│ ├── css/ # Stylesheets
│ └── js/ # JavaScript files
├── templates/ # HTML templates
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore

yaml
Copy code

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rosangithub/ANIDS.git
cd ANIDS
2️⃣ Create and Activate Virtual Environment
bash
Copy code
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Application
bash
Copy code
python app.py
5️⃣ Open in Browser
cpp
Copy code
http://127.0.0.1:5000
⚠️ Note:
Real-time packet capture may require administrator/root privileges.

🔔 Alerting System
Alerts are triggered when an attack flow is detected

Toast notifications in the dashboard

Optional sound alert

Real-time row highlighting

Alerts are emitted from backend using Socket.IO

🔮 Future Enhancements
Cloud deployment (AWS / Azure / GCP)

Deep learning models (LSTM, CNN)

Email / SMS alert integration

Improved scalability for high-speed networks

Role-based access control

👨‍💻 Author
Rosan
GitHub: https://github.com/rosangithub

