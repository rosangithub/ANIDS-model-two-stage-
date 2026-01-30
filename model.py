# models.py
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()  # Initialize SQLAlchemy here

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(50))
    last_login = db.Column(db.DateTime)

    def check_password(self, pwd):
        return check_password_hash(self.password, pwd)

    def set_password(self, new_pwd):
        self.password = generate_password_hash(new_pwd)
        db.session.commit()

# class Upload(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#     filename = db.Column(db.String(200))
#     timestamp = db.Column(db.DateTime)

# class Prediction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#     upload_id = db.Column(db.Integer, db.ForeignKey('upload.id'))
#     result = db.Column(db.String(50))  # BENIGN or ATTACK
#     timestamp = db.Column(db.DateTime)
