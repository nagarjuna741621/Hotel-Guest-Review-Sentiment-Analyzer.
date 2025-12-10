"""
Authentication helpers
"""
from app.models import User
from app import db

def create_user(username, email, password):
    """Create a new user"""
    if User.query.filter_by(username=username).first():
        return None, "Username already exists"
    if User.query.filter_by(email=email).first():
        return None, "Email already registered"
    
    user = User(username=username, email=email)
    user.set_password(password)
    
    try:
        db.session.add(user)
        db.session.commit()
        return user, None
    except Exception as e:
        db.session.rollback()
        return None, str(e)

def get_user_by_username(username):
    """Get user by username"""
    return User.query.filter_by(username=username).first()

def get_user_by_email(email):
    """Get user by email"""
    return User.query.filter_by(email=email).first()

