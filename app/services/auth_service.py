from app.extensions import get_db, bcrypt
from datetime import datetime

def register_user(username, email, password):
    db = get_db()
    # check existing user
    if db.users.find_one({"email": email}):
        return False, "Email already exists"

    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")

    user = {
        "username": username,
        "email": email,
        "password": hashed_pw,
        "created_at": datetime.utcnow()
    }

    db.users.insert_one(user)
    return True, "User registered successfully"


def login_user(email, password):
    db = get_db()
    user = db.users.find_one({"email": email})

    if not user:
        return None, "User not found"

    if not bcrypt.check_password_hash(user["password"], password):
        return None, "Incorrect password"

    return user, "Login successful"