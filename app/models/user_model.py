from flask_login import UserMixin
from bson.objectid import ObjectId

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.username = user_data["username"]
        self.email = user_data["email"]

    @staticmethod
    def get(user_id, db):
        user = db.users.find_one({"_id": ObjectId(user_id)})
        return User(user) if user else None