from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = "auth.login"

mongo_client = None
def get_db():
    return mongo_client.get_database()

def init_db(app):
    global mongo_client
    mongo_client = MongoClient(app.config["MONGO_URI"])