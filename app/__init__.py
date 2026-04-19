from flask import Flask, app
from .config import Config
from .extensions import bcrypt, login_manager, init_db
from .models.user_model import User
from .extensions import get_db, login_manager
from bson.objectid import ObjectId
from .routes.auth_routes import auth_bp
from .routes.prediction_routes import prediction_bp

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    return User.get(user_id, db)

def create_app():
    app = Flask(__name__)
    
    # Load config
    app.config.from_object(Config)

    # Init extensions
    bcrypt.init_app(app)
    login_manager.init_app(app)
    init_db(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)

    return app