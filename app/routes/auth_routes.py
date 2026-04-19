from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user
from app.services.auth_service import register_user, login_user as auth_login
from app.models.user_model import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        success, message = register_user(username, email, password)

        if success:
            flash("Signup successful! Please login.")
            return redirect(url_for("auth.login"))
        else:
            flash(message)

    return render_template("signup.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user_data, message = auth_login(email, password)

        if user_data:
            user = User(user_data)
            login_user(user)
            return redirect(url_for("prediction.home"))
        else:
            flash(message)

    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("auth.login"))