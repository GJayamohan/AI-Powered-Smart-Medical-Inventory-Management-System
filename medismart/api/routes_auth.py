"""
Authentication and user management blueprint.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required, login_user, logout_user

from medismart.db import db
from medismart.db.models import User

auth_bp = Blueprint("auth_bp", __name__, url_prefix="/api/auth")


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    identifier = data.get("username") or data.get("email")
    password = data.get("password")

    if not identifier or not password:
        return jsonify({"success": False, "error": "Username/email and password are required."}), 400

    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()

    if not user or not user.check_password(password):
        return jsonify({"success": False, "error": "Invalid username or password."}), 401

    if not user.is_active:
        return jsonify({"success": False, "error": "Account is inactive."}), 403

    login_user(user, remember=data.get("remember", False))
    return jsonify({
        "success": True,
        "message": f"Welcome back, {user.full_name or user.username}!",
        "user": user.to_dict(),
    })


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"success": True, "message": "Logged out successfully."})


@auth_bp.route("/me", methods=["GET"])
def get_current_user():
    if current_user.is_authenticated:
        return jsonify({"authenticated": True, "user": current_user.to_dict()})
    return jsonify({"authenticated": False, "user": None})


@auth_bp.route("/users", methods=["GET"])
@login_required
def list_users():
    if current_user.role != "admin":
        return jsonify({"success": False, "error": "Admin privileges required."}), 403

    users = User.query.order_by(User.username.asc()).all()
    return jsonify({"success": True, "users": [u.to_dict() for u in users]})
