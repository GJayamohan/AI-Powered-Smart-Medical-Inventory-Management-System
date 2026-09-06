"""
Flask Application Factory.
"""

from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS
from flask_login import LoginManager

from medismart.db import db, get_default_db_uri
from medismart.db.models import User
from medismart.api.routes_auth import auth_bp
from medismart.api.routes_inventory import inventory_bp
from medismart.api.routes_predict import predict_bp
from medismart.api.routes_expiry import expiry_bp
from medismart.api.routes_dashboard import dashboard_bp


def create_app(test_config: dict | None = None) -> Flask:
    root_dir = Path(__file__).resolve().parents[1]
    template_dir = root_dir / "medismart" / "web" / "templates"
    static_dir = root_dir / "medismart" / "web" / "static"

    app = Flask(
        "medismart",
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )

    # Configuration
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "medismart-dev-secret-key-change-in-prod")
    app.config["SQLALCHEMY_DATABASE_URI"] = get_default_db_uri()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JSON_SORT_KEYS"] = False

    if test_config:
        app.config.update(test_config)

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize Database
    db.init_app(app)

    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth_bp.login"

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(inventory_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(expiry_bp)
    app.register_blueprint(dashboard_bp)

    # Global Health Route
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "MediSmart API",
            "database": "connected",
            "version": "1.0.0",
        })

    # Error Handlers
    @app.errorhandler(404)
    def handle_404(e):
        return jsonify({"success": False, "error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def handle_500(e):
        return jsonify({"success": False, "error": "Internal server error"}), 500

    return app
