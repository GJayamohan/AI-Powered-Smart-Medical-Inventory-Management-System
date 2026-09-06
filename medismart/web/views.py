"""
Web UI Views Blueprint.
Renders Jinja2 HTML templates for the pharmacy web application.
"""

from __future__ import annotations

from flask import Blueprint, redirect, render_template, url_for
from flask_login import current_user

web_bp = Blueprint("web_bp", __name__)


@web_bp.route("/")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard")


@web_bp.route("/inventory")
def inventory():
    return render_template("inventory.html", active_page="inventory")


@web_bp.route("/expiry")
def expiry():
    return render_template("expiry.html", active_page="expiry")


@web_bp.route("/predict")
def predict():
    return render_template("predict.html", active_page="predict")


@web_bp.route("/pos")
def pos():
    return render_template("pos.html", active_page="pos")


@web_bp.route("/sales")
def sales():
    return render_template("sales.html", active_page="sales")


@web_bp.route("/login")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("web_bp.dashboard"))
    return render_template("login.html", active_page="login")
