"""
Database initialization and SQLAlchemy instance configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def get_default_db_uri() -> str:
    """Return SQLite URI or MySQL URI from DATABASE_URL environment variable."""
    env_uri = os.getenv("DATABASE_URL")
    if env_uri:
        return env_uri
    
    # Default to SQLite in data/ directory
    root_dir = Path(__file__).resolve().parents[2]
    db_path = root_dir / "data" / "medismart.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"
