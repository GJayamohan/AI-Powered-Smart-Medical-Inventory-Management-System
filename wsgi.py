"""
WSGI Application entry point.

Run locally with:
    python wsgi.py
Or with Flask CLI:
    flask --app wsgi.py run --port 5000
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from medismart.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1")
    print(f"\n========================================================")
    print(f" MediSmart API Server starting on http://127.0.0.1:{port}")
    print(f" SQLite Database: data/medismart.db")
    print(f"========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
