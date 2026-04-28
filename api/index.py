"""Vercel Python serverless entry point.

Vercel loads this file and uses `app` as the WSGI handler.
All routes defined in web/server.py are served through here.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from web.server import app  # noqa: F401  (Vercel picks up `app` automatically)
