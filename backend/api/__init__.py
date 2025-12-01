"""
FactuAI API Module

Flask blueprints for all API endpoints.
Migrated from routes/ for cleaner naming.
"""

from api.auth import auth_bp
from api.profile import profile_bp
from api.factcheck import bp_fact
from api.process import bp as bp_process
from api.image import bp_image
from api.video import bp_video

__all__ = [
    "auth_bp",
    "profile_bp",
    "bp_fact",
    "bp_process",
    "bp_image",
    "bp_video",
]
