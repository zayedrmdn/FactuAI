"""
FactuAI Database Module

Database connection and models.
"""

from database.connection import db
from database.models.user import User

__all__ = ["db", "User"]
