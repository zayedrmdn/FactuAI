#!/usr/bin/env python3
"""
Test PostgreSQL connection using the DB_URI from .env
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Get DB URI
db_uri = os.getenv("DB_URI")
if not db_uri:
    print("ERROR: DB_URI not found in .env")
    exit(1)

print(f"Testing connection to: {db_uri}")

try:
    # Create engine
    engine = create_engine(db_uri)

    # Test connection
    with engine.connect() as conn:
        # Test query
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        print("✅ Connected successfully!")
        print(f"PostgreSQL version: {version}")

        # Check if user table exists
        result = conn.execute(text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'user');"))
        exists = result.fetchone()[0]
        if exists:
            print("✅ 'user' table exists")
            # Count users
            result = conn.execute(text("SELECT COUNT(*) FROM \"user\";"))
            count = result.fetchone()[0]
            print(f"✅ Found {count} users in the table")
        else:
            print("❌ 'user' table does not exist")

except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure PostgreSQL is running")
    print("2. Check the DB_URI in .env - ensure username, password, host, port, and database name are correct")
    print("3. If password contains special characters, try URL encoding them")
    print("4. Try connecting with psql: psql -h localhost -U postgres -d factuai_db")