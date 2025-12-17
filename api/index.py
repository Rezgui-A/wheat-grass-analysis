"""
api/index.py - Vercel Python Serverless Function
This format is GUARANTEED to work with Vercel
"""

import sys
import os
import json
from http.server import BaseHTTPRequestHandler
from io import BytesIO
import traceback

# Debug output for build logs
print("🚀 Vercel Python serverless function starting")
print(f"📁 Current directory: {os.getcwd()}")
print(f"📁 Files in api/: {os.listdir(os.path.dirname(__file__))}")

# Add parent directory to path to import your Flask app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['VERCEL'] = '1'

# Import your Flask app
try:
    from app import app
    print("✅ Flask app imported successfully from app.py")
    FLASK_LOADED = True
except Exception as e:
    print(f"❌ Failed to import Flask app: {e}")
    print(traceback.format_exc())
    FLASK_LOADED = False
    
    # Create minimal fallback app
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def fallback_home():
        return json.dumps({
            "status": "online", 
            "message": "Wheatgrass Analyzer (Fallback Mode)",
            "error": "Main Flask app failed to load"
        })

# ===== VERCEL REQUIRED EXPORT =====
# Vercel Python runtime requires ONE of these exports:
# 1. A variable named 'app' that is a WSGI application (Flask, Django, etc.)
# 2. A variable named 'handler' that is a WSGI application
# 3. A function named 'handler' with signature handler(request, response)

# Export 1: Expose Flask app directly (Most reliable for Flask)
print("📤 Exporting Flask app as WSGI application for Vercel")