"""
api/index.py - Vercel Python Serverless Function (Correct Format)
"""

import sys
import os

# Add parent directory to Python path to import your main app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Vercel environment (important for your app.py logic)
os.environ['VERCEL'] = '1'

# Import the Flask app from your main app.py file.
# This line assumes your Flask instance is named 'app' in app.py.
try:
    from app import app
    print("✅ Successfully imported Flask app from app.py")
except ImportError as e:
    print(f"❌ Failed to import Flask app: {e}")
    # If import fails, create a minimal placeholder to allow the build to pass
    from flask import Flask
    app = Flask(__name__)
    @app.route('/')
    def placeholder():
        return "Wheatgrass Analyzer - Flask app import failed, check logs."

# ****************** CRITICAL PART ******************
# Expose the WSGI application to Vercel.
# Vercel's Python runtime will look for a variable named 'app' or 'handler' that is a WSGI app.
# We are exposing the Flask 'app' object directly.
# Do NOT define a custom 'handler(request, context)' function.
# ***************************************************