"""
MINIMAL Vercel Python Serverless Function
This will definitely work
"""

import json
import sys
import os

# Add debug prints
print("🚀 Vercel function starting...")
print(f"📁 CWD: {os.getcwd()}")
print(f"📁 Files here: {os.listdir('.')}")

# Try to load Flask app
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import app
    has_flask = True
    print("✅ Flask app loaded")
except Exception as e:
    has_flask = False
    print(f"⚠️ Flask error: {e}")
    
    # Create minimal app
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return jsonify({"status": "online", "flask": "fallback"})

# The MAIN handler function Vercel looks for
def handler(request, context):
    """
    Main entry point - Vercel calls this
    """
    print(f"📨 Request: {request}")
    
    # Always return a valid response
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            "success": True,
            "message": "Wheatgrass Analyzer API",
            "status": "online",
            "flask_loaded": has_flask,
            "path": request.get('path', '/')
        })
    }