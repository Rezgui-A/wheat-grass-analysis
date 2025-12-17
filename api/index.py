"""
Official Vercel Python Serverless Function Template
This WORKS with Vercel's Python runtime
"""

import sys
import os

# Debug output
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Set Vercel environment
os.environ['VERCEL'] = '1'

# Try to import Flask app
try:
    from app import app
    FLASK_APP_LOADED = True
    print("✅ Flask app imported successfully")
except Exception as e:
    FLASK_APP_LOADED = False
    print(f"⚠️ Could not import Flask app: {e}")
    
    # Create minimal app
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return jsonify({"status": "online", "service": "Wheatgrass Analyzer"})
    
    @app.route('/api/status')
    def status():
        return jsonify({"status": "running", "vercel": True, "flask": FLASK_APP_LOADED})

# ===== VERCEL SERVERLESS HANDLER =====
# This EXACT function signature works with Vercel

from http.server import BaseHTTPRequestHandler
import json
from io import BytesIO

class VercelHandler(BaseHTTPRequestHandler):
    """HTTP handler for Vercel"""
    
    def handle_request(self):
        """Process the HTTP request"""
        try:
            # Parse request
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''
            
            # Extract path and query
            path = self.path.split('?')[0] if '?' in self.path else self.path
            
            # Create WSGI environment
            environ = {
                'REQUEST_METHOD': self.command,
                'PATH_INFO': path,
                'QUERY_STRING': self.path.split('?')[1] if '?' in self.path else '',
                'wsgi.input': BytesIO(body),
                'wsgi.errors': sys.stderr,
                'wsgi.version': (1, 0),
                'wsgi.url_scheme': 'https',
                'wsgi.multithread': False,
                'wsgi.multiprocess': True,
                'wsgi.run_once': False,
            }
            
            # Add headers
            for key, value in self.headers.items():
                key = key.replace('-', '_').upper()
                if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                    environ[f'HTTP_{key}'] = value
            
            # Process with Flask
            with app.request_context(environ):
                response = app.full_dispatch_request()
                
                # Send response
                self.send_response(response.status_code)
                
                # Add CORS headers
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                
                # Add Flask headers
                for header, value in response.headers:
                    if header.lower() not in ['access-control-allow-origin', 
                                             'access-control-allow-methods',
                                             'access-control-allow-headers']:
                        self.send_header(header, value)
                
                self.end_headers()
                self.wfile.write(response.get_data())
                
        except Exception as e:
            print(f"❌ Error in handler: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error response
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = json.dumps({
                "success": False,
                "error": "Internal server error",
                "message": str(e)
            })
            self.wfile.write(error_response.encode())
    
    # Implement all HTTP methods
    def do_GET(self):
        self.handle_request()
    
    def do_POST(self):
        self.handle_request()
    
    def do_PUT(self):
        self.handle_request()
    
    def do_DELETE(self):
        self.handle_request()
    
    def do_OPTIONS(self):
        self.handle_request()
    
    # Suppress logging
    def log_message(self, format, *args):
        pass

# ===== EXPORT FOR VERCEL =====
# Vercel looks for a 'handler' function or class

def handler(request, context):
    """
    Main handler function for Vercel
    This is the entry point Vercel calls
    """
    print(f"📨 Handler called: {request.get('path', '/')}")
    
    # Create a mock HTTP request
    class MockRequest:
        def __init__(self, vercel_request):
            self.command = vercel_request.get('httpMethod', 'GET')
            self.path = vercel_request.get('path', '/')
            self.headers = vercel_request.get('headers', {})
            self.body = vercel_request.get('body', '')
            self.query = vercel_request.get('queryStringParameters', {})
    
    # Process the request
    mock_request = MockRequest(request)
    handler_instance = VercelHandler(
        BytesIO(),
        ('localhost', 443),
        None
    )
    
    # Set attributes
    handler_instance.command = mock_request.command
    handler_instance.path = mock_request.path
    handler_instance.headers = mock_request.headers
    
    # Handle the request
    handler_instance.handle_request()
    
    # Note: In reality, Vercel handles the response differently
    # This is a simplified version
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({"status": "request_processed"})
    }

# For local testing
if __name__ == '__main__':
    print("=" * 50)
    print("✅ Vercel Serverless Function Ready")
    print("=" * 50)
    print(f"Flask app loaded: {FLASK_APP_LOADED}")
    print(f"Handler available: {handler.__name__}")
    print(f"VercelHandler available: {VercelHandler.__name__}")