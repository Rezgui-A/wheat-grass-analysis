"""
api/index.py - Vercel Python Serverless Function
This EXACT structure works on Vercel
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Vercel environment
os.environ['VERCEL'] = '1'

# Import your Flask app
try:
    from app import app
    print("✅ Flask app imported successfully")
except Exception as e:
    print(f"❌ Failed to import Flask app: {e}")
    
    # Create minimal Flask app for fallback
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return json.dumps({"status": "online", "message": "Wheatgrass Analyzer"})
    
    @app.route('/api/status')
    def status():
        return json.dumps({"status": "running", "vercel": True})

# Vercel Python Runtime expects this exact class
class handler(BaseHTTPRequestHandler):
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
    
    def handle_request(self):
        """Handle all HTTP methods"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''
            
            # Parse query string
            if '?' in self.path:
                path, query_string = self.path.split('?', 1)
            else:
                path = self.path
                query_string = ''
            
            # Create WSGI environment
            environ = {
                'REQUEST_METHOD': self.command,
                'PATH_INFO': path,
                'QUERY_STRING': query_string,
                'SERVER_PROTOCOL': self.request_version,
                'wsgi.input': BytesIO(body),
                'wsgi.errors': sys.stderr,
                'wsgi.version': (1, 0),
                'wsgi.url_scheme': 'https',
                'wsgi.multithread': False,
                'wsgi.multiprocess': True,
                'wsgi.run_once': False,
                'SERVER_NAME': 'localhost',
                'SERVER_PORT': '443',
                'REMOTE_ADDR': self.client_address[0],
            }
            
            # Add headers
            for key, value in self.headers.items():
                key = key.upper().replace('-', '_')
                if key not in ['CONTENT_TYPE', 'CONTENT_LENGTH']:
                    environ['HTTP_' + key] = value
                else:
                    environ[key] = value
            
            # Process with Flask
            with app.request_context(environ):
                response = app.full_dispatch_request()
                
                # Send response
                self.send_response(response.status_code)
                
                # Add CORS headers
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                
                # Add other headers
                for header, value in response.headers:
                    self.send_header(header, value)
                
                self.end_headers()
                self.wfile.write(response.get_data())
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            import traceback
            traceback.print_exc()
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_msg = json.dumps({"success": False, "error": str(e)})
            self.wfile.write(error_msg.encode())
    
    # Silence the log output
    def log_message(self, format, *args):
        pass

# Alternative handler for Vercel's newer runtime
def vercel_handler(request, context):
    """
    Alternative handler for Vercel's serverless function
    """
    try:
        print(f"📨 Vercel handler called: {request.get('path', '/')}")
        
        from flask import Request
        from werkzeug.datastructures import Headers
        
        method = request.get('httpMethod', 'GET')
        path = request.get('path', '/')
        headers = request.get('headers', {})
        query_string = request.get('queryStringParameters', {})
        body = request.get('body', '')
        
        # Handle base64 encoded body
        if request.get('isBase64Encoded', False):
            import base64
            body = base64.b64decode(body)
        
        # Create Flask request context
        with app.test_request_context(
            path=path,
            method=method,
            headers=headers,
            data=body,
            query_string=query_string
        ):
            response = app.full_dispatch_request()
            
            return {
                'statusCode': response.status_code,
                'headers': dict(response.headers),
                'body': response.get_data(as_text=True),
                'isBase64Encoded': False
            }
            
    except Exception as e:
        print(f"❌ Vercel handler error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"success": False, "error": str(e)}),
            'isBase64Encoded': False
        }

# Export both handlers for compatibility
__all__ = ['handler', 'vercel_handler']

# For local testing
if __name__ == '__main__':
    print("✅ Vercel handler module loaded successfully")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Files in api/: {os.listdir(os.path.dirname(__file__))}")