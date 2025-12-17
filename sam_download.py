"""
Vercel Serverless Function Handler
This file MUST be in /api/index.py for Vercel
"""

import os
import sys
import json
from io import BytesIO
import base64

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
    raise

def handler(event, context):
    """
    Vercel Serverless Function Handler
    Converts Vercel request to Flask request
    """
    try:
        # Parse Vercel event
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        headers = {k.lower(): v for k, v in event.get('headers', {}).items()}
        query_params = event.get('queryStringParameters', {}) or {}
        body = event.get('body', '')
        
        # Handle multipart/form-data (file uploads)
        if http_method == 'POST' and 'content-type' in headers:
            content_type = headers['content-type']
            
            if 'multipart/form-data' in content_type:
                # Import here to avoid circular imports
                from flask import request as flask_request
                from werkzeug.datastructures import ImmutableMultiDict
                
                # Create a mock request environment
                environ = {
                    'REQUEST_METHOD': 'POST',
                    'PATH_INFO': path,
                    'QUERY_STRING': '',
                    'CONTENT_TYPE': content_type,
                    'CONTENT_LENGTH': str(len(body)),
                    'wsgi.input': BytesIO(base64.b64decode(body) if body else b''),
                    'wsgi.errors': sys.stderr,
                    'wsgi.version': (1, 0),
                    'wsgi.url_scheme': 'https',
                    'wsgi.multithread': False,
                    'wsgi.multiprocess': True,
                    'wsgi.run_once': False,
                    'SERVER_NAME': 'localhost',
                    'SERVER_PORT': '443',
                    'REMOTE_ADDR': '127.0.0.1',
                }
                
                # Add headers
                for key, value in headers.items():
                    if key not in ['content-type', 'content-length']:
                        environ[f'HTTP_{key.upper().replace("-", "_")}'] = value
                
                # Create request context
                with app.request_context(environ):
                    try:
                        # Parse multipart data
                        data = flask_request.form
                        files = flask_request.files
                        
                        # Process the request
                        response = app.full_dispatch_request()
                    except Exception as e:
                        response = app.make_response((
                            json.dumps({'success': False, 'error': str(e)}),
                            500,
                            {'Content-Type': 'application/json'}
                        ))
            else:
                # Regular JSON request
                environ = {
                    'REQUEST_METHOD': http_method,
                    'PATH_INFO': path,
                    'QUERY_STRING': '&'.join([f'{k}={v}' for k, v in query_params.items()]),
                    'CONTENT_TYPE': content_type,
                    'CONTENT_LENGTH': str(len(body)),
                    'wsgi.input': BytesIO(body.encode() if isinstance(body, str) else body),
                    'wsgi.errors': sys.stderr,
                    'wsgi.version': (1, 0),
                    'wsgi.url_scheme': 'https',
                    'wsgi.multithread': False,
                    'wsgi.multiprocess': True,
                    'wsgi.run_once': False,
                    'SERVER_NAME': 'localhost',
                    'SERVER_PORT': '443',
                    'REMOTE_ADDR': '127.0.0.1',
                }
                
                # Add headers
                for key, value in headers.items():
                    if key not in ['content-type', 'content-length']:
                        environ[f'HTTP_{key.upper().replace("-", "_")}'] = value
                
                # Handle the request
                with app.request_context(environ):
                    response = app.full_dispatch_request()
        else:
            # GET or other simple requests
            environ = {
                'REQUEST_METHOD': http_method,
                'PATH_INFO': path,
                'QUERY_STRING': '&'.join([f'{k}={v}' for k, v in query_params.items()]),
                'wsgi.input': BytesIO(),
                'wsgi.errors': sys.stderr,
                'wsgi.version': (1, 0),
                'wsgi.url_scheme': 'https',
                'wsgi.multithread': False,
                'wsgi.multiprocess': True,
                'wsgi.run_once': False,
                'SERVER_NAME': 'localhost',
                'SERVER_PORT': '443',
                'REMOTE_ADDR': '127.0.0.1',
            }
            
            # Add headers
            for key, value in headers.items():
                if key not in ['content-type', 'content-length']:
                    environ[f'HTTP_{key.upper().replace("-", "_")}'] = value
            
            # Handle the request
            with app.request_context(environ):
                response = app.full_dispatch_request()
        
        # Convert Flask response to Vercel format
        response_data = response.get_data()
        
        return {
            'statusCode': response.status_code,
            'headers': {
                'Content-Type': response.headers.get('Content-Type', 'application/json'),
                'Access-Control-Allow-Origin': '*',
            },
            'body': response_data.decode('utf-8') if isinstance(response_data, bytes) else str(response_data),
            'isBase64Encoded': False
        }
        
    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'success': False,
                'error': f'Server error: {str(e)}',
                'message': 'Check server logs for details'
            }),
            'isBase64Encoded': False
        }

# For local testing
if __name__ == '__main__':
    # Test the handler locally
    test_event = {
        'httpMethod': 'GET',
        'path': '/api/status',
        'headers': {},
        'queryStringParameters': {},
        'body': ''
    }
    
    result = handler(test_event, None)
    print(json.dumps(result, indent=2))