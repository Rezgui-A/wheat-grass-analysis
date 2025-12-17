"""
api/index.py - Vercel Serverless Function
MAKE SURE THIS FILE IS IN /api/ folder!
"""

import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Vercel environment
os.environ['VERCEL'] = '1'

# Import your Flask app
from app import app

# Vercel requires this exact function name
def handler(request, context):
    """
    Vercel Serverless Function Handler
    """
    # Import Flask's request here to avoid circular imports
    from flask import Request
    from werkzeug.datastructures import Headers
    
    # Create a Flask request from Vercel's request
    method = request.get('httpMethod', 'GET')
    headers = Headers(request.get('headers', {}))
    
    # Create request body
    body = request.get('body', '')
    if request.get('isBase64Encoded', False):
        import base64
        body = base64.b64decode(body)
    
    # Create Flask request
    with app.test_request_context(
        path=request.get('path', '/'),
        method=method,
        headers=headers,
        data=body,
        query_string=request.get('queryStringParameters', {})
    ):
        # Process the request
        response = app.full_dispatch_request()
        
        # Convert to Vercel format
        return {
            'statusCode': response.status_code,
            'headers': dict(response.headers),
            'body': response.get_data(as_text=True),
            'isBase64Encoded': False
        }

# Optional: For local testing
if __name__ == '__main__':
    print("✅ Vercel handler ready")