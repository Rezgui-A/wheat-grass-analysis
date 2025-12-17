"""
Vercel Serverless Function Wrapper for Wheatgrass Analyzer
This allows your exact Flask app to run on Vercel
"""

import sys
import os

# Add the parent directory to the path so we can import app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your Flask app
from app import app

# Vercel serverless function handler
def handler(request, context):
    """
    AWS Lambda/API Gateway handler for Vercel
    """
    from flask import make_response
    import io
    import base64
    
    # Parse the request
    method = request['requestContext']['http']['method']
    path = request['rawPath']
    headers = request.get('headers', {})
    query_params = request.get('queryStringParameters', {})
    
    # Handle multipart form data (file uploads)
    if method == 'POST' and 'content-type' in headers:
        content_type = headers['content-type']
        if 'multipart/form-data' in content_type:
            # Parse multipart form data
            import cgi
            import urllib.parse
            
            body = base64.b64decode(request.get('body', ''))
            environ = {
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': content_type,
                'CONTENT_LENGTH': str(len(body)),
                'wsgi.input': io.BytesIO(body),
            }
            
            with app.app_context():
                from werkzeug.datastructures import ImmutableMultiDict
                from werkzeug.formparser import FormDataParser
                
                parser = FormDataParser()
                stream, form, files = parser.parse(environ['wsgi.input'], environ['CONTENT_TYPE'], environ['CONTENT_LENGTH'], {})
                
                # Create request context
                with app.test_request_context(
                    path=path,
                    method=method,
                    headers=headers,
                    data=form,
                    files=files
                ):
                    response = app.full_dispatch_request()
                    return {
                        'statusCode': response.status_code,
                        'headers': dict(response.headers),
                        'body': response.get_data(as_text=True)
                    }
    
    # Handle regular requests
    with app.test_request_context(
        path=path,
        method=method,
        headers=headers,
        query_string=query_params
    ):
        response = app.full_dispatch_request()
        return {
            'statusCode': response.status_code,
            'headers': dict(response.headers),
            'body': response.get_data(as_text=True)
        }

# Alternative: Direct WSGI handler for simpler setup
if __name__ == "__main__":
    # Local development
    app.run(debug=True, host='0.0.0.0', port=5000)