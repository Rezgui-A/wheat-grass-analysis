import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['VERCEL'] = '1'

# This line imports the 'app' variable from your main app.py file
from app import app

# Vercel's runtime will automatically detect the 'app' variable.
# Do NOT define a custom 'handler' function here.