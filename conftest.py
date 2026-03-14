import sys
import os

# Add the root folder to Python's search path
# This lets pytest find state.py, agents/, utils/ etc.
sys.path.insert(0, os.path.dirname(__file__))