"""Shim - redirects to new location."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from landmarks.mediapipe import *
from landmarks.mediapipe import LandmarksDetectorMediapipe
