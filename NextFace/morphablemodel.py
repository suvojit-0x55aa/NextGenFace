"""Shim - redirects to new location."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from facemodel.morphablemodel import *
from facemodel.morphablemodel import MorphableModel
