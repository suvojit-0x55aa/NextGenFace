"""Shim - redirects to new locations after decomposition."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from geometry.obj_export import saveObj
from landmarks._viz import saveLandmarksVerticesProjections
from facemodel._pickle_io import loadDictionaryFromPickle, writeDictionaryToPickle


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
