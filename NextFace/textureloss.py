"""Shim - redirects to new location."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from optim.textureloss import *
from optim.textureloss import TextureLoss
