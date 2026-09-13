"""
Filters package init.
"""

from .animal_classifier import AnimalClassifier
from .skin_color import SkinColorFilter
from .static_photo import StaticPhotoFilter

__all__ = ["AnimalClassifier", "SkinColorFilter", "StaticPhotoFilter"]
