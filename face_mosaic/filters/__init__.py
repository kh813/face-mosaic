"""
Filters package init.
"""

from .animal_classifier import AnimalClassifier
from .skin_color import SkinColorFilter
from .static_photo import StaticPhotoFilter
from .illustration import IllustrationFilter

__all__ = ["AnimalClassifier", "SkinColorFilter", "StaticPhotoFilter", "IllustrationFilter"]
