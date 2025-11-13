"""
Data processing modules for fake news detection
"""

from .preprocess import TextPreprocessor
from .loader import DataLoader

__all__ = ['TextPreprocessor', 'DataLoader']
