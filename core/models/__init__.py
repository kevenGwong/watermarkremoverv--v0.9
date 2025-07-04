"""
Core models module for AI watermark removal
"""

from .base_model import BaseModel
from .iopaint_processor import IOPaintProcessor
from .lama_processor import LamaProcessor

__all__ = ['BaseModel', 'IOPaintProcessor', 'LamaProcessor']