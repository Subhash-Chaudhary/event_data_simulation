"""
Event Camera Data Simulation Package
A production-ready neuromorphic vision simulation system.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Event Camera Data Simulation for Neuromorphic Vision"

from .core.event_detector import EventDetector
from .config import get_config, ConfigManager

__all__ = [
    "EventDetector",
    "get_config", 
    "ConfigManager"
]