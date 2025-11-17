# Custom exceptions for Event Camera Simulation
"""
Custom exception classes for the event simulation system.
Provides specific error types for better error handling and debugging.
"""

class EventSimulationError(Exception):
    """Base exception class for all event simulation errors"""
    pass


class VideoProcessingError(EventSimulationError):
    """Raised when video processing encounters an error"""
    pass


class EventDetectionError(EventSimulationError):
    """Raised when event detection algorithms fail"""
    pass


class ConfigurationError(EventSimulationError):
    """Raised when configuration is invalid or missing"""
    pass


class FileValidationError(EventSimulationError):
    """Raised when file validation fails"""
    pass


class VisualizationError(EventSimulationError):
    """Raised when visualization generation fails"""
    pass


class ExportError(EventSimulationError):
    """Raised when data export operations fail"""
    pass


class MemoryError(EventSimulationError):
    """Raised when memory constraints are exceeded"""
    pass


class ThresholdError(ConfigurationError):
    """Raised when threshold parameters are invalid"""
    pass


class VideoFormatError(VideoProcessingError):
    """Raised when video format is unsupported or corrupted"""
    pass