# Configuration management for Event Camera Simulation
"""
Centralized configuration management for the event simulation system.
Handles environment variables, default settings, and parameter validation.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for event processing parameters"""
    default_threshold: int = 15
    min_threshold: int = 3
    max_threshold: int = 50
    default_max_frames: int = 300
    default_time_window_ms: float = 30.0
    max_events_per_frame: int = 5000
    enable_multi_scale: bool = True
    motion_boost_enabled: bool = True
    noise_level: float = 0.01


@dataclass
class VideoConfig:
    """Configuration for video processing"""
    supported_formats: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm', '.m4v', '.flv')
    default_output_fps: int = 30
    max_video_size_mb: int = 500
    default_codec: str = 'mp4v'
    fallback_codec: str = 'XVID'
    quality_warning_threshold_mb: int = 100


@dataclass
class VisualizationConfig:
    """Configuration for visualization and rendering"""
    positive_color_bgr: tuple = (0, 0, 255)    # Red for brightening
    negative_color_bgr: tuple = (255, 0, 0)    # Blue for darkening
    background_color_bgr: tuple = (0, 0, 0)    # Black background
    decay_factor: float = 0.85
    glow_radius: int = 2
    intensity_high: int = 255
    intensity_medium: int = 200
    intensity_low: int = 150
    visibility_threshold: int = 15


@dataclass
class OutputConfig:
    """Configuration for output generation"""
    output_directory: str = "output"
    csv_filename: str = "events.csv"
    video_filename: str = "event_video.mp4"
    heatmap_filename: str = "activity_heatmap.png"
    analysis_filename: str = "analysis_report.png"
    dpi: int = 300
    figure_size: tuple = (12, 8)


class ConfigManager:
    """Central configuration manager for the application"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to custom configuration file
        """
        self.processing = ProcessingConfig()
        self.video = VideoConfig()
        self.visualization = VisualizationConfig()
        self.output = OutputConfig()
        
        # Load environment variables
        self._load_from_env()
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        # Processing settings
        if threshold := os.getenv('EVENT_THRESHOLD'):
            self.processing.default_threshold = int(threshold)
        
        if max_frames := os.getenv('MAX_FRAMES'):
            self.processing.default_max_frames = int(max_frames)
        
        if time_window := os.getenv('TIME_WINDOW_MS'):
            self.processing.default_time_window_ms = float(time_window)
        
        # Output settings
        if output_dir := os.getenv('OUTPUT_DIRECTORY'):
            self.output.output_directory = output_dir
        
        # Video settings
        if output_fps := os.getenv('OUTPUT_FPS'):
            self.video.default_output_fps = int(output_fps)
        
        # Visualization settings
        if decay_factor := os.getenv('DECAY_FACTOR'):
            self.visualization.decay_factor = float(decay_factor)
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            if 'processing' in config_data:
                for key, value in config_data['processing'].items():
                    if hasattr(self.processing, key):
                        setattr(self.processing, key, value)
            
            if 'video' in config_data:
                for key, value in config_data['video'].items():
                    if hasattr(self.video, key):
                        setattr(self.video, key, value)
            
            if 'visualization' in config_data:
                for key, value in config_data['visualization'].items():
                    if hasattr(self.visualization, key):
                        setattr(self.visualization, key, value)
            
            if 'output' in config_data:
                for key, value in config_data['output'].items():
                    if hasattr(self.output, key):
                        setattr(self.output, key, value)
                        
            logger.info(f"Configuration loaded from {config_file}")
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        # Validate threshold ranges
        if not (self.processing.min_threshold <= self.processing.default_threshold <= self.processing.max_threshold):
            raise ValueError(f"Invalid threshold configuration: {self.processing.default_threshold}")
        
        # Validate output directory
        try:
            Path(self.output.output_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create output directory: {e}")
        
        # Validate color values
        for color in [self.visualization.positive_color_bgr, self.visualization.negative_color_bgr]:
            if not all(0 <= c <= 255 for c in color):
                raise ValueError(f"Invalid color values: {color}")
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get all output file paths"""
        output_dir = Path(self.output.output_directory)
        
        return {
            'csv': str(output_dir / self.output.csv_filename),
            'video': str(output_dir / self.output.video_filename),
            'heatmap': str(output_dir / self.output.heatmap_filename),
            'analysis': str(output_dir / self.output.analysis_filename)
        }


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config