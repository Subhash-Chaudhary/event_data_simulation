# Core event detection algorithms for neuromorphic vision simulation
"""
Advanced event detection system implementing multi-scale temporal integration,
adaptive thresholding, and polarity-aware event generation.

This module contains the core algorithms that convert conventional video frames
into event streams compatible with Dynamic Vision Sensor (DVS) specifications.
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EventData:
    """Data structure for individual event information"""
    timestamp_us: float
    x: int
    y: int
    polarity: int  # +1 for brightening, -1 for darkening
    magnitude: float = 0.0
    confidence: float = 1.0


@dataclass
class FrameProcessingResult:
    """Result data structure for frame processing"""
    events: List[EventData]
    frame_index: int
    processing_time_ms: float
    total_pixels_analyzed: int
    events_detected: int


class ProgressTracker:
    """Simple progress tracking for operations"""
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        
    def update(self):
        self.current_step += 1
        
    def get_progress_info(self):
        percentage = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'percentage': percentage
        }


class EventDetector:
    """
    Advanced event detection engine with multi-scale processing and
    adaptive thresholding for neuromorphic vision simulation.
    """
    
    def __init__(self, 
                 threshold: int = 15,
                 enable_multi_scale: bool = True,
                 motion_boost: bool = True,
                 noise_level: float = 0.01):
        """
        Initialize event detector with configuration
        
        Args:
            threshold: Base threshold for event detection
            enable_multi_scale: Enable multi-resolution processing
            motion_boost: Apply motion magnitude boosting
            noise_level: Amount of realistic noise to add
        """
        self.threshold = threshold
        self.enable_multi_scale = enable_multi_scale
        self.motion_boost = motion_boost
        self.noise_level = noise_level
        
        # Validate threshold
        if not (3 <= threshold <= 50):
            raise ValueError(f"Threshold {threshold} outside valid range [3, 50]")
        
        # Initialize processing state
        self.prev_frame = None
        self.prev_frame_secondary = None
        self.frame_count = 0
        self.total_events_generated = 0
        
        logger.info(f"EventDetector initialized - threshold: {self.threshold}, multi_scale: {self.enable_multi_scale}")
    
    def detect_events_in_frame(self, 
                              current_frame: np.ndarray, 
                              timestamp_us: float,
                              fps: float) -> FrameProcessingResult:
        """
        Detect events in a single frame using advanced algorithms
        
        Args:
            current_frame: Current video frame (BGR format)
            timestamp_us: Frame timestamp in microseconds
            fps: Video frame rate
            
        Returns:
            FrameProcessingResult containing detected events and metadata
        """
        start_time = cv2.getTickCount()
        
        try:
            # Convert to grayscale for processing
            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            events = []
            
            if self.prev_frame is not None:
                # Primary event detection
                primary_events = self._detect_primary_events(gray_frame, timestamp_us, fps)
                events.extend(primary_events)
                
                # Multi-scale detection for small objects
                if self.enable_multi_scale:
                    multiscale_events = self._detect_multiscale_events(gray_frame, timestamp_us, fps)
                    events.extend(multiscale_events)
                
                # Add realistic noise events
                noise_events = self._generate_noise_events(gray_frame, timestamp_us)
                events.extend(noise_events)
            
            # Update frame history
            self._update_frame_history(gray_frame)
            
            # Calculate processing metrics
            end_time = cv2.getTickCount()
            processing_time_ms = (end_time - start_time) / cv2.getTickFrequency() * 1000
            
            self.total_events_generated += len(events)
            
            result = FrameProcessingResult(
                events=events,
                frame_index=self.frame_count,
                processing_time_ms=processing_time_ms,
                total_pixels_analyzed=gray_frame.shape[0] * gray_frame.shape[1],
                events_detected=len(events)
            )
            
            self.frame_count += 1
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            raise
    
    def _detect_primary_events(self, 
                              gray_frame: np.ndarray, 
                              timestamp_us: float, 
                              fps: float) -> List[EventData]:
        """
        Primary event detection using frame differencing and adaptive thresholding
        """
        events = []
        
        try:
            # Calculate frame difference
            frame_diff = cv2.absdiff(gray_frame, self.prev_frame)
            
            # Apply motion boosting if enabled
            if self.motion_boost:
                frame_diff = self._apply_motion_boosting(frame_diff)
            
            # Temporal integration with secondary frame if available
            if self.prev_frame_secondary is not None:
                secondary_diff = cv2.absdiff(gray_frame, self.prev_frame_secondary)
                # Combine differences using numpy maximum (FIXED: was cv2.maximum)
                integrated_diff = np.maximum(frame_diff, (secondary_diff * 0.7).astype(np.uint8))
            else:
                integrated_diff = frame_diff
            
            # Adaptive thresholding
            adaptive_threshold = self._calculate_adaptive_threshold(integrated_diff)
            
            # Find event coordinates
            y_coords, x_coords = np.where(integrated_diff > adaptive_threshold)
            
            if len(y_coords) > 0:
                # Limit events for performance
                events_data = self._process_event_coordinates(
                    x_coords, y_coords, gray_frame, timestamp_us, integrated_diff
                )
                events.extend(events_data)
        
        except Exception as e:
            logger.warning(f"Primary event detection error: {e}")
        
        return events
    
    def _detect_multiscale_events(self, 
                                 gray_frame: np.ndarray, 
                                 timestamp_us: float, 
                                 fps: float) -> List[EventData]:
        """
        Multi-scale event detection for enhanced small object sensitivity
        """
        events = []
        scale_factor = 1.5
        
        try:
            # Upscale current and previous frames
            upscaled_current = cv2.resize(gray_frame, None, fx=scale_factor, fy=scale_factor, 
                                        interpolation=cv2.INTER_CUBIC)
            upscaled_prev = cv2.resize(self.prev_frame, None, fx=scale_factor, fy=scale_factor,
                                     interpolation=cv2.INTER_CUBIC)
            
            # Calculate difference at upscaled resolution
            upscaled_diff = cv2.absdiff(upscaled_current, upscaled_prev)
            
            # Use lower threshold for upscaled processing
            upscaled_threshold = max(int(self.threshold * 0.8), 3)
            
            # Find events at upscaled resolution
            y_coords_up, x_coords_up = np.where(upscaled_diff > upscaled_threshold)
            
            if len(y_coords_up) > 0:
                # Scale coordinates back to original resolution
                x_coords_orig = (x_coords_up / scale_factor).astype(int)
                y_coords_orig = (y_coords_up / scale_factor).astype(int)
                
                # Filter coordinates within original frame bounds
                valid_mask = ((x_coords_orig < gray_frame.shape[1]) & 
                             (y_coords_orig < gray_frame.shape[0]))
                
                x_coords_valid = x_coords_orig[valid_mask]
                y_coords_valid = y_coords_orig[valid_mask]
                
                # Process valid coordinates
                if len(x_coords_valid) > 0:
                    multiscale_events = self._process_event_coordinates(
                        x_coords_valid, y_coords_valid, gray_frame, timestamp_us, upscaled_diff
                    )
                    events.extend(multiscale_events)
        
        except Exception as e:
            logger.warning(f"Multi-scale detection error: {e}")
        
        return events
    
    def _apply_motion_boosting(self, frame_diff: np.ndarray) -> np.ndarray:
        """Apply motion magnitude boosting for enhanced detection of fast objects"""
        motion_magnitude = frame_diff.astype(np.float32)
        motion_threshold = self.threshold * 0.5
        high_motion_mask = motion_magnitude > motion_threshold
        motion_magnitude[high_motion_mask] *= 1.5
        return np.clip(motion_magnitude, 0, 255).astype(np.uint8)
    
    def _calculate_adaptive_threshold(self, frame_diff: np.ndarray) -> int:
        """Calculate adaptive threshold based on local image statistics"""
        base_threshold = self.threshold
        mean_diff = np.mean(frame_diff)
        std_diff = np.std(frame_diff)
        
        if std_diff > 0:
            adaptive_factor = min(2.0, max(0.5, std_diff / 20.0))
            adaptive_threshold = max(base_threshold, int(base_threshold * adaptive_factor))
        else:
            adaptive_threshold = base_threshold
        
        return min(adaptive_threshold, 50)
    
    def _process_event_coordinates(self, 
                                  x_coords: np.ndarray, 
                                  y_coords: np.ndarray,
                                  gray_frame: np.ndarray, 
                                  timestamp_us: float,
                                  diff_frame: np.ndarray) -> List[EventData]:
        """Process detected event coordinates and generate EventData objects"""
        events = []
        max_events = 5000  # Performance limit
        
        # Limit events for performance
        if len(x_coords) > max_events:
            # Prioritize high-magnitude events
            magnitudes = []
            for i in range(len(x_coords)):
                x, y = x_coords[i], y_coords[i]
                if y < diff_frame.shape[0] and x < diff_frame.shape[1]:
                    magnitudes.append(diff_frame[y, x])
                else:
                    magnitudes.append(0)
            
            # Select top magnitude events
            top_indices = np.argsort(magnitudes)[-max_events:]
            x_coords = x_coords[top_indices]
            y_coords = y_coords[top_indices]
        
        # Generate events
        for i in range(len(x_coords)):
            x, y = int(x_coords[i]), int(y_coords[i])
            
            # Bounds checking
            if not (0 <= x < gray_frame.shape[1] and 0 <= y < gray_frame.shape[0]):
                continue
            if not (0 <= x < self.prev_frame.shape[1] and 0 <= y < self.prev_frame.shape[0]):
                continue
            
            try:
                # Calculate polarity and magnitude
                current_intensity = int(gray_frame[y, x])
                previous_intensity = int(self.prev_frame[y, x])
                intensity_diff = current_intensity - previous_intensity
                
                # Skip insignificant changes
                if abs(intensity_diff) < self.threshold * 0.6:
                    continue
                
                # Determine polarity
                polarity = 1 if intensity_diff > 0 else -1
                magnitude = abs(intensity_diff)
                
                # Calculate confidence based on magnitude
                confidence = min(1.0, magnitude / 50.0)
                
                # Create event data
                event = EventData(
                    timestamp_us=timestamp_us,
                    x=x,
                    y=y,
                    polarity=polarity,
                    magnitude=magnitude,
                    confidence=confidence
                )
                
                events.append(event)
                
            except (IndexError, ValueError) as e:
                logger.debug(f"Event processing error at ({x}, {y}): {e}")
                continue
        
        return events
    
    def _generate_noise_events(self, gray_frame: np.ndarray, timestamp_us: float) -> List[EventData]:
        """Generate realistic noise events to simulate sensor characteristics"""
        events = []
        
        if self.noise_level <= 0:
            return events
        
        height, width = gray_frame.shape
        num_noise_events = int(width * height * self.noise_level * 0.0001)
        
        for _ in range(num_noise_events):
            # Generate random noise event
            noise_event = EventData(
                timestamp_us=timestamp_us + np.random.uniform(0, 1000),
                x=np.random.randint(0, width),
                y=np.random.randint(0, height),
                polarity=np.random.choice([-1, 1]),
                magnitude=np.random.uniform(5, 15),
                confidence=0.3
            )
            events.append(noise_event)
        
        return events
    
    def _update_frame_history(self, current_frame: np.ndarray) -> None:
        """Update frame history for temporal integration"""
        self.prev_frame_secondary = self.prev_frame.copy() if self.prev_frame is not None else None
        self.prev_frame = current_frame.copy()
    
    def process_video_stream(self, 
                           video_path: str, 
                           max_frames: Optional[int] = None,
                           progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Process complete video stream and generate event data
        
        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame containing all detected events
        """
        logger.info(f"Starting video stream processing: {video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine processing frame count
        process_frames = min(max_frames or total_frames, total_frames)
        
        logger.info(f"Video properties - Resolution: {width}x{height}, FPS: {fps:.1f}, Processing: {process_frames}/{total_frames} frames")
        
        # Initialize progress tracking
        progress_tracker = ProgressTracker(total_steps=process_frames)
        
        # Process frames
        all_events = []
        processing_stats = []
        
        try:
            while self.frame_count < process_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate frame timestamp
                timestamp_us = self.frame_count / fps * 1_000_000
                
                # Process current frame
                result = self.detect_events_in_frame(frame, timestamp_us, fps)
                all_events.extend(result.events)
                processing_stats.append(result)
                
                # Update progress
                progress_tracker.update()
                if progress_callback:
                    progress_callback(progress_tracker.get_progress_info())
                
                # Periodic logging
                if self.frame_count % 50 == 0:
                    avg_events = np.mean([r.events_detected for r in processing_stats[-50:]])
                    logger.info(f"Progress: {progress_tracker.get_progress_info()['percentage']:.1f}% - Avg events/frame: {avg_events:.1f}")
        
        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            raise
        
        finally:
            cap.release()
        
        # Convert events to DataFrame
        if all_events:
            events_df = self._events_to_dataframe(all_events)
            self._log_processing_summary(events_df, processing_stats)
            return events_df
        else:
            logger.warning("No events detected in video stream")
            return pd.DataFrame(columns=['time', 'x', 'y', 'polarity', 'magnitude', 'confidence'])
    
    def _events_to_dataframe(self, events: List[EventData]) -> pd.DataFrame:
        """Convert list of EventData objects to pandas DataFrame"""
        if not events:
            return pd.DataFrame(columns=['time', 'x', 'y', 'polarity', 'magnitude', 'confidence'])
        
        data = {
            'time': [e.timestamp_us for e in events],
            'x': [e.x for e in events],
            'y': [e.y for e in events],
            'polarity': [e.polarity for e in events],
            'magnitude': [e.magnitude for e in events],
            'confidence': [e.confidence for e in events]
        }
        
        return pd.DataFrame(data)
    
    def _log_processing_summary(self, events_df: pd.DataFrame, processing_stats: List[FrameProcessingResult]) -> None:
        """Log comprehensive processing summary"""
        if len(events_df) == 0:
            logger.warning("No events generated during processing")
            return
        
        # Calculate statistics
        total_events = len(events_df)
        avg_processing_time = np.mean([s.processing_time_ms for s in processing_stats])
        total_processing_time = sum(s.processing_time_ms for s in processing_stats)
        
        positive_events = len(events_df[events_df['polarity'] > 0])
        negative_events = len(events_df[events_df['polarity'] < 0])
        
        avg_magnitude = events_df['magnitude'].mean()
        avg_confidence = events_df['confidence'].mean()
        
        logger.info("=== PROCESSING SUMMARY ===")
        logger.info(f"Frames processed: {len(processing_stats)}")
        logger.info(f"Total events: {total_events:,}")
        logger.info(f"Positive events: {positive_events:,} ({positive_events/total_events*100:.1f}%)")
        logger.info(f"Negative events: {negative_events:,} ({negative_events/total_events*100:.1f}%)")
        logger.info(f"Average magnitude: {avg_magnitude:.2f}")
        logger.info(f"Average confidence: {avg_confidence:.2f}")
        logger.info(f"Avg processing time: {avg_processing_time:.2f}ms per frame")
        logger.info(f"Total processing time: {total_processing_time/1000:.2f} seconds")
        logger.info("=========================")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            'frames_processed': self.frame_count,
            'total_events_generated': self.total_events_generated,
            'events_per_frame': self.total_events_generated / self.frame_count if self.frame_count > 0 else 0,
            'threshold_used': self.threshold,
            'multi_scale_enabled': self.enable_multi_scale,
            'motion_boost_enabled': self.motion_boost,
            'noise_level': self.noise_level
        }
    
    def reset(self) -> None:
        """Reset detector state for new video processing"""
        self.prev_frame = None
        self.prev_frame_secondary = None
        self.frame_count = 0
        self.total_events_generated = 0
        logger.info("EventDetector state reset")